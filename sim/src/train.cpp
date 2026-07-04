// Deep CFR data-generation driver: runs MCCFR traversals in parallel worker
// threads whose policy evaluations are coalesced by a shared InferenceBatcher
// (inference.h) into batched backend calls. Mirrors engine.py's run loop
// shape (see engine.cpp for the bot-vs-bot counterpart).
//
// Batching hyperparameters (--workers, --batch-size, --flush-timeout-us) are
// CLI flags so they can be swept like any other hyperparameter once the
// TensorRT backend lands; the batcher stats printed at the end are the tuning
// signal. See inference.h for how the three interact.
//
// Rounds are statically partitioned across workers (worker w runs rounds
// w+1, w+1+W, ...) with per-round-deterministic seeding, so results are
// reproducible for a given (iteration, workers) pair no matter how the
// batcher groups requests. The reservoir buffers' retained sets are the one
// exception: eviction depends on cross-worker push interleaving.
//
// The flywheel: --out dumps the regret/strategy reservoirs as flat binary for
// the Python trainer (train/train_advantage.py), --load resumes them so the
// buffers accumulate across CFR iterations (the dumps carry full reservoir
// state, see buffers.h), and --engine loads the TorchScript advantage net the
// trainer exports (net_policy.h), closing the data-collection -> training ->
// data-collection loop.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "buffers.h"
#include "deck.h"
#include "driver.h"
#include "inference.h"
#include "net_policy.h"
#include "policy.h"
#include "samples.h"
#include "traverse.h"

namespace pkrbot::engine {

struct TrainConfig {
  int numRounds = 100000;
  // CFR iteration t of this data-collection step, supplied by the outer
  // training loop. Recorded on every sample as the linear-CFR weight (Deep
  // CFR, Brown et al.): all traversals in a step run against the same frozen
  // policy, so they all share t.
  int cfrIteration = 1;
  // Run seed, derived from the iteration: any single step is reproducible,
  // while consecutive flywheel steps explore different deals instead of
  // replaying the same shuffles against each new policy.
  uint64_t seed() const { return static_cast<uint64_t>(cfrIteration); }
  // Traversal workers submitting to the batcher. The effective batch size is
  // capped by this (each blocked worker contributes one pending request), so
  // it must be tuned together with batching.maxBatchSize.
  int numWorkers = 8;
  BatchingConfig batching;
  // Reservoir capacity of each sample buffer. When resuming (--load) a
  // saturated reservoir, this must match the loaded file's retained count
  // (see loadSamples in buffers.h).
  size_t bufferSize = DEFAULT_BUFFER_SIZE;
  std::string enginePath;  // empty: UniformPolicy; else TorchScript module
  std::string outDir;      // empty: discard samples; else write regret.bin/strategy.bin here
  // Resume the reservoirs from a previous step's regret.bin/strategy.bin
  // before collecting, so the buffers accumulate across CFR iterations (Deep
  // CFR trains each net on the memory of ALL iterations, not just the latest).
  std::string loadDir;  // empty: start empty buffers
  // Bypass the batcher: workers call the backend directly with n = 1. Only
  // valid for thread-safe, stateless backends (UniformPolicy), so it is
  // rejected together with --engine. Useful as the A/B baseline when tuning
  // batch size, and as the fast path for uniform-policy data generation
  // (per-request cross-thread handoff is pure overhead on a CPU backend).
  bool noBatch = false;
};

// Run one traversal round. `deck` is reshuffled here. Returns the round's
// value for the traverser.
double runRound(Deck& deck, PolicyProvider& policy, ReservoirWriter<RegretSample>& regretWriter,
                ReservoirWriter<StrategySample>& strategyWriter, int traverserSeat, int iteration,
                std::mt19937_64& rng) {
  StateResult state = makeInitialRound(deck);
  ActionHistory history;
  return traverse(state, history, policy, regretWriter, strategyWriter, traverserSeat, iteration,
                  rng);
}

// splitmix64 finalizer: decorrelates per-round seeds derived from
// (run seed, round index) so every round's deck and rng stream is independent
// of which worker runs it.
inline uint64_t mixSeed(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Run this worker's share of the rounds (round indices workerIndex+1,
// workerIndex+1+numWorkers, ...), accumulating the traverser's value per seat
// into `bankroll`. Each round reseeds the deck and rng from (seed, round), so
// the trajectory of round r is identical regardless of worker count or batch
// grouping. Every sample is stamped with config.cfrIteration.
// `batcher` may be null (--no-batch), in which case workers call `backend`
// directly with single-infoset batches.
// Samples are staged in worker-local ReservoirWriters and folded into the
// shared buffers a batch at a time, so workers never serialize on the buffer
// mutexes per sample.
void runWorker(const TrainConfig& config, int workerIndex, InferenceBatcher* batcher,
               PolicyProvider& backend, SampleBuffer<RegretSample>& regretBuffer,
               SampleBuffer<StrategySample>& strategyBuffer, std::array<double, 2>& bankroll) {
  std::unique_ptr<BatchedPolicy> batched;
  if (batcher) batched = std::make_unique<BatchedPolicy>(*batcher);
  PolicyProvider& policy = batcher ? static_cast<PolicyProvider&>(*batched) : backend;

  ReservoirWriter<RegretSample> regretWriter(regretBuffer);
  ReservoirWriter<StrategySample> strategyWriter(strategyBuffer);

  for (int round = workerIndex + 1; round <= config.numRounds; round += config.numWorkers) {
    uint64_t roundSeed = mixSeed(config.seed() * 0x100000001b3ULL + static_cast<uint64_t>(round));
    Deck deck(roundSeed);
    // Offset to decorrelate opponent-action sampling from the deck's stream.
    std::mt19937_64 rng(roundSeed ^ 0x9e3779b97f4a7c15ULL);
    int traverserSeat = round % 2;
    bankroll[traverserSeat] += runRound(deck, policy, regretWriter, strategyWriter, traverserSeat,
                                        config.cfrIteration, rng);
  }
  regretWriter.flush();
  strategyWriter.flush();
  // Required so the batcher stops waiting on this worker when forming
  // batches; without it the remaining workers' batches only flush on timeout.
  if (batcher) batcher->workerFinished();
}

struct TrainResult {
  std::array<double, 2> bankroll = {0, 0};
  BatcherStats batcherStats;
  long long regretSamples = 0;
  long long strategySamples = 0;
  long long regretSeen = 0;
  long long strategySeen = 0;
};

TrainResult runGame(const TrainConfig& config) {
  // Reservoir-eviction seeds derive from the run seed (distinct offsets per
  // buffer); they only affect which samples are retained, never game
  // trajectories.
  SampleBuffer<RegretSample> regretBuffer(config.bufferSize,
                                          config.seed() ^ 0xd1b54a32d192ed03ULL);
  SampleBuffer<StrategySample> strategyBuffer(config.bufferSize,
                                              config.seed() ^ 0x94d049bb133111ebULL);

  if (!config.loadDir.empty()) {
    loadSamples(regretBuffer, config.loadDir + "/regret.bin");
    loadSamples(strategyBuffer, config.loadDir + "/strategy.bin");
  }

  std::unique_ptr<PolicyProvider> backend;
  if (!config.enginePath.empty()) {
    backend = std::make_unique<NetPolicy>(
        loadInferenceEngine(config.enginePath, config.batching.maxBatchSize));
  } else {
    backend = std::make_unique<UniformPolicy>();
  }

  std::unique_ptr<InferenceBatcher> batcher;
  if (!config.noBatch) {
    batcher =
        std::make_unique<InferenceBatcher>(*backend, config.batching, config.numWorkers);
  }

  std::vector<std::array<double, 2>> bankrolls(config.numWorkers, {0, 0});
  std::vector<std::thread> workers;
  workers.reserve(config.numWorkers);
  for (int w = 0; w < config.numWorkers; ++w) {
    workers.emplace_back(runWorker, std::cref(config), w, batcher.get(), std::ref(*backend),
                         std::ref(regretBuffer), std::ref(strategyBuffer),
                         std::ref(bankrolls[w]));
  }
  for (std::thread& worker : workers) worker.join();

  if (!config.outDir.empty()) {
    std::filesystem::create_directories(config.outDir);
    writeSamples(regretBuffer, config.outDir + "/regret.bin");
    writeSamples(strategyBuffer, config.outDir + "/strategy.bin");
  }

  TrainResult result;
  for (const auto& b : bankrolls) {
    result.bankroll[0] += b[0];
    result.bankroll[1] += b[1];
  }
  result.regretSamples = static_cast<long long>(regretBuffer.size());
  result.strategySamples = static_cast<long long>(strategyBuffer.size());
  result.regretSeen = regretBuffer.totalSeen();
  result.strategySeen = strategyBuffer.totalSeen();
  if (batcher) result.batcherStats = batcher->stats();
  return result;
}

}  // namespace pkrbot::engine

using namespace pkrbot::engine;

namespace {

void printUsage(const char* prog) {
  std::cerr << "usage: " << prog << " [options] [numRounds]\n"
            << "  --rounds N            traversal rounds to run (default 100000)\n"
            << "  --iteration T         CFR iteration of this data-collection step; recorded on\n"
            << "                        every sample as the linear-CFR weight and used as the\n"
            << "                        run seed (default 1)\n"
            << "  --workers W           traversal worker threads (default 8)\n"
            << "  --batch-size B        max infosets per inference call (default 256)\n"
            << "  --flush-timeout-us T  partial-batch flush timeout in microseconds (default 250)\n"
            << "  --buffer-size N       reservoir capacity of each sample buffer (default "
            << DEFAULT_BUFFER_SIZE << ")\n"
            << "  --engine PATH         TorchScript advantage net exported by the trainer\n"
            << "                        (default: uniform policy backend)\n"
            << "  --out DIR             write regret.bin / strategy.bin reservoir state to DIR\n"
            << "  --load DIR            resume the reservoirs from DIR's regret.bin /\n"
            << "                        strategy.bin before collecting, accumulating samples\n"
            << "                        across CFR iterations\n"
            << "  --no-batch            bypass the batcher; workers call the backend directly\n"
            << "                        (uniform backend only; baseline for batch-size tuning)\n";
}

bool parseArgs(int argc, char** argv, TrainConfig& config) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::cerr << arg << " requires a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--rounds") {
      const char* v = next();
      if (!v) return false;
      config.numRounds = std::atoi(v);
    } else if (arg == "--iteration") {
      const char* v = next();
      if (!v) return false;
      config.cfrIteration = std::atoi(v);
    } else if (arg == "--workers") {
      const char* v = next();
      if (!v) return false;
      config.numWorkers = std::atoi(v);
    } else if (arg == "--batch-size") {
      const char* v = next();
      if (!v) return false;
      config.batching.maxBatchSize = std::atoi(v);
    } else if (arg == "--flush-timeout-us") {
      const char* v = next();
      if (!v) return false;
      config.batching.flushTimeout = std::chrono::microseconds(std::atoll(v));
    } else if (arg == "--buffer-size") {
      const char* v = next();
      if (!v) return false;
      long long n = std::atoll(v);
      if (n < 1) {
        std::cerr << "--buffer-size must be >= 1\n";
        return false;
      }
      config.bufferSize = static_cast<size_t>(n);
    } else if (arg == "--engine") {
      const char* v = next();
      if (!v) return false;
      config.enginePath = v;
    } else if (arg == "--out") {
      const char* v = next();
      if (!v) return false;
      config.outDir = v;
    } else if (arg == "--load") {
      const char* v = next();
      if (!v) return false;
      config.loadDir = v;
    } else if (arg == "--no-batch") {
      config.noBatch = true;
    } else if (arg == "--help" || arg == "-h") {
      return false;
    } else if (!arg.empty() && arg[0] != '-' && positional < 1) {
      // Legacy positional form: train [numRounds].
      config.numRounds = std::atoi(arg.c_str());
      ++positional;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return false;
    }
  }
  if (config.numRounds < 1 || config.cfrIteration < 1 || config.numWorkers < 1 ||
      config.batching.maxBatchSize < 1 || config.batching.flushTimeout.count() < 0) {
    std::cerr << "rounds, iteration, workers, and batch size must be >= 1; flush timeout must be "
                 ">= 0\n";
    return false;
  }
  if (config.noBatch && !config.enginePath.empty()) {
    std::cerr << "--no-batch requires the uniform backend; the net backend is not "
                 "thread-safe and must run behind the batcher\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  TrainConfig config;
  if (!parseArgs(argc, argv, config)) {
    printUsage(argv[0]);
    return 1;
  }

  auto start = std::chrono::steady_clock::now();
  TrainResult result;
  try {
    result = runGame(config);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  auto end = std::chrono::steady_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  std::cout << "rounds:            " << config.numRounds << "\n"
            << "workers:           " << config.numWorkers << "\n"
            << "traverser seat 0:  " << result.bankroll[0] << "\n"
            << "traverser seat 1:  " << result.bankroll[1] << "\n"
            << "throughput:        " << static_cast<long long>(config.numRounds / secs)
            << " rounds/s\n"
            << "regret samples:    " << result.regretSamples << " retained of "
            << result.regretSeen << " seen\n"
            << "strategy samples:  " << result.strategySamples << " retained of "
            << result.strategySeen << " seen\n";
  if (!config.outDir.empty()) {
    std::cout << "sample dumps:      " << config.outDir << "/{regret,strategy}.bin\n";
  }
  if (!config.noBatch) {
    const BatcherStats& bs = result.batcherStats;
    std::cout << "inference calls:   " << bs.requests << "\n"
              << "batches:           " << bs.batches << " (avg " << bs.avgBatch() << ", max "
              << bs.maxBatch << ", cap " << config.batching.maxBatchSize << ")\n"
              << "flush reasons:     " << bs.fullBatches << " full, " << bs.starvedBatches
              << " starved, " << bs.timeoutBatches << " timeout\n";
  }
  return 0;
}
