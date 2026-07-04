// Tests for the inference batching layer (inference.h) and the net-policy
// connective layer (net_policy.h, with a mock InferenceEngine). Build target:
// batch_test. Exits 0 and prints "all tests passed" on success.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "encoding.h"
#include "inference.h"
#include "infoset.h"
#include "net_policy.h"
#include "policy.h"

// Not <cassert>'s assert: that is compiled out by -DNDEBUG in Release builds,
// which is how this target is normally built.
#define REQUIRE(cond)                                                                       \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::cerr << "REQUIRE failed at " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

namespace pkrbot::engine {
namespace {

InfoSet makeTestInfoSet(int tag) {
  InfoSet obs{};
  obs.player = tag % 2;
  obs.street = 0;
  obs.pips = {SMALL_BLIND, BIG_BLIND};
  obs.stacks = {STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND};
  obs.board.fill(-1);
  obs.hole = {tag % 52, (tag + 1) % 52, (tag + 2) % 52};
  obs.legalActionMask.fill(0);
  obs.legalActionMask[static_cast<int>(AbstractAction::Fold)] = 1;
  obs.legalActionMask[static_cast<int>(AbstractAction::Call)] = 1;
  return obs;
}

// Encodes each infoset's identity into its policy so tests can verify that
// batched results are routed back to the right caller. Also records batch
// sizes.
struct TaggingBackend : PolicyProvider {
  std::vector<size_t> batchSizes;

  void evaluate(const InfoSet* obs, size_t n, PolicyVector* out) override {
    batchSizes.push_back(n);
    for (size_t i = 0; i < n; ++i) {
      out[i] = {};
      out[i][0] = static_cast<double>(obs[i].hole[0]);
      out[i][1] = static_cast<double>(obs[i].hole[1]);
    }
  }
};

// Every worker's result must match its own infoset even when requests from
// many workers are coalesced into shared batches.
void testRoutingUnderConcurrency() {
  constexpr int kWorkers = 16;
  constexpr int kRequestsPerWorker = 200;

  TaggingBackend backend;
  BatchingConfig config;
  config.maxBatchSize = 8;
  config.flushTimeout = std::chrono::microseconds(500);
  InferenceBatcher batcher(backend, config, kWorkers);

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int w = 0; w < kWorkers; ++w) {
    workers.emplace_back([&, w] {
      for (int r = 0; r < kRequestsPerWorker; ++r) {
        int tag = w * kRequestsPerWorker + r;
        InfoSet obs = makeTestInfoSet(tag);
        PolicyVector result = batcher.evaluate(obs);
        if (result[0] != obs.hole[0] || result[1] != obs.hole[1]) failures.fetch_add(1);
      }
      batcher.workerFinished();
    });
  }
  for (auto& t : workers) t.join();

  REQUIRE(failures.load() == 0);

  BatcherStats stats = batcher.stats();
  REQUIRE(stats.requests == kWorkers * kRequestsPerWorker);
  REQUIRE(stats.maxBatch <= config.maxBatchSize);
  long long total = 0;
  for (size_t n : backend.batchSizes) {
    REQUIRE(n >= 1 && n <= static_cast<size_t>(config.maxBatchSize));
    total += static_cast<long long>(n);
  }
  REQUIRE(total == stats.requests);
  REQUIRE(static_cast<long long>(backend.batchSizes.size()) == stats.batches);
}

// With workers held at the barrier until all have submitted, the batcher must
// coalesce: full batches of maxBatchSize, not per-request singletons.
void testCoalescing() {
  constexpr int kWorkers = 8;
  TaggingBackend backend;
  BatchingConfig config;
  config.maxBatchSize = kWorkers;
  config.flushTimeout = std::chrono::milliseconds(100);
  InferenceBatcher batcher(backend, config, kWorkers);

  std::vector<std::thread> workers;
  for (int w = 0; w < kWorkers; ++w) {
    workers.emplace_back([&, w] {
      InfoSet obs = makeTestInfoSet(w);
      PolicyVector result = batcher.evaluate(obs);
      REQUIRE(result[0] == obs.hole[0]);
      batcher.workerFinished();
    });
  }
  for (auto& t : workers) t.join();

  // All 8 workers block on their single request, so the batcher must have
  // dispatched exactly one full batch (all-workers-waiting == full here).
  REQUIRE(backend.batchSizes.size() == 1);
  REQUIRE(backend.batchSizes[0] == static_cast<size_t>(kWorkers));
  REQUIRE(batcher.stats().fullBatches == 1);
}

// Identity-weight mock: emits a fixed advantage pattern so the decode path
// (regret matching) is exercised end to end through NetPolicy.
struct MockEngine : InferenceEngine {
  int inputSize() const override { return INPUT_SIZE; }
  int outputSize() const override { return NUM_ABSTRACT_ACTIONS; }
  int maxBatchSize() const override { return 4; }

  void infer(const float* input, float* output, int batch) override {
    for (int b = 0; b < batch; ++b) {
      const float* row = input + static_cast<size_t>(b) * INPUT_SIZE;
      // The first hole-card one-hot starts after street + scalars; recover
      // the card index to prove the encoder ran.
      int hole0 = -1;
      for (int c = 0; c < NUM_CARDS; ++c) {
        if (row[NUM_STREETS + NUM_SCALAR_FEATURES + c] == 1.0f) hole0 = c;
      }
      float* out = output + static_cast<size_t>(b) * NUM_ABSTRACT_ACTIONS;
      for (int i = 0; i < NUM_ABSTRACT_ACTIONS; ++i) out[i] = -1.0f;
      out[static_cast<int>(AbstractAction::Fold)] = 1.0f;
      out[static_cast<int>(AbstractAction::Call)] = static_cast<float>(hole0 >= 26 ? 3.0 : 1.0);
    }
  }
};

void testNetPolicyDecode() {
  NetPolicy policy(std::make_unique<MockEngine>());

  // 6 infosets with a 4-row engine limit also exercises the chunked path.
  std::vector<InfoSet> obs;
  for (int tag : {0, 30, 1, 40, 2, 50}) obs.push_back(makeTestInfoSet(tag));
  std::vector<PolicyVector> out(obs.size());
  policy.evaluate(obs.data(), obs.size(), out.data());

  for (size_t i = 0; i < obs.size(); ++i) {
    double expectCall = obs[i].hole[0] >= 26 ? 3.0 / 4.0 : 1.0 / 2.0;
    REQUIRE(std::abs(out[i][static_cast<int>(AbstractAction::Call)] - expectCall) < 1e-9);
    REQUIRE(std::abs(out[i][static_cast<int>(AbstractAction::Fold)] - (1.0 - expectCall)) < 1e-9);
    // Illegal actions must be 0 even though the engine emitted advantages.
    for (int a = 2; a < NUM_ABSTRACT_ACTIONS; ++a) REQUIRE(out[i][a] == 0.0);
  }
}

void testPolicyFromAdvantages() {
  std::array<int, NUM_ABSTRACT_ACTIONS> mask{};
  mask[0] = mask[1] = mask[2] = 1;

  // All advantages non-positive: all mass on the best legal action.
  float allNeg[NUM_ABSTRACT_ACTIONS] = {-3.0f, -1.0f, -2.0f};
  PolicyVector p = policyFromAdvantages(allNeg, mask);
  REQUIRE(p[1] == 1.0 && p[0] == 0.0 && p[2] == 0.0);

  // Mixed: proportional to positive advantages only.
  float mixed[NUM_ABSTRACT_ACTIONS] = {2.0f, -1.0f, 6.0f};
  p = policyFromAdvantages(mixed, mask);
  REQUIRE(std::abs(p[0] - 0.25) < 1e-9 && p[1] == 0.0 && std::abs(p[2] - 0.75) < 1e-9);

  // A positive advantage on an illegal action must be ignored.
  float illegalPos[NUM_ABSTRACT_ACTIONS] = {1.0f, 0.0f, 0.0f, 5.0f};
  p = policyFromAdvantages(illegalPos, mask);
  REQUIRE(p[0] == 1.0 && p[3] == 0.0);
}

}  // namespace
}  // namespace pkrbot::engine

int main() {
  using namespace pkrbot::engine;
  testPolicyFromAdvantages();
  testNetPolicyDecode();
  testCoalescing();
  testRoutingUnderConcurrency();
  std::cout << "all tests passed\n";
  return 0;
}
