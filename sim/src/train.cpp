// Deep CFR data-generation driver: runs MCCFR traversals round after round,
// alternating the traverser seat, and fills the regret/strategy sample
// buffers. Mirrors engine.py's run loop shape (see engine.cpp for the
// bot-vs-bot counterpart).
//
// Pass 2 hooks: flush buffers through a real SampleSink (buffers.h) and swap
// UniformPolicy for network inference (policy.h).

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

#include "buffers.h"
#include "deck.h"
#include "driver.h"
#include "policy.h"
#include "samples.h"
#include "traverse.h"

namespace pkrbot::engine {

// Run one traversal round. `deck` is reshuffled here. Returns the round's
// value for the traverser.
double runRound(Deck& deck, PolicyProvider& policy, SampleBuffer<RegretSample>& regretBuffer,
                SampleBuffer<StrategySample>& strategyBuffer, int traverserSeat, int iteration,
                std::mt19937_64& rng) {
  StateResult state = makeInitialRound(deck);
  ActionHistory history;
  return traverse(state, history, policy, regretBuffer, strategyBuffer, traverserSeat, iteration,
                  rng);
}

// Run traversals for numRounds rounds, alternating the traverser seat each
// round. Returns the traverser's accumulated value per seat it occupied.
std::array<double, 2> runGame(int numRounds, uint64_t seed) {
  Deck deck(seed);
  std::array<double, 2> bankroll = {0, 0};
  UniformPolicy policy;
  // Buffers are large (BUFFER_SIZE * sizeof(sample) each), so keep them off
  // the stack.
  auto regretBuffer = std::make_unique<SampleBuffer<RegretSample>>();
  auto strategyBuffer = std::make_unique<SampleBuffer<StrategySample>>();

  // Derived from the run seed so opponent-action sampling (and thus the whole
  // run) is reproducible; offset to decorrelate from the deck's stream.
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
  std::uniform_int_distribution<int> distrib(0, 1);
  int traverserSeat = distrib(rng);

  for (int round = 1; round <= numRounds; ++round) {
    bankroll[traverserSeat] +=
        runRound(deck, policy, *regretBuffer, *strategyBuffer, traverserSeat, round, rng);
    traverserSeat = 1 - traverserSeat;
  }
  return bankroll;
}

}  // namespace pkrbot::engine

using namespace pkrbot::engine;

int main(int argc, char** argv) {
  int numRounds = argc > 1 ? std::atoi(argv[1]) : 100000;
  uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;

  auto start = std::chrono::steady_clock::now();
  auto bankroll = runGame(numRounds, seed);
  auto end = std::chrono::steady_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  std::cout << "rounds:     " << numRounds << "\n"
            << "traverser seat 0:  " << bankroll[0] << "\n"
            << "traverser seat 1:   " << bankroll[1] << "\n"
            << "throughput: " << static_cast<long long>(numRounds / secs)
            << " rounds/s\n";
  return 0;
}
