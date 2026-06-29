// In-process round/game driver for the simulator (Option B).
//
// No sockets, subprocesses, game clock, or gamelog: bots are linked in and
// called directly. Mirrors engine.py's run_round/run loop, including the
// per-round seat swap (engine.py's players[::-1]).

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <memory>

#include "bot.h"
#include "deck.h"
#include "state.h"
#include "traversal.h"

namespace pkrbot::engine {

// Play one round. `deck` is reshuffled here; seat0/seat1 are the bots sitting in
// seats 0 and 1 for this round. Returns {delta_seat0, delta_seat1}.
std::array<double, 2> runRound(Deck& deck, RegretBuffer& regretBuffer, StrategyBuffer& strategyBuffer, int traverserSeat) {
  deck.reset();
  deck.shuffle();
  std::array<std::vector<Card>, 2> hands = {deck.deal(3), deck.deal(3)};
  RoundState start{0,
                   0,
                   {SMALL_BLIND, BIG_BLIND},
                   {STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND},
                   std::move(hands),
                   {},
                   &deck};

  std::array<double, 2> deltas = {0, 0};
  StateResult state = std::move(start);
  ActionHistory history;
  double traverserValue = traverse(state, history, regretBuffer, strategyBuffer, traverserSeat);
  deltas[traverserSeat] = traverserValue;
  deltas[1 - traverserSeat] = -traverserValue;
  return deltas;
}

// Play a full game, swapping seats each round. Returns bankrolls keyed by the traverser seat.
std::array<double, 2> runGame(int numRounds, uint64_t seed) {
  Deck deck(seed);
  std::array<double, 2> bankroll = {0, 0};
  auto regretBuffer = std::make_unique<RegretBuffer>();
  auto strategyBuffer = std::make_unique<StrategyBuffer>();

  // Seed with a real hardware random value if available
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<int> distrib(0, 1); 

  // Generate random 0 or 1
  int traverserSeat = distrib(gen);

  for (int round = 1; round <= numRounds; ++round) {
    std::array<double, 2> deltas = runRound(deck, *regretBuffer, *strategyBuffer, traverserSeat);
    bankroll[traverserSeat] += deltas[traverserSeat];
    bankroll[1 - traverserSeat] += deltas[1 - traverserSeat];
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