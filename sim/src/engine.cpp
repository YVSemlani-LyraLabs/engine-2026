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

#include "bot.h"
#include "deck.h"
#include "driver.h"
#include "state.h"

namespace pkrbot::engine {

// Coerce a bot's action to a legal one, matching engine.py: an illegal action
// (wrong type, out-of-bounds raise, bad discard index) falls back to Check if
// legal, else Fold.
Action sanitize(const RoundState& rs, Action action) {
  LegalActions legal = rs.legalActions();
  if (legal.has(action.type)) {
    if (action.type == Action::Type::Raise) {
      auto [lo, hi] = rs.raiseBounds();
      if (lo <= action.amount && action.amount <= hi) return action;
    } else if (action.type == Action::Type::Discard) {
      if (action.amount >= 0 && action.amount <= 2) return action;
    } else {
      return action;
    }
  }
  return legal.check ? Action::check() : Action::fold();
}

// Play one round. `deck` is reshuffled here; seat0/seat1 are the bots sitting in
// seats 0 and 1 for this round. Returns {delta_seat0, delta_seat1}.
std::array<int, 2> runRound(Deck& deck, Bot& seat0, Bot& seat1) {
  Bot* seats[2] = {&seat0, &seat1};
  StateResult state = makeInitialRound(deck);
  while (std::holds_alternative<RoundState>(state)) {
    const RoundState& cur = std::get<RoundState>(state);
    int seat = cur.active();
    Action action = sanitize(cur, seats[seat]->getAction(cur, seat));
    state = cur.proceed(action);
  }

  TerminalState terminal = std::get<TerminalState>(state);
  seat0.onRoundEnd(terminal, 0);
  seat1.onRoundEnd(terminal, 1);
  return terminal.deltas;
}

// Play a full game, swapping seats each round. Returns bankrolls keyed by the
// stable bot id: {bankroll_botA, bankroll_botB}.
std::array<long long, 2> runGame(Bot& botA, Bot& botB, int numRounds, uint64_t seed) {
  Deck deck(seed);
  Bot* bots[2] = {&botA, &botB};
  std::array<long long, 2> bankroll = {0, 0};

  for (int round = 1; round <= numRounds; ++round) {
    int seat0 = (round % 2 == 1) ? 0 : 1;  // botA starts in seat 0, then alternate
    int seat1 = 1 - seat0;
    std::array<int, 2> deltas = runRound(deck, *bots[seat0], *bots[seat1]);
    bankroll[seat0] += deltas[0];
    bankroll[seat1] += deltas[1];
  }
  return bankroll;
}

}  // namespace pkrbot::engine

namespace {

using namespace pkrbot::engine;

// Passive bot: discards its first card when required, otherwise checks/calls.
struct CheckCallBot : Bot {
  Action getAction(const RoundState& state, int seat) override {
    (void)seat;
    LegalActions la = state.legalActions();
    if (la.discard) return Action::discard(0);
    if (la.check) return Action::check();
    if (la.call) return Action::call();
    return Action::fold();
  }
};

// Aggressive bot: raises the minimum when allowed, otherwise calls/checks.
struct MinRaiseBot : Bot {
  Action getAction(const RoundState& state, int seat) override {
    (void)seat;
    LegalActions la = state.legalActions();
    if (la.discard) return Action::discard(0);
    if (la.raise) return Action::raise(state.raiseBounds()[0]);
    if (la.call) return Action::call();
    if (la.check) return Action::check();
    return Action::fold();
  }
};

}  // namespace

int main(int argc, char** argv) {
  int numRounds = argc > 1 ? std::atoi(argv[1]) : 100000;
  uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;

  MinRaiseBot botA;
  CheckCallBot botB;

  auto start = std::chrono::steady_clock::now();
  auto bankroll = runGame(botA, botB, numRounds, seed);
  auto end = std::chrono::steady_clock::now();

  double secs = std::chrono::duration<double>(end - start).count();
  std::cout << "rounds:     " << numRounds << "\n"
            << "MinRaiseBot:  " << bankroll[0] << "\n"
            << "CheckCallBot:   " << bankroll[1] << "\n"
            << "throughput: " << static_cast<long long>(numRounds / secs)
            << " rounds/s\n";
  return 0;
}
