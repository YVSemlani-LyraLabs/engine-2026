#pragma once

// In-process port of engine.py's RoundState game tree (Option B simulator).
//
// Differences from engine.py, by design:
//   * Value semantics instead of shared mutable lists + previous_state chains.
//     proceed()/proceedStreet() return a fresh state by value; the caller keeps
//     the prior state, so terminal states only need to carry deltas.
//   * The Deck is dealt once at round setup and only peeked during the round,
//     so RoundState borrows it as `const Deck*` (cursor stays at 6).
//   * Stacks are integral here, so get_delta's fractional floor/ceil-by-button
//     rounding collapses to a plain integer delta.

#include <array>
#include <cassert>
#include <cstdint>
#include <variant>
#include <vector>

#include "deck.h"
#include "eval.h"

namespace pkrbot::engine {

inline constexpr int STARTING_STACK = 400;
inline constexpr int BIG_BLIND = 2;
inline constexpr int SMALL_BLIND = 1;

struct Action {
  enum class Type { Fold, Call, Check, Raise, Discard };
  Type type;
  int amount = 0;  // Raise: target pip amount. Discard: hole-card index 0..2.

  static Action fold() { return {Type::Fold, 0}; }
  static Action call() { return {Type::Call, 0}; }
  static Action check() { return {Type::Check, 0}; }
  static Action raise(int amount) { return {Type::Raise, amount}; }
  static Action discard(int card) { return {Type::Discard, card}; }
};

struct LegalActions {
  bool fold = false;
  bool call = false;
  bool check = false;
  bool raise = false;
  bool discard = false;

  bool has(Action::Type t) const {
    switch (t) {
      case Action::Type::Fold: return fold;
      case Action::Type::Call: return call;
      case Action::Type::Check: return check;
      case Action::Type::Raise: return raise;
      case Action::Type::Discard: return discard;
    }
    return false;
  }
};

struct RoundState;
struct TerminalState;
using StateResult = std::variant<RoundState, TerminalState>;

struct TerminalState {
  std::array<int, 2> deltas;
};

struct RoundState {
  int button;
  int street;
  std::array<int, 2> pips;
  std::array<int, 2> stacks;
  std::array<std::vector<Card>, 2> hands;
  std::vector<Card> board;
  const Deck* deck;

  int active() const { return button % 2; }

  // Delta for player A (-delta for B). winner: 0 = A, 1 = B, 2 = split.
  int getDelta(int winner) const {
    assert(winner == 0 || winner == 1 || winner == 2);
    if (winner == 2) {
      assert(stacks[0] == stacks[1]);  // split only on equal stacks
      return 0;
    }
    return winner == 0 ? STARTING_STACK - stacks[1] : stacks[0] - STARTING_STACK;
  }

  TerminalState showdown() const {
    assert(stacks[0] == stacks[1]);
    uint32_t score0 = evalHand(0);
    uint32_t score1 = evalHand(1);
    int delta = score0 > score1   ? getDelta(0)
                : score0 < score1 ? getDelta(1)
                                  : getDelta(2);
    return TerminalState{{delta, -delta}};
  }

  LegalActions legalActions() const {
    int a = active();
    int continueCost = pips[1 - a] - pips[a];
    LegalActions la;
    if (street == 2 || street == 3) {
      if (a != street % 2)
        la.discard = true;
      else
        la.check = true;
      return la;
    }
    if (continueCost == 0) {
      bool betsForbidden = (stacks[0] == 0 || stacks[1] == 0);
      la.check = true;
      la.fold = true;
      la.raise = !betsForbidden;
      return la;
    }
    bool raisesForbidden = (continueCost == stacks[a] || stacks[1 - a] == 0);
    la.fold = true;
    la.call = true;
    la.raise = !raisesForbidden;
    return la;
  }

  // Returns {min, max} legal raise target pips.
  std::array<int, 2> raiseBounds() const {
    int a = active();
    int continueCost = pips[1 - a] - pips[a];
    int maxContribution = std::min(stacks[a], stacks[1 - a] + continueCost);
    int minContribution =
        std::min(maxContribution, continueCost + std::max(continueCost, BIG_BLIND));
    return {pips[a] + minContribution, pips[a] + maxContribution};
  }

  // Reset pips, advance street, and reveal board cards. Streets: 0,2,3,4,5,6.
  StateResult proceedStreet() const {
    if (street == 6) return showdown();

    std::vector<Card> newBoard = board;
    int newStreet;
    int newButton;
    if (street == 0) {
      newStreet = 2;
      newButton = 1;  // B discards first (out of position)
      auto reveal = deck->peek(2);
      newBoard.insert(newBoard.end(), reveal.begin(), reveal.end());
    } else if (street == 2) {
      newStreet = 3;
      newButton = 0;  // A discards second
    } else if (street == 3) {
      newStreet = 4;
      newButton = 1;  // B acts first after discards
    } else {
      newStreet = street + 1;
      newButton = 1;
      newBoard.push_back(deck->peek(newStreet - 1)[newStreet - 2]);
    }

    return RoundState{newButton, newStreet, {0, 0}, stacks, hands, std::move(newBoard), deck};
  }

  StateResult proceed(Action action) const {
    int a = active();
    switch (action.type) {
      case Action::Type::Discard: {
        auto newHands = hands;
        std::vector<Card> newBoard = board;
        if (!newHands[a].empty()) {
          Card discarded = newHands[a][action.amount];
          newHands[a].erase(newHands[a].begin() + action.amount);
          newBoard.push_back(discarded);
        }
        return RoundState{(1 - a) % 2, street, pips, stacks, std::move(newHands),
                          std::move(newBoard), deck};
      }
      case Action::Type::Fold: {
        int delta = getDelta((1 - a) % 2);  // folder loses; opponent wins
        return TerminalState{{delta, -delta}};
      }
      case Action::Type::Call: {
        if (button == 0) {  // small blind calls big blind
          return RoundState{1,
                            0,
                            {BIG_BLIND, BIG_BLIND},
                            {STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND},
                            hands,
                            board,
                            deck};
        }
        auto newPips = pips;
        auto newStacks = stacks;
        int contribution = newPips[1 - a] - newPips[a];
        newStacks[a] -= contribution;
        newPips[a] += contribution;
        RoundState next{button + 1, street, newPips, newStacks, hands, board, deck};
        return next.proceedStreet();
      }
      case Action::Type::Check: {
        bool bothActed = (street == 0 && button > 0) || button > 1 || street == 2 || street == 3;
        if (bothActed) return proceedStreet();
        return RoundState{button + 1, street, pips, stacks, hands, board, deck};
      }
      case Action::Type::Raise: {
        auto newPips = pips;
        auto newStacks = stacks;
        int contribution = action.amount - newPips[a];
        newStacks[a] -= contribution;
        newPips[a] += contribution;
        return RoundState{button + 1, street, newPips, newStacks, hands, board, deck};
      }
    }
    __builtin_unreachable();
  }

 private:
  uint32_t evalHand(int player) const {
    std::array<uint8_t, 16> ranks{};
    std::array<uint8_t, 16> suits{};
    int n = 0;
    for (const Card& c : board) {
      ranks[n] = c.rank;
      suits[n] = c.suit;
      ++n;
    }
    for (const Card& c : hands[player]) {
      ranks[n] = c.rank;
      suits[n] = c.suit;
      ++n;
    }
    return evaluate(ranks.data(), suits.data(), n);
  }
};

}  // namespace pkrbot::engine
