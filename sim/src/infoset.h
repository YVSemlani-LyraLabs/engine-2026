#pragma once

// Fixed-size, trivially-copyable encoding of what a player can observe at a
// decision point: public state, private hole cards, and the action history so
// far. These structs are the unit that gets stored in sample buffers and
// eventually written out to disk for the Python/PyTorch trainer, so they must
// stay PODs (see the static_asserts below).

#include <array>
#include <stdexcept>
#include <type_traits>

#include "abstraction.h"
#include "state.h"

namespace pkrbot::engine {

constexpr int MAX_ACTIONS = 64;

struct ActionRecord {
  int player;
  int street;
  AbstractAction action;
  Action concreteAction;
};

struct ActionHistory {
  std::array<ActionRecord, MAX_ACTIONS> actionRecords;
  int tail = 0;

  void push(const ActionRecord& record) {
    if (tail >= MAX_ACTIONS) {
      throw std::runtime_error("Action history is full");
      // should never happen. if triggered we should increase the max actions
    }
    actionRecords[tail] = record;
    tail++;
  }

  void pop() {
    if (tail == 0) {
      throw std::runtime_error("Action history is empty");
      // should never happen. if triggered we should increase the max actions
    }
    tail--;
  }

  // Zero out entries past `tail` so removed actions never leak into samples
  // (and later into the on-disk format).
  void clearDeadHistory() {
    for (int i = tail; i < MAX_ACTIONS; i++) {
      actionRecords[i] = ActionRecord();
    }
  }
};

struct InfoSet {
  int player;
  int street;
  std::array<int, 2> pips;
  std::array<int, 2> stacks;
  std::array<int, 7> board;
  std::array<int, 3> hole;

  ActionHistory actionHistory;
  std::array<int, NUM_ABSTRACT_ACTIONS> legalActionMask;
};

static_assert(std::is_trivially_copyable_v<ActionRecord>);
static_assert(std::is_trivially_copyable_v<ActionHistory>);
static_assert(std::is_trivially_copyable_v<InfoSet>);

// Make an info set from a given state and action history.
inline InfoSet makeInfoSet(const RoundState& state, const ActionHistory& history) {
  InfoSet obs;

  obs.player = state.active();
  obs.street = state.street;
  obs.pips[0] = state.pips[obs.player];
  obs.pips[1] = state.pips[1 - obs.player];
  obs.stacks[0] = state.stacks[obs.player];
  obs.stacks[1] = state.stacks[1 - obs.player];

  for (int i = 0; i < 7; i++) {
    if (i < static_cast<int>(state.board.size())) {
      obs.board[i] = state.board[i].code();
    } else {
      obs.board[i] = -1;  // hidden board card
    }
  }

  for (int i = 0; i < 3; i++) {
    if (i < static_cast<int>(state.hands[obs.player].size())) {
      obs.hole[i] = state.hands[obs.player][i].code();
    } else {
      obs.hole[i] = -1;  // discarded hole card
    }
  }
  obs.actionHistory = history;
  obs.legalActionMask = legalAbstractActions(state);
  return obs;
}

}  // namespace pkrbot::engine
