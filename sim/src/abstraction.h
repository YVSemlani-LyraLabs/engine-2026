#pragma once

// Action abstraction for Deep CFR: the engine's concrete action space is
// mapped onto a fixed set of NUM_ABSTRACT_ACTIONS abstract actions so that
// infosets, samples, and network outputs all share one indexing scheme:
//   [0] Fold, [1] Call, [2] Check, [3..5] Discard0..2,
//   [6] RaiseMin, [7] RaiseHalfPot, [8] RaisePot, [9] RaiseMax.
//
// Base legality (which action *types* are allowed) is owned by
// RoundState::legalActions() in state.h; this module only layers the
// abstraction-specific raise sizing on top.

#include <algorithm>
#include <array>
#include <stdexcept>

#include "state.h"

namespace pkrbot::engine {

inline constexpr int NUM_ABSTRACT_ACTIONS = 10;

enum class AbstractAction {
  Fold,
  Call,
  Check,
  Discard0,
  Discard1,
  Discard2,
  RaiseMin,
  RaiseHalfPot,
  RaisePot,
  RaiseMax,
};

// Get the legal abstract actions for a given state.
// ------------------------------------------------------------
// TO-DO: add filters to prevent raise sizes from overlapping with each other
// ------------------------------------------------------------
// and splitting probability mass i.e. raise 1/2 pot == raise Max == raise Pot
// (b/c they get clamped to the max raise)
inline std::array<int, NUM_ABSTRACT_ACTIONS> legalAbstractActions(const RoundState& state) {
  LegalActions legal = state.legalActions();
  std::array<int, NUM_ABSTRACT_ACTIONS> mask{};

  if (legal.discard) {
    mask[static_cast<int>(AbstractAction::Discard0)] = 1;
    mask[static_cast<int>(AbstractAction::Discard1)] = 1;
    mask[static_cast<int>(AbstractAction::Discard2)] = 1;
    return mask;  // discard turns are discard-only
  }

  mask[static_cast<int>(AbstractAction::Fold)] = legal.fold;
  mask[static_cast<int>(AbstractAction::Call)] = legal.call;
  mask[static_cast<int>(AbstractAction::Check)] = legal.check;

  if (legal.raise) {
    int active = state.active();
    int continueCost = state.pips[1 - active] - state.pips[active];
    int pot = (STARTING_STACK - state.stacks[0]) + (STARTING_STACK - state.stacks[1]);
    auto [minRaise, maxRaise] = state.raiseBounds();
    (void)minRaise;

    mask[static_cast<int>(AbstractAction::RaiseMin)] = 1;
    mask[static_cast<int>(AbstractAction::RaiseMax)] = 1;
    if (state.pips[active] + continueCost + (pot / 2) <= maxRaise) {
      mask[static_cast<int>(AbstractAction::RaiseHalfPot)] = 1;
    }
    if (state.pips[active] + continueCost + pot <= maxRaise) {
      mask[static_cast<int>(AbstractAction::RaisePot)] = 1;
    }
  }

  return mask;
}

// Turn an abstract raise action into a concrete action.
inline Action concretizeRaise(const RoundState& state, AbstractAction abstractAction) {
  auto [minRaise, maxRaise] = state.raiseBounds();
  int active = state.active();

  int continueCost = state.pips[1 - active] - state.pips[active];
  int pot = (STARTING_STACK - state.stacks[0]) + (STARTING_STACK - state.stacks[1]);
  int target = minRaise;

  switch (abstractAction) {
    case AbstractAction::RaiseMin:
      target = minRaise;
      break;
    case AbstractAction::RaiseHalfPot:
      target = state.pips[active] + continueCost + (pot / 2);
      break;
    case AbstractAction::RaisePot:
      target = state.pips[active] + continueCost + pot;
      break;
    case AbstractAction::RaiseMax:
      target = maxRaise;
      break;
    default:
      throw std::invalid_argument("Invalid abstract action");
  }

  target = std::max(minRaise, std::min(target, maxRaise));
  return Action::raise(target);
}

// Turn an abstract action into a concrete action.
inline Action concretizeAction(const RoundState& state, AbstractAction abstractAction) {
  switch (abstractAction) {
    case AbstractAction::Fold:
      return Action::fold();
    case AbstractAction::Call:
      return Action::call();
    case AbstractAction::Check:
      return Action::check();
    case AbstractAction::Discard0:
      return Action::discard(0);
    case AbstractAction::Discard1:
      return Action::discard(1);
    case AbstractAction::Discard2:
      return Action::discard(2);
    case AbstractAction::RaiseMin:
    case AbstractAction::RaiseHalfPot:
    case AbstractAction::RaisePot:
    case AbstractAction::RaiseMax:
      return concretizeRaise(state, abstractAction);
    default:
      throw std::invalid_argument("Invalid abstract action");
  }
}

}  // namespace pkrbot::engine
