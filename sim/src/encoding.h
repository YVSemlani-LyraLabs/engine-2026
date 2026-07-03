#pragma once

// Network input/output contract shared by the C++ inference side and the
// Python training side.
//
// The PyTorch trainer consumes InfoSet PODs from disk (buffers.h) and must
// encode them with a bit-identical Python mirror of encodeInfoSet() below, so
// that a model trained in PyTorch and exported to TensorRT sees the same
// tensor layout at inference time. Bump FEATURE_VERSION whenever the layout
// changes and reject mismatched engines/checkpoints on both sides.
//
// Input tensor: float32 [batch, INPUT_SIZE], laid out as
//   [0..5]    street one-hot over the streets that exist in this variant
//             (0 preflop, 2/3 discards, 4 turn, 5 river-1, 6 river-2)
//   [6..10]   scalars: own pip, opp pip, own stack, opp stack (each divided
//             by STARTING_STACK) and pot (divided by 2*STARTING_STACK)
//   [11..]    3 hole-card one-hots over 52 (discarded/absent card = all zero)
//   then      7 board-card one-hots over 52 (hidden card = all zero)
//   then      legal-action mask, NUM_ABSTRACT_ACTIONS floats
//   then      MAX_ACTIONS action-history slots, 2 + NUM_ABSTRACT_ACTIONS
//             floats each: [occupied, actor-is-me, abstract-action one-hot]
//
// Output tensor: float32 [batch, NUM_ABSTRACT_ACTIONS]. For the advantage
// (regret) net these are predicted advantages; policyFromAdvantages() turns
// them into a policy by regret matching. The average-strategy net emits
// logits over legal actions instead and is decoded by masked softmax (to be
// added alongside that net).

#include <array>

#include "abstraction.h"
#include "infoset.h"
#include "state.h"

namespace pkrbot::engine {

inline constexpr int FEATURE_VERSION = 1;

inline constexpr int NUM_CARDS = 52;
inline constexpr int NUM_STREETS = 6;  // 0, 2, 3, 4, 5, 6
inline constexpr int NUM_SCALAR_FEATURES = 5;
inline constexpr int HISTORY_SLOT_SIZE = 2 + NUM_ABSTRACT_ACTIONS;
inline constexpr int INPUT_SIZE = NUM_STREETS + NUM_SCALAR_FEATURES +
                                  (3 + 7) * NUM_CARDS + NUM_ABSTRACT_ACTIONS +
                                  MAX_ACTIONS * HISTORY_SLOT_SIZE;

inline int streetIndex(int street) {
  // Streets take values {0, 2, 3, 4, 5, 6}; pack them densely.
  return street == 0 ? 0 : street - 1;
}

// Write exactly INPUT_SIZE floats for one infoset into out.
inline void encodeInfoSet(const InfoSet& obs, float* out) {
  float* p = out;

  for (int i = 0; i < NUM_STREETS; ++i) p[i] = 0.0f;
  p[streetIndex(obs.street)] = 1.0f;
  p += NUM_STREETS;

  *p++ = static_cast<float>(obs.pips[0]) / STARTING_STACK;
  *p++ = static_cast<float>(obs.pips[1]) / STARTING_STACK;
  *p++ = static_cast<float>(obs.stacks[0]) / STARTING_STACK;
  *p++ = static_cast<float>(obs.stacks[1]) / STARTING_STACK;
  int pot = (STARTING_STACK - obs.stacks[0]) + (STARTING_STACK - obs.stacks[1]) + obs.pips[0] +
            obs.pips[1];
  *p++ = static_cast<float>(pot) / (2 * STARTING_STACK);

  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < NUM_CARDS; ++i) p[i] = 0.0f;
    if (obs.hole[c] >= 0) p[obs.hole[c]] = 1.0f;
    p += NUM_CARDS;
  }
  for (int c = 0; c < 7; ++c) {
    for (int i = 0; i < NUM_CARDS; ++i) p[i] = 0.0f;
    if (obs.board[c] >= 0) p[obs.board[c]] = 1.0f;
    p += NUM_CARDS;
  }

  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; ++i) {
    *p++ = static_cast<float>(obs.legalActionMask[i]);
  }

  for (int slot = 0; slot < MAX_ACTIONS; ++slot) {
    for (int i = 0; i < HISTORY_SLOT_SIZE; ++i) p[i] = 0.0f;
    if (slot < obs.actionHistory.tail) {
      const ActionRecord& record = obs.actionHistory.actionRecords[slot];
      p[0] = 1.0f;
      p[1] = record.player == obs.player ? 1.0f : 0.0f;
      p[2 + static_cast<int>(record.action)] = 1.0f;
    }
    p += HISTORY_SLOT_SIZE;
  }
}

// Regret matching over predicted advantages (Deep CFR, Brown et al.): the
// policy is proportional to the positive advantages of legal actions. If no
// legal action has positive advantage, all mass goes to the legal action with
// the highest advantage. Illegal actions always get probability 0, per the
// PolicyProvider contract.
inline PolicyVector policyFromAdvantages(const float* advantages,
                                         const std::array<int, NUM_ABSTRACT_ACTIONS>& legalMask) {
  PolicyVector policy{};
  double positiveSum = 0.0;
  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; ++i) {
    if (legalMask[i] == 1 && advantages[i] > 0.0f) {
      positiveSum += advantages[i];
    }
  }
  if (positiveSum > 0.0) {
    for (int i = 0; i < NUM_ABSTRACT_ACTIONS; ++i) {
      if (legalMask[i] == 1 && advantages[i] > 0.0f) {
        policy[i] = advantages[i] / positiveSum;
      }
    }
    return policy;
  }

  int best = -1;
  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; ++i) {
    if (legalMask[i] == 1 && (best == -1 || advantages[i] > advantages[best])) {
      best = i;
    }
  }
  if (best >= 0) policy[best] = 1.0;
  return policy;
}

}  // namespace pkrbot::engine
