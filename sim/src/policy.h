#pragma once

// Policy evaluation seam for the traversal. The interface is batched by
// design: real inference (advantage/strategy nets) will evaluate many
// infosets per call, so single-infoset callers pass n = 1 today and the
// traversal never has to change when batching lands.

#include <array>
#include <cstddef>
#include <random>

#include "abstraction.h"
#include "infoset.h"

namespace pkrbot::engine {

struct PolicyProvider {
  virtual ~PolicyProvider() = default;

  // Fill out[0..n) with a policy vector over abstract actions for each
  // infoset in obs[0..n). Illegal actions must get probability 0.
  virtual void evaluate(const InfoSet* obs, size_t n,
                        std::array<double, NUM_ABSTRACT_ACTIONS>* out) = 0;
};

// Uniform over legal actions; placeholder until network inference lands.
struct UniformPolicy : PolicyProvider {
  void evaluate(const InfoSet* obs, size_t n,
                std::array<double, NUM_ABSTRACT_ACTIONS>* out) override {
    for (size_t b = 0; b < n; ++b) {
      std::array<double, NUM_ABSTRACT_ACTIONS>& policyVector = out[b];
      policyVector = {};
      int legalCount = 0;
      for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
        if (obs[b].legalActionMask[i] == 1) {
          legalCount++;
        }
      }
      for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
        if (obs[b].legalActionMask[i] == 1) {
          policyVector[i] = 1.0 / legalCount;
        }
      }
    }
  }
};

// Sample an action index from a policy vector by inverse CDF. The vector's
// mass sits on legal actions (illegal actions are 0 per the PolicyProvider
// contract); sampling over the actual total tolerates numeric drift from 1.
inline int samplePolicy(const std::array<double, NUM_ABSTRACT_ACTIONS>& policyVector,
                        std::mt19937_64& rng) {
  double total = 0.0;
  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
    total += policyVector[i];
  }
  std::uniform_real_distribution<double> uni(0.0, total);
  double r = uni(rng);
  double acc = 0.0;
  int last = 0;
  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
    if (policyVector[i] <= 0.0) continue;
    acc += policyVector[i];
    if (r < acc) return i;
    last = i;
  }
  return last;  // r landed on the tail due to rounding
}

}  // namespace pkrbot::engine
