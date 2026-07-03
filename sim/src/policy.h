#pragma once

// Policy evaluation seam for the traversal. The interface is batched by
// design: real inference (advantage/strategy nets) will evaluate many
// infosets per call, so single-infoset callers pass n = 1 today and the
// traversal never has to change when batching lands.

#include <array>
#include <cstddef>

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

// Pick an action index from a policy vector.
// BUG(pass 2): argmax biases MCCFR sampling; must sample from the
// distribution (with a seeded RNG threaded through the traversal).
inline int samplePolicy(const std::array<double, NUM_ABSTRACT_ACTIONS>& policyVector) {
  int maxIndex = 0;
  for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
    if (policyVector[i] > policyVector[maxIndex]) {
      maxIndex = i;
    }
  }
  return maxIndex;
}

}  // namespace pkrbot::engine
