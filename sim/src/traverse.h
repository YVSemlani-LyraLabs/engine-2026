#pragma once

// External-sampling MCCFR traversal (Deep CFR, Brown et al.). At traverser
// nodes every legal abstract action is explored and an advantage (regret)
// sample is recorded; at opponent nodes one action is sampled from the
// policy and a strategy sample is recorded.

#include <random>

#include "buffers.h"
#include "infoset.h"
#include "policy.h"
#include "samples.h"
#include "state.h"

namespace pkrbot::engine {

// Traverse the game tree rooted at `result` and return its value for
// `traverserSeat`. `iteration` is the CFR iteration t, recorded on every
// sample (see samples.h). `rng` drives opponent-action sampling; seed it for
// reproducible traversals. Samples go through the caller's worker-local
// ReservoirWriters (buffers.h), so recording a sample never touches the
// shared buffers' locks.
double traverse(const StateResult& result, ActionHistory& history, PolicyProvider& policy,
                ReservoirWriter<RegretSample>& regretWriter,
                ReservoirWriter<StrategySample>& strategyWriter, int traverserSeat, int iteration,
                std::mt19937_64& rng);

}  // namespace pkrbot::engine
