#pragma once

// Training samples produced by MCCFR traversals. Both sample types are
// fixed-size PODs so a buffer of them can be written out as flat binary and
// read back in Python as a numpy structured dtype.
//
// `iteration` records the CFR iteration t that produced the sample. It is the
// linear-CFR sample weight (Brown et al.): the buffers retain samples
// uniformly across iterations (reservoir sampling, buffers.h), and the
// trainer weights each sample's loss by t.

#include <array>
#include <type_traits>

#include "abstraction.h"
#include "infoset.h"

namespace pkrbot::engine {

struct RegretSample {
  InfoSet obs;
  std::array<double, NUM_ABSTRACT_ACTIONS> regret;
  int traverserSeat;
  int iteration;

  RegretSample() : obs{}, regret{}, traverserSeat{}, iteration{} {}

  RegretSample(const InfoSet& obs_, const std::array<double, NUM_ABSTRACT_ACTIONS>& regret_,
               int traverserSeat_, int iteration_)
      : obs(obs_), regret(regret_), traverserSeat(traverserSeat_), iteration(iteration_) {
    // clear dead history to prevent removed actions from leaking into the buffers
    obs.actionHistory.clearDeadHistory();
  }
};

struct StrategySample {
  InfoSet obs;
  std::array<double, NUM_ABSTRACT_ACTIONS> strategy;
  int traverserSeat;
  int iteration;

  StrategySample() : obs{}, strategy{}, traverserSeat{}, iteration{} {}

  StrategySample(const InfoSet& obs_, const std::array<double, NUM_ABSTRACT_ACTIONS>& strategy_,
                 int traverserSeat_, int iteration_)
      : obs(obs_), strategy(strategy_), traverserSeat(traverserSeat_), iteration(iteration_) {
    // clear dead history to prevent removed actions from leaking into the buffers
    obs.actionHistory.clearDeadHistory();
  }
};

static_assert(std::is_trivially_copyable_v<RegretSample>);
static_assert(std::is_trivially_copyable_v<StrategySample>);

}  // namespace pkrbot::engine
