#pragma once

// Training samples produced by MCCFR traversals. Both sample types are
// fixed-size PODs so a buffer of them can be written out as flat binary and
// read back in Python as a numpy structured dtype.
//
// `iteration` records the CFR iteration t that produced the sample. It is not
// consumed anywhere yet; it is reserved for linear-CFR sample weighting
// (Brown et al.), which lands in a later pass. Recording it now keeps the
// on-disk format stable.

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
