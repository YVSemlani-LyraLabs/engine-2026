#pragma once

// Bot interface for the in-process simulator. A bot is asked for an action on
// each of its turns and is told which seat (0 or 1) it occupies, so it reads
// its own hole cards via state.hands[seat]. Bots are trusted (full RoundState
// is passed); the driver still sanitizes returned actions for legality.

#include "state.h"

namespace pkrbot::engine {

struct Bot {
  virtual ~Bot() = default;

  virtual Action getAction(const RoundState& state, int seat) = 0;

  // Called once per round on both bots after the round resolves.
  virtual void onRoundEnd(const TerminalState& /*terminal*/, int /*seat*/) {}
};

}  // namespace pkrbot::engine
