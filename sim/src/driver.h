#pragma once

// Round setup shared by the match harness (engine.cpp) and the Deep CFR
// trainer (train.cpp).

#include <array>
#include <vector>

#include "deck.h"
#include "state.h"

namespace pkrbot::engine {

// Reshuffle the deck, deal both hands, and build the pre-flop RoundState.
inline RoundState makeInitialRound(Deck& deck) {
  deck.reset();
  deck.shuffle();
  std::array<std::vector<Card>, 2> hands = {deck.deal(3), deck.deal(3)};
  return RoundState{0,
                    0,
                    {SMALL_BLIND, BIG_BLIND},
                    {STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND},
                    std::move(hands),
                    {},
                    &deck};
}

}  // namespace pkrbot::engine
