#pragma once

// Poker hand evaluator, ported from pkrbot's original C++ implementation
// (the active `evaluate` in pkrbot's source; the shipped library is a Cython
// port of the same algorithm). Decoupled from pybind11: takes raw rank/suit
// arrays instead of py::array_t so the simulator can call it in-process.
//
// Card encoding (matches pkrbot): rank '2'..'A' -> 0..12, suit c,d,h,s -> 0..3.
// Returns a packed uint32_t where higher == stronger; the top nibble (>>20) is
// the hand category 1..9 (high card .. straight flush) and the low bits encode
// kickers, so plain integer comparison ranks hands and equality means a tie.

#include <algorithm>
#include <array>
#include <cstdint>

namespace poker {

inline constexpr std::array<uint16_t, 10> kStraightMasks = {
    0b1111100000000, 0b0111110000000, 0b0011111000000, 0b0001111100000,
    0b0000111110000, 0b0000011111000, 0b0000001111100, 0b0000000111110,
    0b0000000011111, 0b1000000001111};

// Evaluate the best 5-card hand out of `n` cards (5 <= n <= 52).
inline uint32_t evaluate(const uint8_t* ranks, const uint8_t* suits, int n) {
  std::array<uint8_t, 13> counts_r{};
  std::array<uint8_t, 4> counts_s{};
  uint16_t overall = 0;
  uint16_t persuit[4] = {0, 0, 0, 0};
  uint16_t mxfl = 0;

  for (int i = 0; i < n; ++i) {
    counts_r[ranks[i]]++;
    counts_s[suits[i]]++;
    overall |= (1u << ranks[i]);
    persuit[suits[i]] |= (1u << ranks[i]);
  }

  bool is_flush = false, straight_flush = false;
  int strfl_mx = 0, mxfl_cnt = 0;
  for (int s = 0; s < 4; ++s) {
    if (counts_s[s] >= 5) {
      is_flush = true;
      mxfl = std::max<uint16_t>(mxfl, persuit[s]);
      mxfl_cnt = counts_s[s];
      for (int r = 0; r < 10; ++r) {
        if ((persuit[s] & kStraightMasks[r]) == kStraightMasks[r]) {
          strfl_mx = std::max(strfl_mx, 10 - r);
          straight_flush = true;
          break;
        }
      }
    }
  }
  if (straight_flush) return (9u << 20) | (uint32_t(strfl_mx) << 16);

  int mx1 = -1, mx2 = -1, mxp1 = -1, mxp2 = -1, mxq = -1, mxt = -1;
  for (int r = 0; r < 13; ++r) {
    if (counts_r[r] == 4) {
      mxq = r;
    } else if (counts_r[r] == 3) {
      mxp1 = std::max(mxp1, mxt);
      mxt = r;
    } else if (counts_r[r] == 2) {
      if (mxp2 > mx1) {
        mx2 = mx1;
        mx1 = mxp2;
      } else if (mxp2 > mx2) {
        mx2 = mxp2;
      }
      mxp2 = mxp1;
      mxp1 = r;
    } else if (counts_r[r] == 1) {
      mx2 = mx1;
      mx1 = r;
    }
  }

  // quads
  if (mxq != -1)
    return (8u << 20) | (uint32_t(mxq) << 16) |
           (uint32_t(std::max({mx1, mxp1, mxt})) << 12);

  // full house
  if (mxt != -1 && mxp1 != -1)
    return (7u << 20) | (uint32_t(mxt) << 16) | (uint32_t(mxp1) << 12);

  // flush: trim lowest set bits until exactly 5 ranks remain
  if (is_flush) {
    for (int r = 0; mxfl_cnt > 5; ++r) {
      if (mxfl & (1u << r)) {
        mxfl_cnt--;
        mxfl &= ~(1u << r);
      }
    }
    return (6u << 20) | mxfl;
  }

  // straight
  for (int r = 0; r < 10; ++r)
    if ((overall & kStraightMasks[r]) == kStraightMasks[r])
      return (5u << 20) | (uint32_t(10 - r) << 16);

  // trips
  if (mxt != -1)
    return (4u << 20) | (uint32_t(mxt) << 16) | (uint32_t(mx1) << 12) |
           (uint32_t(mx2) << 8);

  // two pair
  if (mxp2 != -1)
    return (3u << 20) | (uint32_t(mxp1) << 16) | (uint32_t(mxp2) << 12) |
           (uint32_t(mx1) << 8);

  // one pair
  if (mxp1 != -1) {
    int mx3 = -1;
    for (int r = mx2 - 1; r >= 0; --r) {
      if (counts_r[r] && r != mxp1) {
        mx3 = r;
        break;
      }
    }
    return (2u << 20) | (uint32_t(mxp1) << 16) | (uint32_t(mx1) << 12) |
           (uint32_t(mx2) << 8) | (uint32_t(mx3 & 0xF) << 4);
  }

  // high card: trim lowest ranks until 5 remain
  if (n > 5) {
    int extra = n;
    for (int r = 0; extra > 5; ++r) {
      if (overall & (1u << r)) {
        extra--;
        overall &= ~(1u << r);
      }
    }
  }
  return (1u << 20) | overall;
}

}  // namespace poker
