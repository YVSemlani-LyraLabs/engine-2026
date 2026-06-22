#pragma once

// In-process port of pkrbot's Card/Deck (the parts the simulator needs).
//
// Card encoding matches pkrbot and eval.h: rank '2'..'A' -> 0..12,
// suit c,d,h,s -> 0..3. A fresh deck is rank-major (index = rank*4 + suit),
// matching pkrbot's order (2c,2d,2h,2s,3c,...).
//
// deal(n) removes the top n cards; peek(n) returns the top n of the remaining
// deck without consuming. Both are modeled with a cursor over a fixed 52-card
// vector, so peek after dealing reads from the post-deal top (verified against
// pkrbot: after two deal(3)s, peek(2) == deck indices 6..7).
//
// Shuffling uses a seedable Mersenne Twister for reproducible self-play; it
// does not reproduce CPython's random.Random bit-for-bit (unnecessary for the
// in-process simulator).

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pkrbot::engine {

inline constexpr char kRanks[] = "23456789TJQKA";
inline constexpr char kSuits[] = "cdhs";

struct Card {
  uint8_t rank = 0;  // 0..12
  uint8_t suit = 0;  // 0..3

  Card() = default;
  constexpr Card(uint8_t rank, uint8_t suit) : rank(rank), suit(suit) {}

  // 0..51, rank-major (matches pkrbot's fresh-deck order).
  constexpr uint8_t code() const { return rank * 4 + suit; }
  static constexpr Card fromCode(uint8_t code) {
    return Card(static_cast<uint8_t>(code / 4), static_cast<uint8_t>(code % 4));
  }

  std::string str() const { return std::string{kRanks[rank], kSuits[suit]}; }

  bool operator==(const Card& o) const { return rank == o.rank && suit == o.suit; }
  bool operator!=(const Card& o) const { return !(*this == o); }
};

class Deck {
 public:
  Deck() : Deck(std::random_device{}()) {}
  explicit Deck(uint64_t seed) : rng_(seed) { reset(); }

  // Restore a fresh, ordered 52-card deck.
  void reset() {
    cards_.resize(52);
    for (int i = 0; i < 52; ++i) cards_[i] = Card::fromCode(static_cast<uint8_t>(i));
    cursor_ = 0;
  }

  // Randomize order. Resets the deal cursor (call before dealing).
  void shuffle() {
    for (int i = 51; i > 0; --i) {
      std::uniform_int_distribution<int> dist(0, i);
      std::swap(cards_[i], cards_[dist(rng_)]);
    }
    cursor_ = 0;
  }

  // Remove and return the top n cards.
  std::vector<Card> deal(int n) {
    requireAvailable(n);
    std::vector<Card> out(cards_.begin() + cursor_, cards_.begin() + cursor_ + n);
    cursor_ += n;
    return out;
  }

  // Return the top n cards of the remaining deck without consuming them.
  std::vector<Card> peek(int n) const {
    requireAvailable(n);
    return std::vector<Card>(cards_.begin() + cursor_, cards_.begin() + cursor_ + n);
  }

  int remaining() const { return static_cast<int>(cards_.size()) - cursor_; }

 private:
  void requireAvailable(int n) const {
    if (n < 0 || cursor_ + n > static_cast<int>(cards_.size()))
      throw std::out_of_range("Deck: not enough cards");
  }

  std::vector<Card> cards_;
  int cursor_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace pkrbot::engine
