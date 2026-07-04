"""Python mirror of the network input/output contract in sim/src/encoding.h.

This module owns only layout *constants and offsets* — the single source of
truth for how the C++ encoder lays out the flat float32 input vector. Anything
here must stay bit-compatible with encodeInfoSet() in sim/src/encoding.h; bump
FEATURE_VERSION on both sides whenever the layout changes.

Input tensor: float32 [batch, INPUT_SIZE], laid out as
  [STREET_OFFSET  ..)  street one-hot over NUM_STREETS (streets {0,2,3,4,5,6}
                       packed densely: 0 -> 0, else street - 1)
  [SCALAR_OFFSET  ..)  own pip, opp pip, own stack, opp stack (each divided by
                       STARTING_STACK) and pot (divided by 2*STARTING_STACK)
  [CARD_OFFSET    ..)  NUM_CARD_SLOTS x 52 card one-hots: 3 hole then 7 board
                       (discarded/absent/hidden card = all zero)
  [LEGAL_OFFSET   ..)  legal-action mask, NUM_ABSTRACT_ACTIONS floats
  [HISTORY_OFFSET ..)  MAX_ACTIONS history slots, HISTORY_SLOT_SIZE floats
                       each: [occupied, actor-is-me, street one-hot,
                       abstract-action one-hot]

Output tensor: float32 [batch, NUM_ABSTRACT_ACTIONS] — advantages for the
regret net (decoded by regret matching on the C++ side), logits for the
average-strategy net (decoded by masked softmax).
"""

FEATURE_VERSION = 2

# Mirrors STARTING_STACK in sim/src/state.h; used to normalize the scalar
# features exactly like encodeInfoSet().
STARTING_STACK = 400

# Card codes match deck.h: rank '2'..'A' -> 0..12, suit c,d,h,s -> 0..3,
# code = rank * 4 + suit (rank-major). The rank/suit factorization in the
# model's card embedding depends on this ordering.
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

NUM_STREETS = 6
NUM_SCALAR_FEATURES = 5
NUM_ABSTRACT_ACTIONS = 10
MAX_ACTIONS = 64

NUM_HOLE_SLOTS = 3
NUM_BOARD_SLOTS = 7
NUM_CARD_SLOTS = NUM_HOLE_SLOTS + NUM_BOARD_SLOTS

HISTORY_SLOT_SIZE = 2 + NUM_STREETS + NUM_ABSTRACT_ACTIONS

STREET_OFFSET = 0
SCALAR_OFFSET = STREET_OFFSET + NUM_STREETS
CARD_OFFSET = SCALAR_OFFSET + NUM_SCALAR_FEATURES
LEGAL_OFFSET = CARD_OFFSET + NUM_CARD_SLOTS * NUM_CARDS
HISTORY_OFFSET = LEGAL_OFFSET + NUM_ABSTRACT_ACTIONS
INPUT_SIZE = HISTORY_OFFSET + MAX_ACTIONS * HISTORY_SLOT_SIZE

# Within-slot offsets for one history slot.
HISTORY_OCCUPIED = 0
HISTORY_ACTOR_IS_ME = 1
HISTORY_STREET = 2
HISTORY_ACTION = 2 + NUM_STREETS

# Card slot semantics, in encoding order (see state.h proceedStreet/proceed):
#   0..2  hole cards (a discarded card leaves an all-zero slot)
#   3..4  flop (two cards, revealed entering street 2)
#   5     first discard (street 2, made by the out-of-position player)
#   6     second discard (street 3)
#   7     street-5 reveal
#   8     street-6 reveal
#   9     never dealt in this variant (board maxes out at 6 cards)
#
# Groups for the card branch: cards within a group are sum-pooled (their order
# carries no information), and groups are kept separate where the *slot* does
# carry information — a face-up discard reads very differently from a dealt
# board card, and the two discards belong to different players. The
# structurally-empty slot 9 rides along with the street-6 group; its all-zero
# one-hot embeds to the zero vector and adds nothing to the pool.
CARD_GROUP_SLICES = (
    ("hole", slice(0, 3)),
    ("flop", slice(3, 5)),
    ("discard_first", slice(5, 6)),
    ("discard_second", slice(6, 7)),
    ("street5", slice(7, 8)),
    ("street6", slice(8, 10)),
)

ABSTRACT_ACTION_NAMES = (
    "Fold",
    "Call",
    "Check",
    "Discard0",
    "Discard1",
    "Discard2",
    "RaiseMin",
    "RaiseHalfPot",
    "RaisePot",
    "RaiseMax",
)


def street_index(street: int) -> int:
    """Densely pack street values {0, 2, 3, 4, 5, 6} into indices 0..5."""
    return 0 if street == 0 else street - 1
