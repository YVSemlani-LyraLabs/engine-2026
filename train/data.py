"""Reader and feature encoder for the C++ simulator's sample dumps.

The simulator (sim/src/train.cpp --out DIR) writes the regret/strategy
reservoirs as flat binary: three uint64 header fields (magic "PKRS",
sizeof(record), record count) followed by raw trivially-copyable records
(sim/src/samples.h). The numpy structured dtypes below mirror those structs
field for field with C alignment (align=True), and the header's record-size
field is checked at load time so struct-layout drift fails loudly.

encode_infosets() is the Python mirror of encodeInfoSet() in
sim/src/encoding.h: it turns a record array of InfoSets into the flat float32
[N, INPUT_SIZE] tensor the network consumes.
"""

import numpy as np

from .features import (
    CARD_OFFSET,
    HISTORY_ACTION,
    HISTORY_ACTOR_IS_ME,
    HISTORY_OCCUPIED,
    HISTORY_OFFSET,
    HISTORY_SLOT_SIZE,
    HISTORY_STREET,
    INPUT_SIZE,
    LEGAL_OFFSET,
    MAX_ACTIONS,
    NUM_ABSTRACT_ACTIONS,
    NUM_BOARD_SLOTS,
    NUM_CARDS,
    NUM_HOLE_SLOTS,
    NUM_STREETS,
    SCALAR_OFFSET,
    STARTING_STACK,
    STREET_OFFSET,
)

SAMPLE_FILE_MAGIC = 0x53524B50  # "PKRS" little-endian

# sim/src/infoset.h ActionRecord: player, street, AbstractAction, Action{type, amount}.
ACTION_RECORD_DTYPE = np.dtype(
    [
        ("player", "<i4"),
        ("street", "<i4"),
        ("action", "<i4"),
        ("concrete_type", "<i4"),
        ("concrete_amount", "<i4"),
    ],
    align=True,
)

ACTION_HISTORY_DTYPE = np.dtype(
    [("records", ACTION_RECORD_DTYPE, (MAX_ACTIONS,)), ("tail", "<i4")],
    align=True,
)

INFOSET_DTYPE = np.dtype(
    [
        ("player", "<i4"),
        ("street", "<i4"),
        ("pips", "<i4", (2,)),
        ("stacks", "<i4", (2,)),
        ("board", "<i4", (7,)),
        ("hole", "<i4", (3,)),
        ("history", ACTION_HISTORY_DTYPE),
        ("legal_mask", "<i4", (NUM_ABSTRACT_ACTIONS,)),
    ],
    align=True,
)


def _sample_dtype(target_field: str) -> np.dtype:
    return np.dtype(
        [
            ("obs", INFOSET_DTYPE),
            (target_field, "<f8", (NUM_ABSTRACT_ACTIONS,)),
            ("traverser_seat", "<i4"),
            ("iteration", "<i4"),
        ],
        align=True,
    )


REGRET_SAMPLE_DTYPE = _sample_dtype("regret")
STRATEGY_SAMPLE_DTYPE = _sample_dtype("strategy")


def load_samples(path: str, dtype: np.dtype) -> np.ndarray:
    """Load a sample dump written by writeSamples() (sim/src/buffers.h)."""
    with open(path, "rb") as f:
        magic, record_size, count = np.fromfile(f, dtype="<u8", count=3)
        if magic != SAMPLE_FILE_MAGIC:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        if record_size != dtype.itemsize:
            raise ValueError(
                f"{path}: record size {record_size} != dtype itemsize {dtype.itemsize}; "
                "the C++ structs and the numpy dtypes have drifted"
            )
        samples = np.fromfile(f, dtype=dtype, count=int(count))
    if len(samples) != count:
        raise ValueError(f"{path}: expected {count} records, read {len(samples)}")
    return samples


def _street_index(street: np.ndarray) -> np.ndarray:
    """Vectorized mirror of streetIndex(): {0,2,3,4,5,6} -> 0..5."""
    return np.where(street == 0, 0, street - 1)


def encode_infosets(obs: np.ndarray) -> np.ndarray:
    """Encode a record array of InfoSets to [N, INPUT_SIZE] float32.

    Bit-identical mirror of encodeInfoSet() in sim/src/encoding.h.
    """
    n = len(obs)
    x = np.zeros((n, INPUT_SIZE), dtype=np.float32)
    rows = np.arange(n)

    x[rows, STREET_OFFSET + _street_index(obs["street"])] = 1.0

    pips = obs["pips"].astype(np.float32)
    stacks = obs["stacks"].astype(np.float32)
    pot = (STARTING_STACK - stacks[:, 0]) + (STARTING_STACK - stacks[:, 1]) + pips[:, 0] + pips[:, 1]
    x[:, SCALAR_OFFSET + 0] = pips[:, 0] / STARTING_STACK
    x[:, SCALAR_OFFSET + 1] = pips[:, 1] / STARTING_STACK
    x[:, SCALAR_OFFSET + 2] = stacks[:, 0] / STARTING_STACK
    x[:, SCALAR_OFFSET + 3] = stacks[:, 1] / STARTING_STACK
    x[:, SCALAR_OFFSET + 4] = pot / (2 * STARTING_STACK)

    for c in range(NUM_HOLE_SLOTS):
        card = obs["hole"][:, c]
        valid = card >= 0
        x[rows[valid], CARD_OFFSET + c * NUM_CARDS + card[valid]] = 1.0
    for c in range(NUM_BOARD_SLOTS):
        card = obs["board"][:, c]
        valid = card >= 0
        base = CARD_OFFSET + (NUM_HOLE_SLOTS + c) * NUM_CARDS
        x[rows[valid], base + card[valid]] = 1.0

    x[:, LEGAL_OFFSET:HISTORY_OFFSET] = obs["legal_mask"]

    records = obs["history"]["records"]
    tail = obs["history"]["tail"]
    for slot in range(MAX_ACTIONS):
        occupied = slot < tail
        if not occupied.any():
            break  # slots are filled front to back; the rest are empty too
        r = rows[occupied]
        base = HISTORY_OFFSET + slot * HISTORY_SLOT_SIZE
        rec = records[occupied, slot]
        x[r, base + HISTORY_OCCUPIED] = 1.0
        x[r, base + HISTORY_ACTOR_IS_ME] = (rec["player"] == obs["player"][occupied]).astype(
            np.float32
        )
        x[r, base + HISTORY_STREET + _street_index(rec["street"])] = 1.0
        x[r, base + HISTORY_ACTION + rec["action"]] = 1.0

    return x
