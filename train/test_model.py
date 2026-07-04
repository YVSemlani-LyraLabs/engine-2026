"""Tests for the split-trunk model and the feature-layout mirror.

Run from the repo root:  python3 -m train.test_model   (or pytest train/)
"""

import io

import torch

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
    NUM_CARDS,
    NUM_SCALAR_FEATURES,
    NUM_STREETS,
    SCALAR_OFFSET,
    STREET_OFFSET,
    street_index,
)
from .data import INFOSET_DTYPE, REGRET_SAMPLE_DTYPE, STRATEGY_SAMPLE_DTYPE
from .model import DeepCFRNet, NetConfig, export_onnx, export_torchscript, masked_softmax

SMALL_CONFIG = NetConfig(
    card_emb_dim=8, card_out_dim=16, hist_slot_dim=4, hist_out_dim=16,
    trunk_dim=32, trunk_blocks=2,
)


def make_input(batch: int, seed: int = 0) -> torch.Tensor:
    """Build structurally valid encoded vectors (one-hots where the C++
    encoder writes one-hots, zeros elsewhere)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, INPUT_SIZE)
    for b in range(batch):
        x[b, STREET_OFFSET + int(torch.randint(0, NUM_STREETS, (1,), generator=gen))] = 1.0
        x[b, SCALAR_OFFSET : SCALAR_OFFSET + NUM_SCALAR_FEATURES] = torch.rand(
            NUM_SCALAR_FEATURES, generator=gen
        )
        cards = torch.randperm(NUM_CARDS, generator=gen)[:6]
        # 3 hole cards + a 2-card flop + one discard; later slots stay hidden.
        for slot, card in enumerate(cards):
            x[b, CARD_OFFSET + slot * NUM_CARDS + card] = 1.0
        legal = torch.randint(0, 2, (NUM_ABSTRACT_ACTIONS,), generator=gen).float()
        legal[1] = 1.0  # at least one legal action
        x[b, LEGAL_OFFSET:HISTORY_OFFSET] = legal
        occupied = int(torch.randint(1, 10, (1,), generator=gen))
        for slot in range(occupied):
            base = HISTORY_OFFSET + slot * HISTORY_SLOT_SIZE
            x[b, base + HISTORY_OCCUPIED] = 1.0
            x[b, base + HISTORY_ACTOR_IS_ME] = float(slot % 2)
            x[b, base + HISTORY_STREET + street_index(0)] = 1.0
            x[b, base + HISTORY_ACTION + int(torch.randint(0, NUM_ABSTRACT_ACTIONS, (1,), generator=gen))] = 1.0
    return x


def swap_card_slots(x: torch.Tensor, slot_a: int, slot_b: int) -> torch.Tensor:
    y = x.clone()
    a = slice(CARD_OFFSET + slot_a * NUM_CARDS, CARD_OFFSET + (slot_a + 1) * NUM_CARDS)
    b = slice(CARD_OFFSET + slot_b * NUM_CARDS, CARD_OFFSET + (slot_b + 1) * NUM_CARDS)
    y[:, a], y[:, b] = x[:, b], x[:, a]
    return y


def test_layout_matches_cpp_contract():
    # Pinned totals from sim/src/encoding.h (FEATURE_VERSION 2).
    assert HISTORY_SLOT_SIZE == 18
    assert INPUT_SIZE == 1693
    assert HISTORY_OFFSET + MAX_ACTIONS * HISTORY_SLOT_SIZE == INPUT_SIZE


def test_forward_shape_and_finite():
    model = DeepCFRNet(SMALL_CONFIG)
    out = model(make_input(7))
    assert out.shape == (7, NUM_ABSTRACT_ACTIONS)
    assert torch.isfinite(out).all()


def test_hole_permutation_invariance():
    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(4, seed=1)
    for other in (1, 2):
        torch.testing.assert_close(model(x), model(swap_card_slots(x, 0, other)))


def test_flop_permutation_invariance():
    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(4, seed=2)
    torch.testing.assert_close(model(x), model(swap_card_slots(x, 3, 4)))


def test_board_groups_are_not_interchangeable():
    # A discard slot and a flop slot must NOT be exchangeable: they are
    # distinct groups precisely because slot identity carries information.
    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(4, seed=3)
    assert not torch.allclose(model(x), model(swap_card_slots(x, 4, 5)))


def test_unoccupied_history_slots_are_inert():
    # Garbage in a slot whose occupied flag is 0 must be gated out entirely.
    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(4, seed=4)
    y = x.clone()
    base = HISTORY_OFFSET + (MAX_ACTIONS - 1) * HISTORY_SLOT_SIZE
    y[:, base + HISTORY_ACTOR_IS_ME] = 1.0
    y[:, base + HISTORY_STREET + 3] = 1.0
    y[:, base + HISTORY_ACTION + 7] = 1.0
    torch.testing.assert_close(model(x), model(y))


def test_occupied_history_slots_matter():
    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(4, seed=5)
    y = x.clone()
    base = HISTORY_OFFSET  # slot 0 is always occupied in make_input
    y[:, base + HISTORY_ACTOR_IS_ME] = 1.0 - y[:, base + HISTORY_ACTOR_IS_ME]
    assert not torch.allclose(model(x), model(y))


def test_masked_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p = masked_softmax(logits, mask)
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert p[0, 1] == 0.0 and p[0, 3] == 0.0
    assert p[0, 2] > p[0, 0]


def test_sample_dtype_layout():
    # Pinned sizeof()s of the C++ PODs (sim/src/infoset.h, samples.h). The
    # sample-file header re-checks record size at load time; this test catches
    # dtype drift without needing a dump on hand.
    assert INFOSET_DTYPE.itemsize == 1388
    assert REGRET_SAMPLE_DTYPE.itemsize == 1480
    assert STRATEGY_SAMPLE_DTYPE.itemsize == 1480


def test_torchscript_export():
    import os
    import tempfile

    model = DeepCFRNet(SMALL_CONFIG).eval()
    x = make_input(3, seed=6)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.pt")
        export_torchscript(model, path)
        extra = {"feature_version": "", "input_size": "", "output_size": ""}
        loaded = torch.jit.load(path, _extra_files=extra)
        assert extra["feature_version"].decode() == "2"
        assert int(extra["input_size"]) == INPUT_SIZE
        assert int(extra["output_size"]) == NUM_ABSTRACT_ACTIONS
        torch.testing.assert_close(loaded(x), model(x))


def test_onnx_export():
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("  (onnx not installed, skipping export test)")
        return
    import os
    import tempfile

    model = DeepCFRNet(SMALL_CONFIG)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.onnx")
        export_onnx(model, path)
        assert os.path.getsize(path) > 0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
