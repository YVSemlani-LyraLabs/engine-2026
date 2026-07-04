"""Split-trunk Deep CFR network (Brown et al. 2019, adapted to this variant).

The network consumes the flat float32 [batch, INPUT_SIZE] vector produced by
sim/src/encoding.h — slicing into branches happens *inside* forward(), so the
ONNX/TensorRT contract stays a single flat input with a dynamic batch axis.

Architecture:
  * Card branch — every card one-hot goes through a shared embedding that sums
    card, rank, and suit components (code = rank*4 + suit, per deck.h), so
    knowledge generalizes across suits and ranks instead of being memorized
    per column. Embeddings are sum-pooled within each semantic group (hole,
    flop, each discard, each late-street reveal; see CARD_GROUP_SLICES), which
    bakes in permutation invariance for the unordered groups. An absent card
    (all-zero one-hot) embeds to the zero vector and drops out of the pool.
  * History branch — one linear projection shared across all MAX_ACTIONS
    slots (weight sharing across positions), gated by each slot's occupied
    flag so empty slots contribute exactly zero, then flattened so global
    action order is preserved with position-specific weights downstream.
  * Trunk — branch outputs concatenated with the raw context block (street
    one-hot, scalars, legal mask), then LayerNorm residual MLP blocks.

The head is a plain linear layer with no output activation: the same class
serves both nets. For the advantage (regret) net the outputs are signed
advantages, decoded by regret matching (policyFromAdvantages in encoding.h);
for the average-strategy net they are logits, decoded by masked_softmax.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (
    CARD_GROUP_SLICES,
    CARD_OFFSET,
    FEATURE_VERSION,
    HISTORY_OFFSET,
    HISTORY_OCCUPIED,
    HISTORY_SLOT_SIZE,
    INPUT_SIZE,
    LEGAL_OFFSET,
    MAX_ACTIONS,
    NUM_ABSTRACT_ACTIONS,
    NUM_CARDS,
    NUM_CARD_SLOTS,
    NUM_RANKS,
    NUM_SCALAR_FEATURES,
    NUM_STREETS,
    NUM_SUITS,
    STREET_OFFSET,
)

NUM_CONTEXT_FEATURES = NUM_STREETS + NUM_SCALAR_FEATURES + NUM_ABSTRACT_ACTIONS


@dataclass
class NetConfig:
    card_emb_dim: int = 64
    card_out_dim: int = 192
    hist_slot_dim: int = 16
    hist_out_dim: int = 192
    trunk_dim: int = 256
    trunk_blocks: int = 3


class CardEmbedding(nn.Module):
    """Embed a card one-hot as the sum of card + rank + suit embeddings.

    Implemented as bias-free linears over the one-hot (and its rank/suit
    marginals), which is an embedding lookup that also maps the all-zero
    "absent card" input to the zero vector with no special casing.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.card = nn.Linear(NUM_CARDS, dim, bias=False)
        self.rank = nn.Linear(NUM_RANKS, dim, bias=False)
        self.suit = nn.Linear(NUM_SUITS, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., NUM_CARDS] one-hot. code = rank*4 + suit, so reshaping to
        # [..., 13, 4] and marginalizing recovers rank/suit one-hots.
        by_rank_suit = x.unflatten(-1, (NUM_RANKS, NUM_SUITS))
        rank_onehot = by_rank_suit.sum(-1)
        suit_onehot = by_rank_suit.sum(-2)
        return self.card(x) + self.rank(rank_onehot) + self.suit(suit_onehot)


class CardBranch(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.embed = CardEmbedding(emb_dim)
        self.fc = nn.Linear(len(CARD_GROUP_SLICES) * emb_dim, out_dim)

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        # cards: [B, NUM_CARD_SLOTS, NUM_CARDS]
        emb = self.embed(cards)
        groups = [emb[:, sl].sum(dim=1) for _, sl in CARD_GROUP_SLICES]
        return F.relu(self.fc(torch.cat(groups, dim=-1)))


class HistoryBranch(nn.Module):
    def __init__(self, slot_dim: int, out_dim: int):
        super().__init__()
        self.slot = nn.Linear(HISTORY_SLOT_SIZE, slot_dim)
        self.fc = nn.Linear(MAX_ACTIONS * slot_dim, out_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, MAX_ACTIONS, HISTORY_SLOT_SIZE]. Gate by the occupied
        # flag so unoccupied slots contribute exactly zero (the shared linear
        # has a bias that would otherwise leak a constant into empty slots).
        occupied = history[..., HISTORY_OCCUPIED : HISTORY_OCCUPIED + 1]
        slots = F.relu(self.slot(history)) * occupied
        return F.relu(self.fc(slots.flatten(1)))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class DeepCFRNet(nn.Module):
    FEATURE_VERSION = FEATURE_VERSION

    def __init__(self, config: NetConfig = NetConfig()):
        super().__init__()
        self.config = config
        self.cards = CardBranch(config.card_emb_dim, config.card_out_dim)
        self.history = HistoryBranch(config.hist_slot_dim, config.hist_out_dim)
        merge_dim = config.card_out_dim + config.hist_out_dim + NUM_CONTEXT_FEATURES
        self.merge = nn.Linear(merge_dim, config.trunk_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.trunk_dim) for _ in range(config.trunk_blocks)
        )
        self.out_norm = nn.LayerNorm(config.trunk_dim)
        self.head = nn.Linear(config.trunk_dim, NUM_ABSTRACT_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, INPUT_SIZE] flat encoding (see features.py for the layout).
        # Street one-hot and scalars are adjacent and both feed the trunk raw,
        # so they are taken as one slice.
        context_head = x[:, STREET_OFFSET:CARD_OFFSET]
        legal_mask = x[:, LEGAL_OFFSET:HISTORY_OFFSET]
        cards = x[:, CARD_OFFSET:LEGAL_OFFSET].unflatten(1, (NUM_CARD_SLOTS, NUM_CARDS))
        history = x[:, HISTORY_OFFSET:].unflatten(1, (MAX_ACTIONS, HISTORY_SLOT_SIZE))

        merged = torch.cat(
            [self.cards(cards), self.history(history), context_head, legal_mask],
            dim=-1,
        )
        h = F.relu(self.merge(merged))
        for block in self.blocks:
            h = block(h)
        return self.head(self.out_norm(h))


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Decode average-strategy logits: softmax over legal actions only."""
    masked = logits.masked_fill(legal_mask <= 0, float("-inf"))
    return torch.softmax(masked, dim=-1)


def export_onnx(model: DeepCFRNet, path: str, opset: int = 17) -> None:
    """Export with a dynamic batch axis — the only dynamic dim TensorRT needs."""
    model.eval()
    dummy = torch.zeros(1, INPUT_SIZE)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )


def export_torchscript(model: DeepCFRNet, path: str) -> None:
    """Export for the C++ simulator's libtorch engine (sim/src/torch_engine.cpp).

    Traced rather than scripted: forward() is trace-safe (fixed slices,
    batch-agnostic ops), so the trace stays valid for any batch size. The
    layout contract is embedded as extra files and checked against encoding.h
    when the C++ side loads the module.
    """
    model.eval()
    traced = torch.jit.trace(model.cpu(), torch.zeros(2, INPUT_SIZE))
    extra_files = {
        "feature_version": str(FEATURE_VERSION),
        "input_size": str(INPUT_SIZE),
        "output_size": str(NUM_ABSTRACT_ACTIONS),
    }
    traced.save(path, _extra_files=extra_files)
