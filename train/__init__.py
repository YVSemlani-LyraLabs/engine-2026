"""Python training side of the Deep CFR pipeline.

Modules:
  features        — layout constants mirroring sim/src/encoding.h (FEATURE_VERSION-gated)
  data            — sample-dump reader and Python mirror of the C++ feature encoder
  model           — split-trunk advantage/strategy network and export helpers
  train_advantage — advantage-net fit + TorchScript export (one flywheel step)
"""

from .data import REGRET_SAMPLE_DTYPE, STRATEGY_SAMPLE_DTYPE, encode_infosets, load_samples
from .features import FEATURE_VERSION, INPUT_SIZE, NUM_ABSTRACT_ACTIONS
from .model import DeepCFRNet, NetConfig, export_onnx, export_torchscript, masked_softmax

__all__ = [
    "FEATURE_VERSION",
    "INPUT_SIZE",
    "NUM_ABSTRACT_ACTIONS",
    "REGRET_SAMPLE_DTYPE",
    "STRATEGY_SAMPLE_DTYPE",
    "DeepCFRNet",
    "NetConfig",
    "encode_infosets",
    "export_onnx",
    "export_torchscript",
    "load_samples",
    "masked_softmax",
]
