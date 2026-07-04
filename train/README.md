# train/ — Python training side of the Deep CFR pipeline

Counterpart to the C++ simulator/traversal side in `sim/`. The two sides share
two contracts:

- the flat float32 feature vector defined in `sim/src/encoding.h` and mirrored
  here in `features.py`/`data.py`, gated by `FEATURE_VERSION` (currently 2,
  `INPUT_SIZE = 1693`);
- the raw sample-dump format written by `sim/build/train --out` (header +
  trivially-copyable records, `sim/src/buffers.h`), read back by `data.py` as
  numpy structured dtypes whose record size is checked at load time.

## Modules

- `features.py` — layout constants and offsets mirroring `encodeInfoSet()`.
  Never hand-edit one side without the other; bump `FEATURE_VERSION` on both.
- `data.py` — sample-dump reader (`load_samples`) and the Python mirror of the
  C++ feature encoder (`encode_infosets`).
- `model.py` — `DeepCFRNet`, the split-trunk network (card-embedding branch +
  weight-shared history branch + residual MLP trunk, ~780k params at the
  default `NetConfig`). One class serves both nets: advantage (regret) outputs
  decoded by regret matching in C++, average-strategy logits decoded by
  `masked_softmax`. `export_torchscript()` is what the simulator consumes
  (loaded by libtorch in `sim/src/torch_engine.cpp`, with the layout contract
  embedded as extra files); `export_onnx()` remains for a future TensorRT path.
- `train_advantage.py` — fits the advantage net from scratch on a regret dump
  (weighted full-vector MSE, linear-CFR weights from each sample's
  `iteration`) and exports the TorchScript module.
- `test_model.py` — layout pins, shape/invariance tests, export smokes.

## The flywheel

One data-collection → training → data-collection step, from the repo root
(build `sim/` first, see `sim/CMakeLists.txt` for the libtorch configure line):

```
sim/build/train --rounds 100000 --iteration 1 --out data/iter1
python3 -m train.train_advantage --data data/iter1 --out nets/advantage.pt
sim/build/train --rounds 100000 --iteration 2 --engine nets/advantage.pt --out data/iter2
```

## Running tests

From the repo root (needs `torch`; `onnx` optional for the export test):

```
python3 -m train.test_model
```

## Still to come

- The outer training loop driving the flywheel across CFR iterations, and the
  final average-strategy fit from the strategy dumps (linear-CFR weighted,
  `masked_softmax` decode).
- A golden test that `encode_infosets()` matches the C++ `encodeInfoSet()`
  byte for byte (invariant checks against real dumps exist; a fixed C++
  reference vector does not).
