"""Fit the advantage (regret) net on a simulator sample dump and export it.

One flywheel step:

    sim/build/train --rounds N --iteration T --out data/iterT
    python3 -m train.train_advantage --data data/iterT --out nets/advantage.pt
    sim/build/train --rounds N --iteration T+1 --engine nets/advantage.pt --out data/iterT+1

Per Deep CFR (Brown et al.) the net is trained from scratch on the regret
reservoir each iteration, with each sample's loss weighted by its CFR
iteration t (linear CFR). The loss is MSE over the full advantage vector;
illegal actions carry zero targets and are masked out again at decode time.
"""

import argparse
import os

import torch

from .data import REGRET_SAMPLE_DTYPE, encode_infosets, load_samples
from .model import DeepCFRNet, NetConfig, export_torchscript


def fit(model: DeepCFRNet, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
        epochs: int, batch_size: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=x.device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x[idx])
            loss = (w[idx] * ((pred - y[idx]) ** 2).mean(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{epochs}  loss {total / n:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="directory containing regret.bin")
    parser.add_argument("--out", required=True, help="output TorchScript module path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    samples = load_samples(os.path.join(args.data, "regret.bin"), REGRET_SAMPLE_DTYPE)
    print(f"loaded {len(samples)} regret samples from {args.data}")

    device = torch.device(args.device)
    x = torch.from_numpy(encode_infosets(samples["obs"])).to(device)
    y = torch.from_numpy(samples["regret"]).float().to(device)
    # Linear-CFR weights, normalized to mean 1 so lr is iteration-independent.
    w = torch.from_numpy(samples["iteration"]).float().to(device)
    w /= w.mean()

    model = DeepCFRNet(NetConfig()).to(device)
    fit(model, x, y, w, args.epochs, args.batch_size, args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_torchscript(model, args.out)
    print(f"exported TorchScript module to {args.out}")


if __name__ == "__main__":
    main()
