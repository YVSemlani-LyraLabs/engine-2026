# Advantage Net Bot

Python skeleton bot that plays from an advantage-net checkpoint exported by:

```bash
python3 -m train.train_advantage --data data/iter1 --out nets/advantage.pt
```

## Configure

Point the bot at a checkpoint with an environment variable before running the
engine:

```bash
export ADVANTAGE_NET_CHECKPOINT=/absolute/path/to/advantage.pt
```

Then set one player path in `config.py`:

```python
PLAYER_2_PATH = "./advantage_net_bot"
```

Optional environment variables:

- `ADVANTAGE_NET_DEVICE=cpu` or `cuda`
- `ADVANTAGE_NET_DETERMINISTIC=1` to always choose the highest-probability
  abstract action
- `ADVANTAGE_NET_TEMPERATURE=1.0` to scale advantages before regret matching

You can also edit `commands.json` to pass the same options directly, for
example:

```json
{
    "build": [],
    "run": ["python3", "player.py", "--checkpoint", "../nets/advantage.pt"]
}
```

The checkpoint may be either the TorchScript module produced by
`train.train_advantage` or a raw `DeepCFRNet` state dict.
