'''
Advantage-net pokerbot, written in Python from the Python skeleton template.
'''
import argparse
import os
import random
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

try:
    from .skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
    from .skeleton.bot import Bot
    from .skeleton.runner import run_bot
    from .skeleton.states import STARTING_STACK
except ImportError:
    from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
    from skeleton.bot import Bot
    from skeleton.runner import run_bot
    from skeleton.states import STARTING_STACK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.features import (  # noqa: E402
    FEATURE_VERSION,
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
    SCALAR_OFFSET,
    STREET_OFFSET,
    street_index,
)

ABSTRACT_FOLD = 0
ABSTRACT_CALL = 1
ABSTRACT_CHECK = 2
ABSTRACT_DISCARD0 = 3
ABSTRACT_DISCARD1 = 4
ABSTRACT_DISCARD2 = 5
ABSTRACT_RAISE_MIN = 6
ABSTRACT_RAISE_HALF_POT = 7
ABSTRACT_RAISE_POT = 8
ABSTRACT_RAISE_MAX = 9

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_code(card):
    return RANKS.index(card[0]) * len(SUITS) + SUITS.index(card[1])


def legal_abstract_actions(round_state):
    legal_actions = round_state.legal_actions()
    mask = [0] * NUM_ABSTRACT_ACTIONS
    if DiscardAction in legal_actions:
        mask[ABSTRACT_DISCARD0] = 1
        mask[ABSTRACT_DISCARD1] = 1
        mask[ABSTRACT_DISCARD2] = 1
        return mask

    mask[ABSTRACT_FOLD] = int(FoldAction in legal_actions)
    mask[ABSTRACT_CALL] = int(CallAction in legal_actions)
    mask[ABSTRACT_CHECK] = int(CheckAction in legal_actions)

    if RaiseAction in legal_actions:
        active = round_state.button % 2
        continue_cost = round_state.pips[1 - active] - round_state.pips[active]
        pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
        _, max_raise = round_state.raise_bounds()
        mask[ABSTRACT_RAISE_MIN] = 1
        mask[ABSTRACT_RAISE_MAX] = 1
        if round_state.pips[active] + continue_cost + (pot // 2) <= max_raise:
            mask[ABSTRACT_RAISE_HALF_POT] = 1
        if round_state.pips[active] + continue_cost + pot <= max_raise:
            mask[ABSTRACT_RAISE_POT] = 1
    return mask


def raise_targets(round_state):
    active = round_state.button % 2
    continue_cost = round_state.pips[1 - active] - round_state.pips[active]
    pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
    min_raise, max_raise = round_state.raise_bounds()
    targets = {
        ABSTRACT_RAISE_MIN: min_raise,
        ABSTRACT_RAISE_HALF_POT: round_state.pips[active] + continue_cost + (pot // 2),
        ABSTRACT_RAISE_POT: round_state.pips[active] + continue_cost + pot,
        ABSTRACT_RAISE_MAX: max_raise,
    }
    return {action: max(min_raise, min(target, max_raise)) for action, target in targets.items()}


def abstract_from_action(round_state, action):
    if isinstance(action, FoldAction):
        return ABSTRACT_FOLD
    if isinstance(action, CallAction):
        return ABSTRACT_CALL
    if isinstance(action, CheckAction):
        return ABSTRACT_CHECK
    if isinstance(action, DiscardAction):
        return ABSTRACT_DISCARD0 + action.card
    if isinstance(action, RaiseAction):
        targets = raise_targets(round_state)
        return min(targets, key=lambda abstract: abs(targets[abstract] - action.amount))
    raise ValueError(f"unknown action {action!r}")


def action_from_abstract(round_state, abstract_action):
    legal_actions = round_state.legal_actions()
    if abstract_action == ABSTRACT_FOLD and FoldAction in legal_actions:
        return FoldAction()
    if abstract_action == ABSTRACT_CALL and CallAction in legal_actions:
        return CallAction()
    if abstract_action == ABSTRACT_CHECK and CheckAction in legal_actions:
        return CheckAction()
    if ABSTRACT_DISCARD0 <= abstract_action <= ABSTRACT_DISCARD2 and DiscardAction in legal_actions:
        card = abstract_action - ABSTRACT_DISCARD0
        return DiscardAction(min(card, len(round_state.hands[round_state.button % 2]) - 1))
    if abstract_action >= ABSTRACT_RAISE_MIN and RaiseAction in legal_actions:
        return RaiseAction(raise_targets(round_state)[abstract_action])

    if CheckAction in legal_actions:
        return CheckAction()
    if CallAction in legal_actions:
        return CallAction()
    return FoldAction()


def policy_from_advantages(advantages, legal_mask):
    positives = [max(float(advantages[i]), 0.0) if legal_mask[i] else 0.0 for i in range(NUM_ABSTRACT_ACTIONS)]
    total = sum(positives)
    if total > 0:
        return [value / total for value in positives]

    legal = [i for i, flag in enumerate(legal_mask) if flag]
    if not legal:
        raise ValueError("no legal abstract actions")
    best = max(legal, key=lambda i: float(advantages[i]))
    return [1.0 if i == best else 0.0 for i in range(NUM_ABSTRACT_ACTIONS)]


def choose_action(policy, deterministic):
    if deterministic:
        return max(range(NUM_ABSTRACT_ACTIONS), key=lambda i: policy[i])

    r = random.random()
    running = 0.0
    for i, probability in enumerate(policy):
        running += probability
        if r <= running:
            return i
    return max(range(NUM_ABSTRACT_ACTIONS), key=lambda i: policy[i])


class Player(Bot):
    '''
    A pokerbot powered by an advantage-net checkpoint.
    '''

    def __init__(self, checkpoint_path, device="cpu", deterministic=False, temperature=1.0):
        if checkpoint_path is None:
            raise SystemExit(
                "Set ADVANTAGE_NET_CHECKPOINT or pass --checkpoint with a TorchScript "
                "module exported by train.train_advantage."
            )
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = (Path.cwd() / self.checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise SystemExit(f"advantage checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
        except ImportError as exc:
            raise SystemExit("advantage_net_bot requires torch; install project dependencies with uv sync") from exc

        self.torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.temperature = max(float(temperature), 1e-6)
        self.model = self._load_model()
        self.model.eval()
        self.action_history = []

    def _load_model(self):
        torch = self.torch
        try:
            extra_files = {"feature_version": "", "input_size": "", "output_size": ""}
            model = torch.jit.load(str(self.checkpoint_path), map_location=self.device, _extra_files=extra_files)
            self._check_torchscript_metadata(extra_files)
            return model.to(self.device)
        except RuntimeError:
            from train.model import DeepCFRNet, NetConfig

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = DeepCFRNet(NetConfig())
            model.load_state_dict(state_dict)
            return model.to(self.device)

    def _check_torchscript_metadata(self, extra_files):
        feature_version = extra_files.get("feature_version", "")
        input_size = extra_files.get("input_size", "")
        output_size = extra_files.get("output_size", "")
        if feature_version and int(feature_version) != FEATURE_VERSION:
            raise SystemExit(
                f"checkpoint feature version {int(feature_version)} does not match bot version {FEATURE_VERSION}"
            )
        if input_size and int(input_size) != INPUT_SIZE:
            raise SystemExit(f"checkpoint input size {int(input_size)} does not match bot input size {INPUT_SIZE}")
        if output_size and int(output_size) != NUM_ABSTRACT_ACTIONS:
            raise SystemExit(
                f"checkpoint output size {int(output_size)} does not match bot output size {NUM_ABSTRACT_ACTIONS}"
            )

    def reset_action_history(self):
        self.action_history = []

    def observe_action(self, round_state, action):
        if len(self.action_history) >= MAX_ACTIONS:
            return
        self.action_history.append(
            {
                "player": round_state.button % 2,
                "street": round_state.street,
                "action": abstract_from_action(round_state, action),
            }
        )

    def handle_new_round(self, game_state, round_state, active):
        self.reset_action_history()

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def encode_infoset(self, round_state, active):
        x = [0.0] * INPUT_SIZE
        x[STREET_OFFSET + street_index(round_state.street)] = 1.0

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1]) + sum(round_state.pips)

        x[SCALAR_OFFSET + 0] = my_pip / STARTING_STACK
        x[SCALAR_OFFSET + 1] = opp_pip / STARTING_STACK
        x[SCALAR_OFFSET + 2] = my_stack / STARTING_STACK
        x[SCALAR_OFFSET + 3] = opp_stack / STARTING_STACK
        x[SCALAR_OFFSET + 4] = pot / (2 * STARTING_STACK)

        cards_offset = SCALAR_OFFSET + 5
        for i, card in enumerate(round_state.hands[active][:NUM_HOLE_SLOTS]):
            x[cards_offset + i * NUM_CARDS + card_code(card)] = 1.0
        board_offset = cards_offset + NUM_HOLE_SLOTS * NUM_CARDS
        for i, card in enumerate(round_state.board[:NUM_BOARD_SLOTS]):
            x[board_offset + i * NUM_CARDS + card_code(card)] = 1.0

        legal_mask = legal_abstract_actions(round_state)
        x[LEGAL_OFFSET:HISTORY_OFFSET] = [float(value) for value in legal_mask]

        for slot, record in enumerate(self.action_history[:MAX_ACTIONS]):
            base = HISTORY_OFFSET + slot * HISTORY_SLOT_SIZE
            x[base + HISTORY_OCCUPIED] = 1.0
            x[base + HISTORY_ACTOR_IS_ME] = 1.0 if record["player"] == active else 0.0
            x[base + HISTORY_STREET + street_index(record["street"])] = 1.0
            x[base + HISTORY_ACTION + record["action"]] = 1.0
        return x, legal_mask

    def get_action(self, game_state, round_state, active):
        x, legal_mask = self.encode_infoset(round_state, active)
        torch = self.torch
        with torch.no_grad():
            tensor = torch.tensor([x], dtype=torch.float32, device=self.device)
            advantages = self.model(tensor)[0].detach().cpu().tolist()

        if self.temperature != 1.0:
            advantages = [value / self.temperature for value in advantages]
        policy = policy_from_advantages(advantages, legal_mask)
        abstract_action = choose_action(policy, self.deterministic)
        return action_from_abstract(round_state, abstract_action)


def parse_args():
    parser = argparse.ArgumentParser(prog="python3 player.py")
    parser.add_argument("--host", type=str, default="localhost", help="Host to connect to, defaults to localhost")
    parser.add_argument("--checkpoint", default=os.environ.get("ADVANTAGE_NET_CHECKPOINT"))
    parser.add_argument("--device", default=os.environ.get("ADVANTAGE_NET_DEVICE", "cpu"))
    parser.add_argument("--deterministic", action="store_true", default=os.environ.get("ADVANTAGE_NET_DETERMINISTIC") == "1")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("ADVANTAGE_NET_TEMPERATURE", "1.0")))
    parser.add_argument("port", type=int, help="Port on host to connect to")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_bot(
        Player(
            checkpoint_path=args.checkpoint,
            device=args.device,
            deterministic=args.deterministic,
            temperature=args.temperature,
        ),
        args,
    )
