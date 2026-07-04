import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from player import (  # noqa: E402
    ABSTRACT_CALL,
    ABSTRACT_CHECK,
    ABSTRACT_DISCARD0,
    ABSTRACT_DISCARD1,
    ABSTRACT_DISCARD2,
    ABSTRACT_FOLD,
    ABSTRACT_RAISE_HALF_POT,
    ABSTRACT_RAISE_MAX,
    ABSTRACT_RAISE_MIN,
    ABSTRACT_RAISE_POT,
    Player,
    action_from_abstract,
    card_code,
    legal_abstract_actions,
    policy_from_advantages,
)
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction  # noqa: E402
from skeleton.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK, RoundState  # noqa: E402
from train.features import CARD_OFFSET, HISTORY_OFFSET, INPUT_SIZE, LEGAL_OFFSET, NUM_CARDS, SCALAR_OFFSET  # noqa: E402


def new_round_state(active=0):
    return RoundState(
        active,
        0,
        [SMALL_BLIND, BIG_BLIND],
        [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
        [["As", "Kd", "2c"], []],
        [],
        None,
    )


def test_card_code_matches_training_contract():
    assert card_code("2c") == 0
    assert card_code("2d") == 1
    assert card_code("2h") == 2
    assert card_code("2s") == 3
    assert card_code("As") == 51


def test_legal_abstract_actions_preflop_call_spot():
    mask = legal_abstract_actions(new_round_state(active=0))
    assert mask[ABSTRACT_FOLD] == 1
    assert mask[ABSTRACT_CALL] == 1
    assert mask[ABSTRACT_CHECK] == 0
    assert mask[ABSTRACT_RAISE_MIN] == 1
    assert mask[ABSTRACT_RAISE_HALF_POT] == 1
    assert mask[ABSTRACT_RAISE_POT] == 1
    assert mask[ABSTRACT_RAISE_MAX] == 1


def test_legal_abstract_actions_discard_spot():
    state = RoundState(
        1,
        2,
        [0, 0],
        [398, 398],
        [[], ["As", "Kd", "2c"]],
        ["3c", "4d"],
        None,
    )
    mask = legal_abstract_actions(state)
    assert mask == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert isinstance(action_from_abstract(state, ABSTRACT_DISCARD2), DiscardAction)


def test_raise_concretization_matches_abstraction():
    state = new_round_state(active=0)
    assert action_from_abstract(state, ABSTRACT_RAISE_MIN) == RaiseAction(4)
    assert action_from_abstract(state, ABSTRACT_RAISE_HALF_POT) == RaiseAction(4)
    assert action_from_abstract(state, ABSTRACT_RAISE_POT) == RaiseAction(5)
    assert action_from_abstract(state, ABSTRACT_RAISE_MAX) == RaiseAction(400)


def test_policy_from_advantages_uses_regret_matching_and_best_fallback():
    mask = [0, 1, 1] + [0] * 7
    policy = policy_from_advantages([10.0, 2.0, 6.0] + [0.0] * 7, mask)
    assert policy[ABSTRACT_CALL] == 0.25
    assert policy[ABSTRACT_CHECK] == 0.75

    fallback = policy_from_advantages([10.0, -2.0, -1.0] + [0.0] * 7, mask)
    assert fallback[ABSTRACT_CHECK] == 1.0


def test_encode_infoset_places_scalars_cards_legal_mask_and_history():
    player = Player.__new__(Player)
    player.action_history = [
        {"player": 0, "street": 0, "action": ABSTRACT_CALL},
        {"player": 1, "street": 0, "action": ABSTRACT_RAISE_MAX},
    ]
    state = new_round_state(active=0)
    x, legal_mask = Player.encode_infoset(player, state, 0)

    assert len(x) == INPUT_SIZE
    assert x[SCALAR_OFFSET + 0] == SMALL_BLIND / STARTING_STACK
    assert x[SCALAR_OFFSET + 1] == BIG_BLIND / STARTING_STACK
    assert x[CARD_OFFSET + card_code("As")] == 1.0
    assert x[CARD_OFFSET + NUM_CARDS + card_code("Kd")] == 1.0
    assert x[LEGAL_OFFSET:HISTORY_OFFSET] == [float(v) for v in legal_mask]


def test_action_fallback_prefers_check_call_then_fold():
    check_state = RoundState(1, 0, [2, 2], [398, 398], [[], []], [], None)
    assert isinstance(action_from_abstract(check_state, ABSTRACT_FOLD), FoldAction)
    assert isinstance(action_from_abstract(check_state, ABSTRACT_CHECK), CheckAction)

    call_state = new_round_state(active=0)
    assert isinstance(action_from_abstract(call_state, ABSTRACT_CHECK), CallAction)
