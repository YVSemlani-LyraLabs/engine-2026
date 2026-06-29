// Data structures and functions for running MCCFR traversals.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "bot.h"
#include "deck.h"
#include "state.h"

namespace pkrbot::engine {

constexpr int BUFFER_SIZE = 100000;
constexpr int MAX_ACTIONS = 64;

// classes for recording state

enum class AbstractAction {
  Fold,
  Call,
  Check,
  Discard0,
  Discard1,
  Discard2,
  RaiseMin,
  RaiseHalfPot,
  RaisePot,
  RaiseMax,
};

struct ActionRecord {
    int player;
    int street;
    AbstractAction action;
    Action concreteAction;
    std::array<int, 2> pipsBefore;
    std::array<int, 2> stacksBefore;
};

struct ActionHistory {
    std::array<ActionRecord, MAX_ACTIONS> actionRecords;
    int tail = 0;

    void push(ActionRecord& record) {
        if (tail >= MAX_ACTIONS) {
            throw std::runtime_error("Action history is full");
            // should never happen. if triggered we should increase the max actions
        }
        actionRecords[tail] = record;
        tail++;
    }

    void pop() {
        if (tail == 0) {
            throw std::runtime_error("Action history is empty");
            // should never happen. if triggered we should increase the max actions
        }
        tail--;
    }

    void clearDeadHistory() {
        for (int i = tail; i < MAX_ACTIONS; i++) {
            actionRecords[i] = ActionRecord();
        }
    }
};

struct InfoSet {
    int player;
    int street;
    std::array<int, 2> pips;
    std::array<int, 2> stacks;
    std::array<int, 7> board;
    std::array<int, 3> hole;

    ActionHistory actionHistory;
    std::array<int, 10> legalActionMask;
};

// sample classes

struct RegretSample {
    InfoSet obs;
    std::array<double, 10> regret;
    std::array<int, 10> actionMask;
    int traverserSeat;

    RegretSample() : obs{}, regret{}, actionMask{}, traverserSeat{} {}

    RegretSample(InfoSet& obs_, std::array<double, 10> regret_, std::array<int, 10> actionMask_, int traverserSeat_) : obs(obs_), regret(regret_), actionMask(actionMask_), traverserSeat(traverserSeat_) {
        obs.actionHistory.clearDeadHistory(); // clear dead history to prevent removed actions from leaking into the buffers
    }
};

struct StrategySample {
    InfoSet obs;
    std::array<double, 10> strategy;
    int traverserSeat;

    StrategySample() : obs{}, strategy{}, traverserSeat{} {}

    StrategySample(InfoSet& obs_, std::array<double, 10> strategy_, int traverserSeat_) : obs(obs_), strategy(strategy_), traverserSeat(traverserSeat_) {
        obs.actionHistory.clearDeadHistory(); // clear dead history to prevent removed actions from leaking into the bufferss
    }
};


// buffers for storing samples to be trained on

struct RegretBuffer {
    std::array<RegretSample, BUFFER_SIZE> samples;
    int idx = 0;

    void push(RegretSample& sample) {
        samples[idx] = sample;
        idx = (idx + 1) % BUFFER_SIZE;
    }
};

struct StrategyBuffer {
    std::array<StrategySample, BUFFER_SIZE> samples;
    int idx = 0;

    void push(StrategySample& sample) {
        samples[idx] = sample;
        idx = (idx + 1) % BUFFER_SIZE;
    }
};

// get the legal abstract actions for a given state
// ------------------------------------------------------------
// TO-DO: add filters to prevent raise sizes from overlapping with each other 
// ------------------------------------------------------------
// and splitting probability mass i.e. raise 1/2 pot == raise Max == raise Pot (b/c they get clamped to the max raise)
std::array<int, 10> legalAbstractActions(const RoundState& state) {
    int active = state.active();
    int continueCost = state.pips[1 - active] - state.pips[active];
    int pot = (STARTING_STACK - state.stacks[0]) + (STARTING_STACK - state.stacks[1]);
    std::array<int, 10> legalAbstractActions = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // [0] = fold, [1] = call, [2] = check, [3] = discard0, [4] = discard1, [5] = discard2, 
    // [6] = raise min, [7] = raise half pot, [8] = raise pot, [9] = max raise

    if (state.street == 2 || state.street == 3) {
        if (active != state.street % 2) {
            legalAbstractActions[3] = 1; // discard0 allowed
            legalAbstractActions[4] = 1; // discard1 allowed
            legalAbstractActions[5] = 1; // discard2 allowed
        } else {
            legalAbstractActions[2] = 1; // check allowed
        }

        return legalAbstractActions;
    }

    if (continueCost == 0) {
        bool betsForbidden = (state.stacks[0] == 0 || state.stacks[1] == 0);
        legalAbstractActions[0] = 1; // fold allowed
        legalAbstractActions[2] = 1; // check allowed


        auto [minRaise, maxRaise] = state.raiseBounds();
        legalAbstractActions[6] = !betsForbidden; // raise min allowed
        legalAbstractActions[9] = !betsForbidden; // raise max allowed

        if (state.pips[active] + continueCost + (pot / 2) <= maxRaise) {
            legalAbstractActions[7] = 1; // raise half pot allowed
        }

        if (state.pips[active] + continueCost + pot <= maxRaise) {
            legalAbstractActions[8] = 1; // raise pot allowed
        }
        return legalAbstractActions;
    }

    bool raisesForbidden = (continueCost == state.stacks[active] || state.stacks[1 - active] == 0);
    legalAbstractActions[0] = 1; // fold allowed
    legalAbstractActions[1] = 1; // call allowed

    auto [minRaise, maxRaise] = state.raiseBounds();
    legalAbstractActions[6] = !raisesForbidden; // raise min allowed
    legalAbstractActions[9] = !raisesForbidden; // raise max allowed

    if (state.pips[active] + continueCost + (pot / 2) <= maxRaise) {
        legalAbstractActions[7] = 1; // raise half pot allowed
    }

    if (state.pips[active] + continueCost + pot <= maxRaise) {
        legalAbstractActions[8] = 1; // raise pot allowed
    }

    return legalAbstractActions;
}

// turn an abstract raise action into a concrete action
Action concretizeRaise(const RoundState& state, AbstractAction abstractAction) { 
    auto [minRaise, maxRaise] = state.raiseBounds();
    int active = state.active();

    int continueCost = state.pips[1 - active] - state.pips[active];
    int pot = (STARTING_STACK - state.stacks[0]) + (STARTING_STACK - state.stacks[1]);
    int target = minRaise;

    switch (abstractAction) {
        case AbstractAction::RaiseMin:
            target = minRaise;
            break;
        case AbstractAction::RaiseHalfPot:
            target = state.pips[active] + continueCost + (pot / 2);
            break;
        case AbstractAction::RaisePot:
            target = state.pips[active] + continueCost + pot;
            break;
        case AbstractAction::RaiseMax:
            target = maxRaise;
            break;
        default:
            throw std::invalid_argument("Invalid abstract action");
    }

    target = std::max(minRaise, std::min(target, maxRaise));
    return Action::raise(target);

}

// turn an abstract action into a concrete action
Action concretizeAction(const RoundState& state, AbstractAction abstractAction) {
    switch (abstractAction) {
        case AbstractAction::Fold:
            return Action::fold();
        case AbstractAction::Call:
            return Action::call();
        case AbstractAction::Check:
            return Action::check();
        case AbstractAction::Discard0:
            return Action::discard(0);
        case AbstractAction::Discard1:
            return Action::discard(1);
        case AbstractAction::Discard2:
            return Action::discard(2);
        case AbstractAction::RaiseMin:
            return concretizeRaise(state, AbstractAction::RaiseMin);
        case AbstractAction::RaiseHalfPot:
            return concretizeRaise(state, AbstractAction::RaiseHalfPot);
        case AbstractAction::RaisePot:
            return concretizeRaise(state, AbstractAction::RaisePot);
        case AbstractAction::RaiseMax:
            return concretizeRaise(state, AbstractAction::RaiseMax);
        default:
            throw std::invalid_argument("Invalid abstract action");
    }
}

// get the policy vector for a given info set
std::array<double, 10> policy(const InfoSet& obs) {
    std::array<double, 10> policyVector = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int legalCount = 0;
    for (int i = 0; i < 10; i++) {
        if (obs.legalActionMask[i] == 1) {
            legalCount++;
        }
    }
    for (int i = 0; i < 10; i++) {
        if (obs.legalActionMask[i] == 1) {
            policyVector[i] = 1.0 / legalCount;
        }
    }
    return policyVector;
}

// sample an action from the policy
int samplePolicy(const std::array<double, 10>& policyVector) {

    // for now just argmax the policy
    int maxIndex = 0;
    for (int i = 0; i < 10; i++) {
        if (policyVector[i] > policyVector[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;

}

// make an info set from a given state and action history
InfoSet makeInfoSet(const RoundState& state, ActionHistory& history) {
    InfoSet obs;

    obs.player = state.active();
    obs.street = state.street;
    obs.pips[0] = state.pips[obs.player];
    obs.pips[1] = state.pips[1 - obs.player];
    obs.stacks[0] = state.stacks[obs.player];
    obs.stacks[1] = state.stacks[1 - obs.player];

    for (int i = 0; i < 7; i++) {
        if (i < state.board.size()) {
            obs.board[i] = state.board[i].code();
        } else {
            obs.board[i] = -1; // hidden board card
        }
    }

    for (int i = 0; i < 3; i++) {
        if (i < state.hands[obs.player].size()) {
            obs.hole[i] = state.hands[obs.player][i].code();
        } else {
            obs.hole[i] = -1; // discarded hole card
        }
    }
    obs.actionHistory = history;
    obs.legalActionMask = legalAbstractActions(state);
    return obs;
}

// traverse a given game
double traverse(const StateResult& result, ActionHistory& history, RegretBuffer& regretBuffer, StrategyBuffer& strategyBuffer, int traverserSeat) {
    if (std::holds_alternative<TerminalState>(result)) {
        // terminal node
        const TerminalState& terminalState = std::get<TerminalState>(result);
        return terminalState.deltas[traverserSeat];
    }

    const RoundState& state = std::get<RoundState>(result);
    
    int active = state.active();
    std::array<double, 10> regret = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
    // [0] = fold, [1] = call, [2] = check, [3] = discard0, [4] = discard1, [5] = discard2, 
    // [6] = raise min, [7] = raise half pot, [8] = raise pot, [9] = all in
    InfoSet obs = makeInfoSet(state, history);
    if (active == traverserSeat) {
        // traverser's turn
        // evaluate every legal action
        for (int i = 0; i < 10; i++) {
            if (obs.legalActionMask[i] == 1) {
                Action action = concretizeAction(state, static_cast<AbstractAction>(i));
                StateResult next = state.proceed(action);
                ActionRecord record = {
                    .player = active,
                    .street = state.street,
                    .action = static_cast<AbstractAction>(i),
                    .concreteAction = action,
                    .pipsBefore = {state.pips[0], state.pips[1]},
                    .stacksBefore = {state.stacks[0], state.stacks[1]},
                };
                history.push(record);
                regret[i] = traverse(next, history, regretBuffer, strategyBuffer, traverserSeat);
                history.pop();
            } else {
                regret[i] = -1; // illegal action
            } 
        }

        std::array<double, 10> policyVector = policy(obs);
        double nodeValue = 0.0;

        for (int i = 0; i < 10; i++) {
            if (obs.legalActionMask[i] == 1) {
                nodeValue += policyVector[i] * regret[i];
            }
        }

        // this is a Q table type of regret matching but we dont really need this for MCCFR + Deep CFR
        for (int i = 0; i < 10; i++) {
            if (obs.legalActionMask[i] == 1) {
                regret[i] -= nodeValue;
            }
        }

        RegretSample regretSample(obs, regret, obs.legalActionMask, traverserSeat);
        regretBuffer.push(regretSample);
        return nodeValue;

    } else {
        // opponent's turn
        std::array<double, 10> policyVector = policy(obs);
        int sampledActionIndex = samplePolicy(policyVector);
        AbstractAction sampledAction = static_cast<AbstractAction>(sampledActionIndex);
        Action action = concretizeAction(state, sampledAction);
        StateResult next = state.proceed(action);
        ActionRecord record = {
            .player = active,
            .street = state.street,
            .action = sampledAction,
            .concreteAction = action,
            .pipsBefore = {state.pips[0], state.pips[1]},
            .stacksBefore = {state.stacks[0], state.stacks[1]},
        };
        history.push(record);
        double nodeValue = traverse(next, history, regretBuffer, strategyBuffer, traverserSeat);
        history.pop();

        StrategySample strategySample(obs, policyVector, traverserSeat);
        strategyBuffer.push(strategySample);

        return nodeValue;
    }
}

} // namespace pkrbot::engine