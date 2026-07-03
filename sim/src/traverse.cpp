#include "traverse.h"

#include <array>
#include <variant>

#include "abstraction.h"

namespace pkrbot::engine {

double traverse(const StateResult& result, ActionHistory& history, PolicyProvider& policy,
                SampleBuffer<RegretSample>& regretBuffer,
                SampleBuffer<StrategySample>& strategyBuffer, int traverserSeat, int iteration,
                std::mt19937_64& rng) {
  if (std::holds_alternative<TerminalState>(result)) {
    const TerminalState& terminalState = std::get<TerminalState>(result);
    return terminalState.deltas[traverserSeat];
  }

  const RoundState& state = std::get<RoundState>(result);

  int active = state.active();
  InfoSet obs = makeInfoSet(state, history);
  PolicyVector policyVector;
  // A single call per decision point; when `policy` is a BatchedPolicy this
  // blocks while the shared batcher coalesces it with other workers' requests
  // into one backend evaluation (see inference.h).
  policy.evaluate(&obs, 1, &policyVector);

  if (active == traverserSeat) {
    // Traverser's turn: evaluate every legal action. Illegal actions keep
    // their zero-initialized regret.
    std::array<double, NUM_ABSTRACT_ACTIONS> regret{};
    for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
      if (obs.legalActionMask[i] == 1) {
        Action action = concretizeAction(state, static_cast<AbstractAction>(i));
        StateResult next = state.proceed(action);
        ActionRecord record = {
            .player = active,
            .street = state.street,
            .action = static_cast<AbstractAction>(i),
            .concreteAction = action,
        };
        history.push(record);
        regret[i] = traverse(next, history, policy, regretBuffer, strategyBuffer, traverserSeat,
                             iteration, rng);
        history.pop();
      }
    }

    double nodeValue = 0.0;
    for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
      if (obs.legalActionMask[i] == 1) {
        nodeValue += policyVector[i] * regret[i];
      }
    }

    // Convert action values into counterfactual regrets against the node value.
    for (int i = 0; i < NUM_ABSTRACT_ACTIONS; i++) {
      if (obs.legalActionMask[i] == 1) {
        regret[i] -= nodeValue;
      }
    }

    RegretSample regretSample(obs, regret, traverserSeat, iteration);
    regretBuffer.push(regretSample);
    return nodeValue;
  }

  // Opponent's turn: sample one action from the policy.
  int sampledActionIndex = samplePolicy(policyVector, rng);
  AbstractAction sampledAction = static_cast<AbstractAction>(sampledActionIndex);
  Action action = concretizeAction(state, sampledAction);
  StateResult next = state.proceed(action);
  ActionRecord record = {
      .player = active,
      .street = state.street,
      .action = sampledAction,
      .concreteAction = action,
  };
  history.push(record);
  double nodeValue = traverse(next, history, policy, regretBuffer, strategyBuffer, traverserSeat,
                              iteration, rng);
  history.pop();

  StrategySample strategySample(obs, policyVector, traverserSeat, iteration);
  strategyBuffer.push(strategySample);

  return nodeValue;
}

}  // namespace pkrbot::engine
