from ..framework import Agent
from cosp.utils.misc import _debug
from cosp.utils import cfg
cfg.DEBUG_LEVEL = 1

class HierarchicalPlanningAgent(Agent):

    """
    an Agent that makes high-level decisions with a high-level
    planner and then carries out low-level actions with a low-level
    planner.
    """

    def __init__(self,
                 high_level_pomdp):
        self.high_level_pomdp = high_level_pomdp
        self.low_level_pomdp = None
        self._last_decision = None

    def act(self):
        """Hierarchical planning.
        The high-level POMDP will always make a decision;
        If this decision is different from the previous one,
        then it overwrites the current low-level POMDP.
        Action is obtained by solving the low-level POMDP."""
        _debug("Planning high level decision...")
        decision = self.high_level_pomdp.plan_step()
        pomdp_args = self._decision_made(decision)
        _debug("  Decision made: {}".format(decision), c="green")
        if self._last_decision is None:
            _debug("Decision changed. Making new low level POMDP", c="blue")
            self.low_level_pomdp = decision.form_pomdp(pomdp_args)
        _debug("Planning low level POMDP")
        pomdp_action = self.low_level_pomdp.plan_step()
        action = self._action_computed(pomdp_action)
        self._last_decision = self._last_decision
        return action

    def _decision_made(self, decision):
        raise NotImplementedError

    def _action_computed(self, pomdp_action):
        raise NotImplementedError

    def update(self, action, observation):
        """
        The beliefs in both high-level and low-level POMDP
        are updated given the same observation and action;
        Indeed, the action is part of the decision.
        """
        _debug("Updating beliefs")
        observation, reward = observation
        _debug("Updating high level POMDP belief...")
        self.high_level_pomdp.update(self._last_decision, observation)
        _debug("Updating low level POMDP belief...")
        self.low_level_pomdp.update(action, observation)

    @property
    def high_level_belief(self):
        return self.high_level_pomdp.belief

    @property
    def low_level_belief(self):
        return self.low_level_pomdp.belief
