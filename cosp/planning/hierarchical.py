from ..framework import Agent

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
        decision = self.high_level_pomdp.plan_step()
        if decision != self._last_decision:
            self.low_level_pomdp = decision.form_pomdp()
        action = self.low_level_pomdp.plan_step()
        self._last_decision = self._last_decision
        return action

    def update(self, action, observation):
        """
        The beliefs in both high-level and low-level POMDP
        are updated given the same observation and action;
        Indeed, the action is part of the decision.
        """
        self.high_level_pomdp.update_belief(observation, self._last_decision)
        self.low_level_pomdp.update_belief(observation, action)
