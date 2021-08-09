import pomdp_py
from ..framework import Decision, Action

# # Low




# class LowLevelTransitionModel(pomdp_py.TransitionModel):
#     pass







class MoveDecision(Decision):
    def __init__(self, dest):
        super().__init__("move-to-{}".format(dest))
        self.dest = dest

    # def form_pomdp(self, ...):
    #     transition_model = make_transition_model(robotstate, movements)
    #     observation_model = make_observation_model(robotpose)
    #     policy_model = make_policy_model(movements)
    #     reward_model = reachtodestreward
    #     init_belief = currentdist



        # pass

class SearchDecision(Decision):
    def __init__(self):
        super().__init__("search")

class DoneDecision(Decision):
    def __init__(self):
        super().__init__("done")
