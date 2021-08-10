import pomdp_py
from ..framework import Decision, Action
from .low_level import (LowLevelTransitionModel,
                        LowLevelObservationModel,
                        LowLevelRobotState,
                        LowLevelOOBelief,
                        RobotPoseSensorModel)
from . import constants

class MoveDecision(Decision):
    def __init__(self, dest):
        dest = (-1.0, 1.25)
        super().__init__("move-to-{}".format(dest))
        self.dest = dest

    def form_pomdp(self, pomdp_args):
        robot_id = pomdp_args["robot_id"]
        robot_pose = pomdp_args["robot_pose"]
        move_actions = pomdp_args["move_actions"]
        planning_config = pomdp_args["planning_config"]

        transition_model = LowLevelTransitionModel()
        observation_model = LowLevelObservationModel(RobotPoseSensorModel())
        policy_model = MoveDecision.PolicyModel(move_actions)
        reward_model = MoveDecision.RewardModel(self.dest)
        init_robot_belief = pomdp_py.Histogram({LowLevelRobotState(robot_pose): 1.0})
        init_belief = LowLevelOOBelief(robot_id, {robot_id: init_robot_belief})
        return MoveDecision.POMDP(init_belief,
                                  policy_model,
                                  transition_model,
                                  observation_model,
                                  reward_model,
                                  planning_config,
                                  self)

    class RewardModel(pomdp_py.RewardModel):
        """This is a RewardModel for the low-level POMDP"""
        def __init__(self, dest):
            self.dest = dest

        def sample(self, state, action, next_state):
            position, rotation = next_state.robot_state["pose"]
            x, _, z = position
            if (x,z) == self.dest:
                return constants.TOS_REWARD_HI
            else:
                return constants.TOS_REWARD_STEP

    class PolicyModel(pomdp_py.UniformPolicyModel):
        """This is a RewardModel for the low-level POMDP"""
        def __init__(self, move_actions):
            """move actions: list of MoveActions"""
            self.move_actions = move_actions
            self.action_prior = None
            super().__init__(move_actions)

    class POMDP(pomdp_py.Agent):
        def __init__(self, init_belief,
                     policy_model,
                     transition_model,
                     observation_model,
                     reward_model,
                     planning_config,
                     parent_decision):
            super().__init__(init_belief, policy_model,
                             transition_model,
                             observation_model,
                             reward_model)
            self.parent_decision = parent_decision
            self._planner = pomdp_py.POUCT(max_depth=planning_config["max_depth"],
                                           discount_factor=planning_config["discount_factor"],
                                           num_sims=planning_config["num_sims"],
                                           exploration_const=planning_config["exploration_const"],
                                           rollout_policy=policy_model,
                                           action_prior=policy_model.action_prior)
        def plan_step(self):
            action = self._planner.plan(self)
            self.debug_last_plan()
            return action

        def update(self, action, tos_observation):
            """
            action (Action): Action taken by low level planner
            tos_observation (TOS_Observation): image, depth image, detections.
            """
            next_robot_state = self.transition_model.sample(self.belief.mpe(), action).robot_state
            next_belief = self.parent_decision.update_pomdp_belief(self,
                                                                   action,
                                                                   tos_observation)


            self._planner.update()
            print(next_robot_state)
            raise NotImplementedError

        def debug_last_plan(self):
            pomdp_py.print_preferred_actions(self.tree)



class SearchDecision(Decision):
    def __init__(self):
        super().__init__("search")

class DoneDecision(Decision):
    def __init__(self):
        super().__init__("done")
