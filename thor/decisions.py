# import pomdp_py
# from ..framework import Decision, Action
# from .low_level import (LowLevelTransitionModel,
#                         LowLevelObservationModel,
#                         LowLevelRobotState,
#                         LowLevelObservation,
#                         LowLevelOOBelief,
#                         RobotPoseSensorModel,
#                         MoveAction)
# from . import constants

# class MoveDecision(Decision):
#     def __init__(self, dest):
#         super().__init__("move-to-{}".format(dest))
#         self.dest = dest

#     def form_pomdp(self, pomdp_args):
#         return MoveDecision.POMDP(self, **pomdp_args)

#     class RewardModel(pomdp_py.RewardModel):
#         """This is a RewardModel for the low-level POMDP"""
#         def __init__(self, dest):
#             self.dest = dest

#         def sample(self, state, action, next_state):
#             position, rotation = next_state.robot_state["pose"]
#             x, _, z = position
#             if (x,z) == self.dest:
#                 return constants.TOS_REWARD_HI
#             else:
#                 return constants.TOS_REWARD_STEP

#     class PolicyModel(pomdp_py.UniformPolicyModel):
#         """This is a RewardModel for the low-level POMDP"""
#         def __init__(self, move_actions):
#             """move actions: list of MoveActions"""
#             self.move_actions = move_actions
#             self.action_prior = None
#             super().__init__(move_actions)

#     class POMDP(pomdp_py.Agent):
#         """POMDP for MoveDecision"""
#         def __init__(self, parent_decision, **pomdp_args):
#             self.robot_id = pomdp_args["robot_id"]
#             init_robot_pose = pomdp_args["robot_pose"]
#             move_actions = pomdp_args["move_actions"]
#             planning_config = pomdp_args["planning_config"]

#             transition_model = LowLevelTransitionModel()
#             observation_model = LowLevelObservationModel(RobotPoseSensorModel())
#             policy_model = MoveDecision.PolicyModel(move_actions)
#             reward_model = MoveDecision.RewardModel(parent_decision.dest)

#             # TODO: Use New Belief
#             init_robot_belief = pomdp_py.Histogram({LowLevelRobotState(init_robot_pose): 1.0})
#             init_belief = LowLevelOOBelief(self.robot_id, {self.robot_id: init_robot_belief})
#             super().__init__(init_belief, policy_model,
#                              transition_model,
#                              observation_model,
#                              reward_model)

#             self.parent_decision = parent_decision
#             self._planner =\
#                 pomdp_py.POUCT(max_depth=planning_config["max_depth"],
#                                discount_factor=planning_config["discount_factor"],
#                                num_sims=planning_config["num_sims"],
#                                exploration_const=planning_config["exploration_const"],
#                                rollout_policy=policy_model,
#                                action_prior=policy_model.action_prior)
#         def plan_step(self):
#             action = self._planner.plan(self)
#             self.debug_last_plan()
#             return action

#         def update(self, tos_action, tos_observation):
#             """
#             action (TOS_Action): TOS_Action executed by environment, planned by low level planner
#             tos_observation (TOS_Observation): image, depth image, detections.
#             """
#             # action = MoveAction.from_tos_action(tos_action)
#             # next_robot_state = self.transition_model.sample(self.belief.mpe(), action).robot_state
#             # next_belief = LowLevelOOBelief(self.robot_id,
#             #                                {self.robot_id: pomdp_py.Histogram({next_robot_state: 1.0})})
#             # observation = LowLevelObservation(next_robot_state["pose"])
#             # self._planner.update(self, action, observation)
#             # self.set_belief(next_belief)

#         def debug_last_plan(self):
#             pomdp_py.print_preferred_actions(self.tree)



# class SearchDecision(Decision):
#     def __init__(self):
#         super().__init__("search")

# class DoneDecision(Decision):
#     def __init__(self):
#         super().__init__("done")
