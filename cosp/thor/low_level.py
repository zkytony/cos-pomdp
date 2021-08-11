# import random
# import pomdp_py
# from thortils.navigation import transform_pose, convert_movement_to_action

# from . import constants
# from ..utils.math import indicator, normalize, euclidean_dist
# from ..framework import Action

# class MoveAction(Action):
#     def __init__(self, name, delta):
#         """
#         name is e.g. MoveAhead
#         delta is a tuple (forward, h_angle, v_angle)
#         """
#         self.name = name
#         self.delta = delta
#         super().__init__(name)

#     @staticmethod
#     def from_tos_action(tos_action):
#         name, delta = convert_movement_to_action(
#             tos_action.name,
#             {tos_action.name:tos_action.params})
#         return MoveAction(name, delta)

# def robot_pose_transition(robot_pose, action, **kwargs):
#     """
#     Uses the transform_pose function to compute the next pose,
#     given a robot pose and an action.

#     Args:
#         robot_pose (position, rotation)
#         action (MoveAction)
#         see transform_pose for kwargs
#     """
#     return transform_pose(robot_pose, (action.name, action.delta),
#                           grid_size=constants.GRID_SIZE, **kwargs)

# # class LowLevelOOState(pomdp_py.OOState):
# #     def __init__(self, robot_id, object_states):
# #         """
# #         env_object_states: states for environment objects
# #         """
# #         self.robot_id = robot_id
# #         self.object_states = object_states
# #         super().__init__(object_states)

# #     @property
# #     def robot_state(self):
# #         return self.object_states[self.robot_id]

# #     def copy(self):
# #         object_states = {objid: self.object_states[objid].copy()
# #                          for objid in self.object_states}
# #         return LowLevelOOState(self.robot_id, object_states)

# # class LowLevelRobotState(pomdp_py.ObjectState):
# #     def __init__(self, pose):
# #         super().__init__("robot", dict(pose=pose))

# #     def copy(self):
# #         return LowLevelRobotState(self["pose"])

# # class LowLevelObjectState(pomdp_py.ObjectState):
# #     def __init__(self, objclass, attributes):
# #         super().__init__(objclass, attributes)

# #     def copy(self):
# #         return LowLevelObjectState(self.objclass, dict(self.attributes))

# # class LowLevelOOBelief(pomdp_py.OOBelief):
# #     def __init__(self, robot_id, object_beliefs):
# #         self.robot_id = robot_id
# #         super().__init__(object_beliefs)
# #     @property
# #     def robot_belief(self):
# #         return self.object_beliefs[self.robot_id]
# #     def random(self, rnd=random):
# #         return LowLevelOOState(self.robot_id,
# #                                super().random(rnd=random, return_oostate=False))
# #     def mpe(self, rnd=random):
# #         return LowLevelOOState(self.robot_id,
# #                                super().mpe(return_oostate=False))


# class LowLevelTransitionModel(pomdp_py.TransitionModel):
#     def sample(self, state, action):
#         current_robot_pose = state.robot_state["pose"]
#         next_state = state.copy()
#         next_robot_pose = current_robot_pose
#         if isinstance(action, MoveAction):
#             next_robot_pose = robot_pose_transition(
#                 current_robot_pose, action)
#         next_state.robot_state["pose"] = next_robot_pose
#         return next_state

#     def probability(self, next_state, state, action):
#         expected_next_state = self.sample(state, action)
#         return indicator(expected_next_state == next_state, epsilon=1e-12)

# # class LowLevelObservation(pomdp_py.Observation):
# #     def __init__(self, stuff):
# #         self.stuff = stuff

# #     def __eq__(self, other):
# #                                        if isinstance(other, LowLevelObservation):
# #             return self.stuff == other.stuff
# #         return False

# #     def __hash__(self):
# #         return hash(self.stuff)

# # class LowLevelObservationModel(pomdp_py.ObservationModel):
# #     def __init__(self, sensor_model):
# #         self.sensor_model = sensor_model

# #     def sample(self, next_state, action):
# #         """
# #         next_state (LowLevelOOState)
# #         """
# #         return LowLevelObservation(
# #             self.sensor_model.sample(next_state,
# #                                      action))

# #     def probability(self, observation, next_state, action):
# #         """
# #         next_state (LowLevelOOState)
# #         """
# #         return self.sensor_model.probability(observation.stuff,
# #                                              next_state,
# #                                              action)

# # ----------- Sensor models -------------- #
# class RobotPoseSensorModel:
#     def sample(self, next_state, *args):
#         return next_state.robot_state["pose"]

#     def probability(self, z, next_state, *args):
#         return z == next_state.robot_state["pose"]
