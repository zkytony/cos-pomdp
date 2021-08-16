# import pomdp_py
# from .domain.state import HLObjectState
# from ..utils.math import normalize

# class HighLevelTargetBelief(pomdp_py.GenerativeDistribution):

#     """Belief over high-level target state,
#     which is the location at which the robot is
#     expected to see the object."""

#     def __init__(self, target_class, search_region, prior="uniform"):
#         """
#         search_region: A set of locations (location's meaning is interpreted
#             at the high level, which is where the robot needs to be to observe
#             the target)
#         """
#         self.search_region = search_region
#         if prior == "uniform":
#             self.hist = pomdp_py.Histogram(
#                 normalize({HLObjectState(target_class, {"pos": pos}): 1.0
#                            for pos in search_region}))

#     def update(self, action, observation):
#         for pos in self.search_region:
#             for i, robot_pose in enumerate(robot_poses_at(pos)):
