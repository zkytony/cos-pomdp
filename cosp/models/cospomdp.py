# COS-POMDP: Correlation Object Search POMDP
import pomdp_py
from pomdp_py.utils import typ
from ..framework import Agent, Decision
from ..utils.misc import resolve_robot_target_args

class SearchRegion:
    # DOMAIN-SPECIFIC
    """domain-specific / abstraction-specific host of a set of locations. All that
    it needs to support is enumerability (which could technically be implemented
    by sampling)
    """
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __contains__(self):
        pass


class DetectionModel:
    # DOMAIN-SPECIFIC
    """Interface for Pr(zi | si, srobot'); Domain-specific"""
    def __init__(self):
        pass

    def probability(self, zi, si, srobot, a=None):
        """
        zi: object observation
        si: object state
        srobot: robot state
        a (optional): action taken
        """
        raise NotImplementedError

    def sample(self, si, srobot, a=None):
        raise NotImplementedError


# class ReducedState(pomdp_py.OOState):
#     """Reduced state that only contains robot and target states.
#     Both robot_state and target_state are pomdp_py.ObjectState"""
#     def __init__(self, robot_id, target_id, *args):
#         self.robot_id = robot_id
#         self.target_id = target_id
#         robot_state, target_state = resolve_robot_target_args(robot_id, target_id, *args)
#         super().__init__({self.robot_id: robot_state,
#                           self.target_id: target_state})

#     def __str__(self):
#         return\
#             "{}(\n"\
#             "    {},\n"\
#             "    {})".format(type(self), self.robot_state, self.target_state)

#     def __repr__(self):
#         return str(self)

#     @property
#     def robot_state(self):
#         return self.object_states[self.robot_id]

#     @property
#     def target_state(self):
#         return self.object_states[self.target_id]


# class Observation(pomdp_py.Observation):
#     def __init__(self, object_observations):
#         """
#         object_observations (tuple): vector of observations of each object
#         """
#         self.object_observations = object_observations
#         self._hash = hash(self.object_observations)

#     def __eq__(self, other):
#         if isinstance(other, Observation):
#             return self.object_observations == other.object_observations
#         return False

#     def __hash__(self):
#         return self._hash

#     def __repr__(self):
#         obzstr = ["{}:{}".format(o.objclass, o.location)
#                   for o in self.object_observations]
#         return "Obz({})".format(obzstr)

#     def __str__(self):
#         obzstr = ["{}:{}".format(o.objclass, o.location)
#                   for o in self.object_observations]
#         return typ.blue("Obz({})".format(obzstr))

#     def __iter__(self):
#         return iter(self.object_observations)

#     def __len__(self):
#         return len(self.object_observations)


# class ObjectObservation(pomdp_py.SimpleObservation):
#     """Keep it simple for now"""
#     def __init__(self, objclass, location):
#         """
#         Args:
#             objclass (str): Object class
#             location (object): Object location, hashable; None if not observed
#         """
#         self.objclass = objclass
#         self.location = location
#         super().__init__((objclass, location))
