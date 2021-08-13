import random
from pomdp_py import GenerativeDistribution, Histogram, OOBelief
from .state import JointState2D, ObjectState2D
from ..utils.math import normalize

class LocBelief2D(Histogram):
    """Intended for high-level belief over an object's locations."""
    @classmethod
    def uniform(cls, objclass, search_region):
        """
        Args:
            objclass: Class of object belief is formed
            search_region: locations the object could be in
            prior: maps from location to a probability, or "uniform"
        """
        hist = {}
        for loc in search_region:
            s = ObjectState2D(objclass, dict(loc=loc))
            hist[s] = 1.0
        return LocBelief2D(normalize(hist))

    @classmethod
    def informed(cls, objclass, true_loc, search_region):
        hist = {}
        import pdb; pdb.set_trace()
        for loc in search_region:
            s = ObjectState2D(objclass, dict(loc=loc))
            if loc == true_loc:
                hist[s] = 1.0
            else:
                hist[s] = 1e-12
        return LocBelief2D(normalize(hist))

class JointBelief2D(OOBelief):
    def __init__(self, robot_id, target_id, robot_belief, target_belief):
        self.robot_id = robot_id
        self.target_id = target_id
        super().__init__({robot_id: robot_belief,
                          target_id: target_belief})

    def random(self, rnd=random):
        return JointState2D(self.robot_id, self.target_id,
                            super().random(rnd=random, return_oostate=False))
    def mpe(self, rnd=random):
        # import pdb; pdb.set_trace()
        return JointState2D(self.robot_id, self.target_id,
                            super().mpe(return_oostate=False))
    @property
    def target_belief(self):
        return self.object_beliefs[self.target_id]
    @property
    def robot_belief(self):
        return self.object_beliefs[self.robot_id]
