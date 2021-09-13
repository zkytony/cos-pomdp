import random
import pomdp_py
from ..domain.state import CosState, ObjectState

class CosJointBelief(pomdp_py.OOBelief):
    def __init__(self, object_beliefs):
        super().__init__(object_beliefs)

    def random(self, rnd=random):
        return CosState(super().random(rnd=random, return_oostate=False))

    def mpe(self):
        # import pdb; pdb.set_trace()
        return CosState(super().mpe(return_oostate=False))

    def b(self, objid):
        return self.object_beliefs[objid]

    def set_b(self, objid, belief):
        self.object_beliefs[objid] = belief
