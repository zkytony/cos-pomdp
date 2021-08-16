import random
import pomdp_py
from ..domain.state import CosState2D

class CosJointBelief(pomdp_py.OOBelief):
    def __init__(self, object_beliefs):
        super().__init__(object_beliefs)

    def random(self, rnd=random):
        return CosState2D(super().random(rnd=random, return_oostate=False))

    def mpe(self, rnd=random):
        # import pdb; pdb.set_trace()
        return CosState2D(super().mpe(return_oostate=False))

    def b(self, objid):
        return self.object_beliefs[objid]
