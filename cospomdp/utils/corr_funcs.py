import pomdp_py
import numpy as np
from cospomdp.utils.math import euclidean_dist

def around(loc1, loc2, objid1, objid2, d=None):
    return euclidean_dist(loc1, loc2) <= d


class ConditionalSpatialCorrelation:
    """
    Represents the model Pr(s_i | s_target) and provides
    a straightfoward way to learn this distribution based on
    distances of instances of si and starget.
    """

    def __init__(self, target, other, distances, sigma=0.5):
        """
        target (ID, class): the target object
        other (ID, class): the other object
        """
        if type(target) == str:
            target = (target, target)  # just to be consistent with other code
        if type(other) == str:
            other = (other, other)
        self.target = target
        self.other = other

        if len(distances) == 1:
            self._gaussian = pomdp_py.Gaussian([distances[0]],
                                               [sigma**2])
        else:
            self._gaussian = pomdp_py.Gaussian([np.mean(distances)],
                                               [np.var(distances)])



    def func(self, target_loc, other_loc, target_id, other_id):
        if target_id != self.target[0]:
            raise ValueError(f"unexpected target id {target_id}")
        if other_id != self.other[0]:
            raise ValueError(f"unexpected other id {other_id}")
        dist = [euclidean_dist(target_loc, other_loc)]
        return self._gaussian[dist]
