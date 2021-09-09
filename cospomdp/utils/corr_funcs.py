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

    def __init__(self, target, other, distances, nearby_thres, dist_scaling=0.8, reverse=False):
        """
        target (ID, class): the target object
        other (ID, class): the other object
        nearby_thres (float): If target and other have
            an average distance less than this threshold,
            then the correlation distance will be a half
            gaussian, highest at target_loc; Otherwise,
            it will be the complement of that gaussian.
        """
        if type(target) == str:
            target = (target, target)  # just to be consistent with other code
        if type(other) == str:
            other = (other, other)
        self.target = target
        self.other = other
        self._distances = distances

        self._nearby_thres = nearby_thres
        self._mean_dist = dist_scaling*np.mean(distances)
        self._reverse = reverse

    def func(self, target_loc, other_loc, target_id, other_id):
        if target_id != self.target[0]:
            raise ValueError(f"unexpected target id {target_id}")
        if other_id != self.other[0]:
            raise ValueError(f"unexpected other id {other_id}")

        gaussian = pomdp_py.Gaussian([*other_loc],
                                     [[self._mean_dist**2, 0],
                                      [0, self._mean_dist**2]])

        dist = euclidean_dist(target_loc, other_loc)
        if not self._reverse:
            close = self._mean_dist < self._nearby_thres
        else:
            close = self._mean_dist > self._nearby_thres

        if close:
            return gaussian[target_loc]
        else:
            return gaussian[other_loc] - gaussian[target_loc]
