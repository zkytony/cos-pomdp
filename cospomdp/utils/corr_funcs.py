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

    def __init__(self, target, other, distances,
                 sigma=0.1, for_gt=False):
        """
        target (ID, class): the target object
        other (ID, class): the other object
        """
        self.target = target
        self.other = other

        if not for_gt:
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

    def gt_func(self, target_loc, other_loc, target_id, other_id):
        """
        Used for the "groundtruth" correlation baselines.

        Here is what I mean by "groundtruth correlation"
        Pr(si | starget) returns 1.0 (or with some little noise, gaussian noise?)
        if si is the **True** target state, regardless
        of starget's value. Basically, as long as you observe the correlated object,
        your belief will be updated so that it'll be highest on the true target location.
        This is the best correlation can ever do.
        """
