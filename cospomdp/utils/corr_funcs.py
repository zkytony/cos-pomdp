# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pomdp_py
import numpy as np
from cospomdp.utils.math import euclidean_dist

def around(loc1, loc2, objid1, objid2, d=None):
    return euclidean_dist(loc1, loc2) <= d

def apart(loc1, loc2, objid1, objid2, d=None):
    return euclidean_dist(loc1, loc2) >= d

class ConditionalSpatialCorrelation:
    """
    Represents the model Pr(s_i | s_target) and provides
    a straightfoward way to learn this distribution based on
    distances of instances of si and starget.

    Grid coordinates.
    """

    def __init__(self, target, other, distances,
                 nearby_thres, reverse=False, learned=False):
        """
        target (ID, class): the target object
        other (ID, class): the other object
        nearby_thres (float): If target and other have
            an average distance less than this threshold.
        """
        if type(target) == str:
            target = (target, target)  # just to be consistent with other code
        if type(other) == str:
            other = (other, other)
        self.target = target
        self.other = other
        self._distances = distances

        self._nearby_thres = nearby_thres
        self._mean_dist = np.mean(distances)
        self._reverse = reverse
        self._learned = learned

    @property
    def corr_type(self):
        if self._learned:
            assert self._reverse is False
            return "learned"
        else:
            if self._reverse:
                return "wrong"
            else:
                return "correct"

    def should_be_close(self):
        if not self._reverse:
            return self._mean_dist < self._nearby_thres
        else:
            return self._mean_dist >= self._nearby_thres

    def func(self, target_loc, other_loc, target_id, other_id, **kwargs):
        if target_id != self.target[0]:
            raise ValueError(f"unexpected target id {target_id}")
        if other_id != self.other[0]:
            raise ValueError(f"unexpected other id {other_id}")

        dist = euclidean_dist(target_loc, other_loc)
        if not self._reverse:
            close = self._mean_dist < self._nearby_thres
        else:
            close = self._mean_dist > self._nearby_thres

        if close:
            if dist < self._mean_dist:
                return True
            else:
                return False
        else:
            if dist < self._mean_dist:
                return False
            else:
                return True

    def __str__(self):
        return "SpCorr({}, {})[min_dist:{:.3f}]".format(self.target[1], self.other[1], self._mean_dist)

    def __repr__(self):
        return str(self)
