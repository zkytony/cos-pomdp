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
import random
import pytest
import matplotlib.pyplot as plt
from cospomdp.models.observation_model import (CosObjectObservationModel,
                                               CosObservationModel,
                                               FanModelYoonseon,
                                               FanModelNoFP)
from cospomdp.models.correlation import CorrelationDist
from cospomdp.domain.state import (ObjectState,
                                   CosState,
                                   RobotState2D,
                                   RobotStatus)
from cospomdp.domain.observation import Loc, CosObservation
from cospomdp.utils.math import euclidean_dist, normalize
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.utils.plotting import plot_pose
import numpy as np


CORR = {
    (0, 1): 0.7,
}

DIST = 2.0

def corr_func(target_loc, object_loc, target_id, object_id):
    corr = CORR[(target_id, object_id)]
    dist = euclidean_dist(target_loc, object_loc)
    if corr > 0:
        return dist <= DIST
    else:
        return dist >= DIST

@pytest.fixture
def search_region():
    w, l = 15, 15
    locations = [(x,y) for x in range(w) for y in range(l)]
    search_region = SearchRegion2D(locations)
    return search_region

@pytest.fixture
def show_plots():
    return True

def test_observation_model(search_region, show_plots):
    """mainly used for testing correlation observation model belief update"""
    target = (0, "target")
    other = (1, "other")
    robot_id = -1
    fan_params = dict(fov=90, min_range=0, max_range=3)
    detector_target = FanModelNoFP(other[0], fan_params, (0.9, 0.1), round_to=None)

    fan_params = dict(fov=90, min_range=0, max_range=5)
    detector_other = FanModelNoFP(target[0], fan_params, (0.7, 0.1), round_to=None)

    omodel_target = CosObjectObservationModel(target[0], target[0], robot_id, detector_target)
    corr_dist = CorrelationDist(other, target, search_region, corr_func)
    omodel_other = CosObjectObservationModel(other[0], target[0], robot_id, detector_target, corr_dist)
    omodel = CosObservationModel(robot_id, target[0], {0:omodel_target, 1:omodel_other})

    srobot = RobotState2D(robot_id, (5, 5.5, 0), RobotStatus())

    # sampling
    for i in range(200):
        loc = random.sample(search_region.locations, 1)[0]
        state = CosState({target[0]: ObjectState(target[0], target[1], loc),
                          robot_id: srobot})
        z = omodel.sample(state)
        if z[target[0]].loc is not None and z[other[0]].loc is not None:
            assert euclidean_dist(z[target[0]].loc, z[other[0]].loc) <= DIST

    # belief update
    uniform_belief = pomdp_py.Histogram(normalize({ObjectState(target[0], target[1], loc):1.0
                                                   for loc in search_region}))
    new_belief = {}
    other_loc = (7, 5)
    z_other = Loc(other[0], other_loc)
    z_target = Loc(other[0], None)
    z = CosObservation(srobot, {target[0]: z_target, other[0]: z_other})
    for starget in uniform_belief:
        s = CosState({target[0]: starget, robot_id:srobot})
        new_belief[starget] = omodel.probability(z, s) * uniform_belief[starget]
    new_belief = pomdp_py.Histogram(normalize(new_belief))
    assert euclidean_dist(new_belief.mpe()['loc'], z_other.loc)\
        < euclidean_dist(uniform_belief.mpe()["loc"], z_other.loc)

    if show_plots:
        fig, ax = plt.subplots()
        plot_belief(new_belief, (15, 15), ax)
        ox, oy = z_other.loc
        plt.scatter([ox], [oy], s=150, color="orange", marker="*")
        plot_pose(ax, srobot['pose'][:2], srobot['pose'][2])
        plt.show(block=False)
        plt.pause(1)
        ax.clear()

def plot_belief(belief, dim, ax):
    x = []
    y = []
    c = []
    for starget in belief:
        x.append(starget['loc'][0])
        y.append(starget['loc'][1])
        c.append(np.array([0.1, 0.5, 0.1, belief[starget]]))
    ax.scatter(x, y, s=100, c=c, marker='s')
    ax.set_xlim(-1, dim[0])
    ax.set_ylim(-1, dim[1])
    ax.set_aspect("equal")
