import pomdp_py
import pytest
import matplotlib.pyplot as plt
from cospomdp.models.observation_model import (CosObjectObservationModel2D,
                                               CosObservationModel2D,
                                               FanModelYoonseon,
                                               FanModelNoFP)
from cospomdp.models.correlation import CorrelationDist
from cospomdp.domain.state import ObjectState2D, CosState2D, RobotState2D, RobotStatus
from cospomdp.domain.observation import Loc2D, CosObservation2D
from cospomdp.utils.math import euclidean_dist, normalize
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.utils.plotting import plot_pose
import numpy as np


CORR = {
    (0, 1): 0.7,
    (0, 2): -0.6,
    (0, 3): 0.2,
}

def corr_func(target_loc, object_loc, target_id, object_id):
    corr = CORR[(target_id, object_id)]
    dist = euclidean_dist(target_loc, object_loc)
    if corr > 0:
        return dist <= 2.0
    else:
        return dist >= 2.0

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
    target = (0, "target")
    other = (1, "other")
    robot_id = -1
    fan_params = dict(fov=90, min_range=0, max_range=3)
    detector_target = FanModelNoFP(other[0], fan_params, (0.9, 0.1), round_to=None)

    fan_params = dict(fov=90, min_range=0, max_range=5)
    detector_other = FanModelNoFP(target[0], fan_params, (0.7, 0.1), round_to=None)

    omodel_target = CosObjectObservationModel2D(target[0], target[0], robot_id, detector_target)
    corr_dist = CorrelationDist(other, target, search_region, corr_func)
    omodel_other = CosObjectObservationModel2D(other[0], target[0], robot_id, detector_target, corr_dist)
    omodel = CosObservationModel2D(target[0], {0:omodel_target, 1:omodel_other})

    srobot = RobotState2D(robot_id, (5, 5.5, 0), RobotStatus())
    uniform_belief = pomdp_py.Histogram(normalize({ObjectState2D(target[0], target[1], loc):1.0
                                                   for loc in search_region}))
    new_belief = {}
    other_loc = (7, 5)
    z_other = Loc2D(other[0], other_loc)
    z_target = Loc2D(other[0], None)
    z = CosObservation2D({target[0]: z_target, other[0]: z_other})
    for starget in uniform_belief:
        s = CosState2D({target[0]: starget, robot_id:srobot})
        new_belief[starget] = omodel.probability(z, s) * uniform_belief[starget]
        print(omodel.probability(z, s))
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
        plt.pause(3)
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
