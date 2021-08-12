from cosp.models.state import ObjectState2D, JointState2D
from cosp.models.correlation import CorrelationDist
from cosp.models.observation import FanModelYoonseon, FanModelNoFP, CorrObservationModel, ObjectDetection2D
from cosp.utils.math import to_rad, normalize
from cosp.utils.plotting import plot_pose
from cosp.models.search_region import SearchRegion2D
import matplotlib.pyplot as plt
from pprint import pprint
import time
import numpy as np

from test_sensor import plot_laser_sensor_geometry

from cosp.utils.math import euclidean_dist

# Hard coded correlation
CORR_MATRIX = {
    ("Apple", "CounterTop"): 0.7,
    ("Apple", "Bread"): 0.8,
    ("Bread", "CounterTop"): 0.7,
}
for k in list(CORR_MATRIX.keys()):
    CORR_MATRIX[tuple(reversed(k))] = CORR_MATRIX[k]

def corr_func(target_pos, object_pos,
              target_class, objclass):
    """
    Returns a float value to essentially mean
    Pr(Si = object_pos | Starget = target_pos)
    """
    # This is a rather arbitrary function for now.
    if target_class == objclass:
        corr = 1.0
    else:
        corr = CORR_MATRIX[(target_class, objclass)]
    distance = euclidean_dist(target_pos, object_pos)
    if corr > 0:
        return distance <= 2.0
    else:
        return distance >= 2.0


def test_corr_model():
    objclass = "CounterTop"
    target_class = "Apple"

    w, l = 20, 20
    locations = [(x,y) for x in range(w) for y in range(l)]
    search_region = SearchRegion2D(locations)
    corr_dist = CorrelationDist(objclass, target_class,
                                search_region, corr_func)
    fan_params = dict(fov=90, min_range=0, max_range=5)
    detector = FanModelNoFP("Mug", fan_params, (0.9, 0.1), round_to=None)
    print("Creating corr model...")
    corr_model = CorrObservationModel(objclass, target_class, detector, corr_dist)

    # which area becomes likely of the target if objclass is observed
    fig, ax = plt.subplots()
    uniform_belief = normalize({ObjectState2D(target_class, dict(loc=loc)):1.0
                              for loc in locations})
    plot_belief(uniform_belief, (w,l), ax)
    plt.show(block=False)
    plt.pause(1)
    ax.clear()

    srobot = ObjectState2D("robot",
                           dict(pose=(4, 5.5, to_rad(0))))
    zi = ObjectDetection2D(objclass, (7, 5))
    print("Belief update...")
    new_belief = {}
    for starget in uniform_belief:
        s = JointState2D("robot", starget.objclass,
                         {"robot": srobot, starget.objclass: starget})
        new_belief[starget] = corr_model.probability(zi, s) * uniform_belief[starget]
    new_belief = normalize(new_belief)
    plot_belief(new_belief, (w,l), ax)
    ox, oy = zi.loc
    plt.scatter([ox], [oy], s=150, color="orange", marker="*")
    plot_pose(ax, srobot['pose'][:2], srobot['pose'][2])
    plt.show(block=False)
    print("Waiting to close...")
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

if __name__ == "__main__":
    # test_fan_model_yoonseon()
    test_corr_model()
