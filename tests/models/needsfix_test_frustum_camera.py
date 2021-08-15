from cosp.models.state import ObjectState2D
from cosp.models.observation import FrustumModelFull
from cosp.utils.math import to_rad, normalize
from cosp.utils.plotting import plot_pose
from pprint import pprint
from test_sensor import plot_laser_sensor_geometry
import time

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_frustum_model():
    objclass = "Mug"
    camera_params = dict(fov=60,
                         near=0.1,
                         far=10.0,
                         aspect_ratio=1.0)
    quality_params = (1000., 0.0, 1.)
    model = FrustumModelFull(objclass, camera_params, quality_params)

    points = []
    w, l, h = 30, 30, 30
    for i in range(40000):
        x = random.uniform(0,w)
        y = random.uniform(0,l)
        z = random.uniform(0,h)
        points.append((x,y,z))

    fig = plt.gcf()
    ax = fig.add_subplot(1,1,1,projection="3d")
    for i, th in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        pose = (15, 15, 15, 0, th, 0)
        plot_camera_fov(model, pose, (w,l,h), points, ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if i == 0:
            plt.show(block=False)
        plt.pause(1)
        ax.clear()

def plot_camera_fov(model, pose, dim, points, ax):
    w, l, h = dim
    p, r = model.sensor.transform_camera(pose)
    px = []
    py = []
    pz = []
    pc = []
    for x, y, z in points:
        if model.sensor.within_range((p,r), (x,y,z,1)):
            pc.append("gray")
            px.append(x)
            py.append(y)
            pz.append(z)
    ax.scatter(px, py, pz, c=pc)
    ax.set_xlim(-1, w)
    ax.set_ylim(-1, l)
    ax.set_zlim(-1, h)

if __name__ == "__main__":
    test_frustum_model()
