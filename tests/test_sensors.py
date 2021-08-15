import pytest
import random
import math
import matplotlib.pyplot as plt
from cospomdp.models.sensors import FanSensor, FrustumCamera
from cospomdp.utils.math import to_rad
from cospomdp.utils.plotting import plot_pose

@pytest.fixture
def fansensor():
    return FanSensor(fov=75, min_range=0, max_range=4)

@pytest.fixture
def dim():
    return (30, 30)

@pytest.fixture
def robot_pose():
    return (2, 4, to_rad(0))

@pytest.fixture
def show_plots():
    return False

def test_fansensor_geometry(fansensor, dim, show_plots):
    if show_plots:
        fig, ax = plt.subplots()

    w, l = dim
    thetas = [0, 30, 60, 90, 135, 180, 225, 275, 360]
    for i in range(len(thetas)):
        x = random.uniform(2, w-2)
        y = random.uniform(2, l-2)
        th = thetas[i]
        robot_pose = (x, y, th)

        if show_plots:
            plot_pose(ax, robot_pose[0:2], robot_pose[2], color='red')

        samples_x = []
        samples_y = []
        for i in range(500):
            point = fansensor.uniform_sample_sensor_region(robot_pose)
            samples_x.append(point[0])
            samples_y.append(point[1])
            assert fansensor.in_range(point, robot_pose)

        if show_plots:
            ax.scatter(samples_x, samples_y, zorder=1, s=50, alpha=0.6)
            ax.set_xlim(0, w)
            ax.set_ylim(0, l)
            ax.set_aspect("equal")
    if show_plots:
        plt.show(block=False)
        plt.pause(1)
        plt.close()


@pytest.fixture
def camera():
    return FrustumCamera(fov=90, aspect_ratio=1.0, near=1, far=5)

def test_frustum_camera(camera, show_plots):
    points = []
    w, l, h = 30, 30, 30
    for i in range(500):
        x = random.uniform(0,w)
        y = random.uniform(0,l)
        z = random.uniform(0,h)
        points.append((x,y,z))

    if show_plots:
        fig = plt.gcf()
        ax = fig.add_subplot(1,1,1,projection="3d")

    for i, th in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        pose = (15, 15, 15, 0, th, 0)
        if show_plots:
            plot_camera_fov(camera, pose, (w,l,h), points, ax)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Test FrustumCamera (th=%d)" % th)
            if i == 0:
                plt.show(block=False)
            plt.pause(1)
            ax.clear()

        points = camera.get_volume(pose)
        for p in points:
            assert camera.in_range(p, pose)
        if show_plots:
            px = points[:, 0]
            py = points[:, 1]
            pz = points[:, 2]
            ax.scatter(px, py, pz)

def plot_camera_fov(camera, pose, dim, points, ax):
    w, l, h = dim
    px = []
    py = []
    pz = []
    pc = []
    for x, y, z in points:
        if camera.in_range((x,y,z), pose):
            pc.append("gray")
            px.append(x)
            py.append(y)
            pz.append(z)
    ax.scatter(px, py, pz, c=pc)
    ax.set_xlim(-1, w)
    ax.set_ylim(-1, l)
    ax.set_zlim(-1, h)
