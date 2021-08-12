import math
import matplotlib.pyplot as plt
from cosp.models.fansensor import FanSensor
from cosp.utils.math import to_rad
from cosp.utils.plotting import plot_pose

def plot_laser_sensor_geometry(sensor, dim, robot_pose,
                               ax, num_samples=1000,
                               discrete=True, color="darkorange", alpha=1.0):
    w, l = dim
    if discrete:
        accepted_x = []
        accepted_y = []
        locations = [(x,y) for x in range(w) for y in range(l)]
        for point in locations:
            if sensor.in_range(point, robot_pose):
                accepted_x.append(point[0])
                accepted_y.append(point[1])
        ax.scatter(accepted_x, accepted_y, zorder=2)

    plot_pose(ax, robot_pose[0:2], robot_pose[2])
    samples_x = []
    samples_y = []
    for i in range(num_samples):
        point = sensor.uniform_sample_sensor_region(robot_pose)
        samples_x.append(point[0])
        samples_y.append(point[1])
    ax.scatter(samples_x, samples_y, zorder=1, s=50, color=color, alpha=alpha)
    ax.set_xlim(0, w)
    ax.set_ylim(0, l)
    ax.set_aspect("equal")

if __name__ == "__main__":
    w = 10
    l = 10
    robot_pose = (2, 4, to_rad(0))
    sensor = FanSensor(fov=75, min_range=0, max_range=4)
    fig, ax = plt.subplots()
    plot_laser_sensor_geometry(sensor, (w,l), robot_pose, ax, num_samples=1000)

    plt.show(block=False)
    plt.pause(3)
    plt.close()
