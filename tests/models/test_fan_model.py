from cosp.models.state import ObjectState2D
from cosp.models.observation import FanModel
from cosp.utils.math import to_rad, normalize
from cosp.utils.plotting import plot_pose
import matplotlib.pyplot as plt
from pprint import pprint
from test_sensor import plot_laser_sensor_geometry
import time


def test_fan_model():
    w, l = 20, 20
    srobot = ObjectState2D("robot",
                           dict(pose=(1, 6, to_rad(0))))
    si = ObjectState2D("Mug",
                       dict(loc=(5, 6)))

    fan_params = dict(fov=90,
                      min_range=0,
                      max_range=5)

    models = [
        FanModel("Mug", fan_params, (0, 1), round_to=None), # perfect sensor
        FanModel("Mug", fan_params, (0.1, 1), round_to=None),
        FanModel("Mug", fan_params, (0.25, 1), round_to=None),
        FanModel("Mug", fan_params, (0.5, 1), round_to=None),
        FanModel("Mug", fan_params, (0.75, 1), round_to=None),
        FanModel("Mug", fan_params, (1, 1), round_to=None),
        FanModel("Mug", fan_params, (0, 1), round_to=None), # perfect sensor
        FanModel("Mug", fan_params, (0, 0.95), round_to=None), # perfect sensor
        FanModel("Mug", fan_params, (0, 0.9), round_to=None), # perfect sensor
        FanModel("Mug", fan_params, (0, 0.8), round_to=None), # perfect sensor
        FanModel("Mug", fan_params, (0, 0.75), round_to=None),
        FanModel("Mug", fan_params, (0, 0.6), round_to=None),
        FanModel("Mug", fan_params, (0, 0.5), round_to=None),
        FanModel("Mug", fan_params, (0, 0.25), round_to=None)
    ]

    fig, ax = plt.subplots()

    for i, model in enumerate(models):
        ax.set_title("$\sigma, \epsilon$ = {}".format(model.params))
        plot_fan_model_samples(model, si, srobot, (w,l), ax)
        if i == 0:
            plt.show(block=False)
        plt.pause(1.5)
        ax.clear()



def plot_fan_model_samples(model, si, srobot, dim, ax, num_samples=1000):
    w, l = dim
    px = []
    py = []
    pc = []
    counts = {"A":0,"B":0,"C":0}
    for i in range (1000):
        zi, event = model.sample(si, srobot, return_event=True)
        counts[event] += 1
        if zi.loc is not None:
            px.append(zi.loc[0])
            py.append(zi.loc[1])
            if event == "A":
                pc.append("lightgreen")
            elif event == "B":
                pc.append("crimson")
    pprint(normalize(counts))
    plot_laser_sensor_geometry(model.sensor, (w,l), srobot["pose"],
                               ax, discrete=False, alpha=0.5)

    ox, oy = si["loc"]
    plt.scatter([ox], [oy], s=150, color="blue", marker="*")

    plt.scatter(px, py, color=pc, alpha=0.6)

    plot_pose(ax, srobot['pose'][:2], srobot['pose'][2])

    ax.set_xlim(0, w)
    ax.set_ylim(0, l)


if __name__ == "__main__":
    test_fan_model()
