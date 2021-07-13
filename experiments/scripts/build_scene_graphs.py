"""
Warning: This is very slow. It takes several hours to build a graph for a single scene,
due to all the Unity processing.
"""
import os

from cosp.thor.scene_graph import build_scene_graph
from cosp.thor.trial import build_object_search_trial,\
    build_object_search_movements


def main():
    trial = build_object_search_trial("Apple", "class")
    controller = trial._start_controller()

    outdir = "../tmp-test"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "graph-{}.json".format(trial.scene)), "w")as f:
        graph = build_scene_graph(controller,
                                  build_object_search_movements(),
                                  f, indent=4)


if __name__ == "__main__":
    main()
