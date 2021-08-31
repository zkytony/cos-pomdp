import random
import time
import numpy as np
import matplotlib.pyplot as plt
from thortils.utils.visual import GridMapVisualizer
from thortils.grid_map import GridMap
from cospomdp.utils.math import indicies2d
from cospomdp_apps.thor.agent.cospomdp_complete\
    import _sample_topo_map
from cospomdp_apps.thor.agent.components.topo_map\
    import TopoMap, draw_edge, draw_topo, mark_cell


worldstr =\
"""
......
x.....
......
..xxx.
..xxx.
..Xxx.
......
x.....
......
"""

def _test_topo_map_sampling(worldstr, num_samples=10, seed=100, sleep=10, sep=2.0):
    obstacles = set()
    target_hist = {}
    reachable_positions = []

    lines = list(reversed(worldstr.strip().split("\n")))
    length = len(lines)
    width = len(lines[0])

    for y, line in enumerate(lines):
        line = line.strip()
        for x, c in enumerate(line):
            if c == ".":
                reachable_positions.append((x,y))
            elif c == "x":
                target_hist[(x,y)] = 1.0
                obstacles.add((x,y))
            elif c == "X":
                target_hist[(x,y)] = 10.0
                obstacles.add((x,y))

    topo_map = _sample_topo_map(target_hist,
                                reachable_positions,
                                num_samples,
                                degree=3,
                                sep=2.0,
                                rnd=random.Random(seed))
    grid_map = GridMap(width, length, obstacles)
    viz = GridMapVisualizer(grid_map=grid_map)
    viz.on_init()
    img = viz.render()
    img = draw_topo(img, topo_map, viz._res, draw_grid_path=True,
                    edge_color=(200, 40, 20))
    viz.show_img(img)
    time.sleep(sleep)

if __name__ == "__main__":
    _test_topo_map_sampling(worldstr, seed=100, sleep=1, num_samples=30)
    _test_topo_map_sampling(worldstr, seed=200, sleep=1, num_samples=30)
    _test_topo_map_sampling(worldstr, seed=300, sleep=1, num_samples=30)
