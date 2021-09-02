import random
import time
import numpy as np
import matplotlib.pyplot as plt
from thortils.utils.visual import GridMapVisualizer
from thortils.grid_map import GridMap
from cospomdp.utils.math import indicies2d, normalize
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

worldstr1 =\
"""
......
x.....
......
......
"""

def _test_topo_map_sampling(worldstr,
                            num_samples=10,
                            seed=100,
                            sleep=10,
                            sep=2.0,
                            degree=2):
    print("Test topo map sampling")
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

    target_hist = normalize(target_hist)
    topo_map = _sample_topo_map(target_hist,
                                reachable_positions,
                                num_samples,
                                degree=degree,
                                sep=2.0,
                                rnd=random.Random(seed))
    print(topo_map.total_prob(target_hist))
    print("Degrees:")
    for nid in topo_map.nodes:
        print("    {}: {}".format(topo_map.nodes[nid], len(topo_map.edges_from(nid))))

    grid_map = GridMap(width, length, obstacles)
    viz = GridMapVisualizer(grid_map=grid_map)
    viz.on_init()
    img = viz.render()
    img = draw_topo(img, topo_map, viz._res, draw_grid_path=True,
                    edge_color=(200, 40, 20))
    viz.show_img(img)
    time.sleep(sleep)

def _test_topo_map_sampling_multiple():
    _test_topo_map_sampling(worldstr, seed=100, sleep=3, num_samples=30, degree=(2,3))
    _test_topo_map_sampling(worldstr, seed=200, sleep=3, num_samples=30)
    _test_topo_map_sampling(worldstr, seed=300, sleep=3, num_samples=30)

if __name__ == "__main__":
    _test_topo_map_sampling_multiple()
