# Visualization
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # hide "hello from pygame community"x

from cospomdp_apps.basic.visual import GridMapVizualizer, BasicViz2D
from . import constants


class ThorObjectSearchViz2D(GridMapVizualizer):
    def __init__(self, **config):
        super().__init__(**config)

    def render(self, task_env, agent, step):
        if self._grid_map is None:
            # First time visualize is called
            self._grid_map = agent.grid_map
            self._region = self._grid_map

        objlocs = {}
        for objid in agent.detectable_objects:
            thor_x, _, thor_z = task_env.get_object_loc(objid)
            loc = agent.grid_map.to_grid_pos(thor_x, thor_z)
            objlocs[objid] = loc

        return BasicViz2D.render(self, agent.cos_agent,
                                 objlocs, draw_fov=step > 0)
