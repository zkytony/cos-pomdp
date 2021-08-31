# Visualization
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # hide "hello from pygame community"

from cospomdp_apps.basic.visual import GridMapVisualizer, BasicViz2D
from .agent.components.topo_map import draw_topo
from . import constants


class ThorObjectSearchViz2D(GridMapVisualizer):
    def __init__(self, **config):
        super().__init__(**config)
        self._draw_topo = config.get("draw_topo", True)
        self._draw_topo_grid_path = config.get("draw_topo_grid_path", False)

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

        img = self._make_gridworld_image(self._res)
        if hasattr(agent, "topo_map") and self._draw_topo:
            # Draw topo map
            img = draw_topo(img, agent.topo_map, self._res,
                            draw_grid_path=self._draw_topo_grid_path)

        return BasicViz2D.render(self, agent.cos_agent,
                                 objlocs, draw_fov=step > 0, img=img)
