# Visualization
import cv2
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # hide "hello from pygame community"

from thortils.vision import thor_topdown_img
from cospomdp_apps.basic.visual import GridMapVisualizer, BasicViz2D
from cospomdp.domain.state import RobotState2D
from .agent.components.topo_map import draw_topo
from .agent import (ThorObjectSearchCosAgent,
                    ThorObjectSearchRandomAgent,
                    ThorObjectSearchGreedyNbvAgent,
                    ThorObjectSearchKeyboardAgent)
from . import constants
from .replay import ReplaySolver


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

        _draw_topo = hasattr(agent, "topo_map") and self._draw_topo

        # if hasattr(agent, "solver") and isinstance(agent.solver, ReplaySolver):
        #     _draw_topo = False

        objlocs = {}
        for objid in agent.detectable_objects:
            thor_x, _, thor_z = task_env.get_object_loc(objid)
            loc = agent.grid_map.to_grid_pos(thor_x, thor_z)
            objlocs[objid] = loc

        img = self._make_gridworld_image(self._res)
        if _draw_topo:
            # Draw topo map
            img = draw_topo(img, agent.topo_map, self._res,
                            draw_grid_path=self._draw_topo_grid_path)

        if isinstance(agent, ThorObjectSearchCosAgent):
            return BasicViz2D.render(self, agent, objlocs, draw_fov=step > 0, img=img)

        elif isinstance(agent, ThorObjectSearchRandomAgent)\
             or isinstance(agent, ThorObjectSearchKeyboardAgent):
            thor_robot_pos, thor_robot_rot = task_env.get_state().agent_pose
            thor_robot_pose2d = (thor_robot_pos[0], thor_robot_pos[2], thor_robot_rot[1])
            robot_state = RobotState2D(agent.robot_id,
                                       self._grid_map.to_grid_pose(*thor_robot_pose2d))
            return BasicViz2D.render(self, agent, objlocs,
                                     robot_state=robot_state,
                                     draw_fov=False,
                                     draw_belief=False, img=img)
