# Visualization
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # hide "hello from pygame community"x
import pygame

import cv2
import numpy as np
import math

from thortils import convert_scene_to_grid_map

from cospomdp.utils.visual import Visualizer2D, BasicViz2D
from cospomdp.utils.math import to_rad
from cospomdp.utils.images import overlay, cv2shape
from cospomdp.utils.colors import lighter, lighter_with_alpha
from . import constants

class GridMapVizualizer(Visualizer2D):
    def __init__(self, **config):
        self._grid_map = config.get("grid_map", None)
        super().__init__(**config)
        self._region = self._grid_map

    def render(self):
        return self._make_gridworld_image(self._res)

    def highlight(self, img, locations, color=(128,128,128), shape="rectangle", thor=False):
        if thor:
            locations = [self._grid_map.to_grid_pos(thor_x, thor_y)
                         for thor_x, thor_y in locations]
        return super().highlight(img, locations, color=color, shape=shape)


class ThorObjectSearchViz2D(GridMapVizualizer):
    def __init__(self, **config):
        super().__init__(**config)

    def visualize(self, task_env, agent, step):
        if self._grid_map is None:
            # First time visualize is called
            self._grid_map = agent.grid_map
            self._region = self._grid_map

        objlocs = {}
        for objid in agent.detectable_objects:
            thor_x, _, thor_z = task_env.get_object_loc(objid)
            loc = agent.grid_map.to_grid_pos(thor_x, thor_z)
            objlocs[objid] = loc

        BasicViz2D.visualize(self, agent.cos_agent,
                             objlocs, draw_fov=step > 0)
