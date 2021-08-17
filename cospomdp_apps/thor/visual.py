# Visualization

import pygame
import cv2
import numpy as np
import math

from thortils import convert_scene_to_grid_map

from cospomdp.utils.visual import Visualizer2D
from cospomdp.utils.math import to_rad
from cospomdp.utils.images import overlay, cv2shape
from cospomdp.utils.colors import lighter, lighter_with_alpha
from . import constants

class GridMapVizualizer(Visualizer2D):
    def __init__(self, **config):
        self._grid_map = config.get("grid_map", None)
        super().__init__(**config)
        self._region = self._grid_map

    def visualize(self, task_env, agent):
        raise NotImplementedError

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

    def visualize(self, task_env, agent):
        if self._grid_map is None:
            # First time visualize is called
            self._grid_map = agent.grid_map

        img = self._make_gridworld_image(self._res)

        # Draw belief about robot
        x, y, th = self._get_robot_grid_pose(agent)
        robot_color = self.get_color(task_env.robot_id)
        img = self.draw_robot(img, x, y, th, color=robot_color, thickness=5)

        # Draw belief about target
        belief = self._get_target_belief(agent)
        target_color = self.get_color(task_env.target_id)
        img = self.draw_object_belief(img, belief, target_color)

        # Draw field of view
        sensor = agent.observation_model.zi_models[agent.target_class].detection_model.sensor
        img = self.draw_fov(img, sensor, (x, y, th))

        self.show_img(img)

    def _get_robot_grid_pose(self, agent):
        mpe_state = agent.belief.mpe()
        robot_state = mpe_state.robot_state
        pos = robot_state["pose"][:2]
        th = robot_state["pose"][2]
        return (*pos, th)

    def _get_target_belief(self, agent):
        return agent.belief.target_belief

    def draw_fov(self, img, sensor, robot_pose):
        size = self._res // 2
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        for x in range(self._grid_map.width):
            for y in range(self._grid_map.length):
                if sensor.in_range((x,y), robot_pose):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, [233, 233, 8], thickness=-1, alpha=0.7)
        return img
