# Visualization

import pygame
import cv2
import numpy as np

from thortils import convert_scene_to_grid_map

from ..utils.images import overlay
from ..framework import Visualizer
from ..planning import HierarchicalPlanningAgent
from . import constants

class ThorObjectSearchViz(Visualizer):
    def __init__(self, **config):
        self._res = config.get("res", 30)   # resolution
        self._grid_map = None
        self._linewidth = config.get("linewidth", 1)
        self._bg_path = config.get("bg_path", None)
        self._colors = config.get("colors", {})

    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    @property
    def img_width(self):
        return self._grid_map.width * self._res

    @property
    def img_height(self):
        return self._grid_map.length * self._res

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._grid_map.width, self._grid_map.length
        img = np.full((w*r, l*r, 4), 255, dtype=np.uint8)

        # Make an image of grids
        if self._bg_path is not None:
            bgimg = cv2.imread(self._bg_path, cv2.IMREAD_UNCHANGED)
            bgimg = cv2.resize(bgimg, (w*r, l*r))
            img = overlay(img, bgimg, opacity=1.0)

        for x in range(w):
            for y in range(l):
                if (x, y) in self._grid_map.obstacles:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), self._linewidth)
        return img

    def visualize(self, task_env, agent):
        if self._grid_map is None:
            # First time visualize is called
            self._grid_map = convert_scene_to_grid_map(
                task_env.controller, task_env.scene, constants.GRID_SIZE)
            self.on_init()

        img = self._make_gridworld_image(self._res)

        robot_pose = task_env.get_state().agent_pose
        thor_x, _, thor_z = robot_pose[0]

        # Draw belief about robot
        if isinstance(agent, HierarchicalPlanningAgent):
            mpe_state = agent.high_level_belief.mpe()
            robot_state = mpe_state.robot_state
            robot_grid_pos = self._grid_map.to_grid_pos(*robot_state["pos"])
            print("robot state: true position {}\t believed position {}\t believed grid pos {}"\
                  .format((thor_x, thor_z), robot_state["pos"], robot_grid_pos))
            x, y = robot_grid_pos
            robot_color = self.get_color(mpe_state.robot_id)
            img = self.draw_robot(img, x, y, None, color=robot_color, thickness=5)

        # Draw belief about target
        self.show_img(img)

    def show_img(self, img):
        """
        Internally, the img origin (0,0) is top-left (that is the opencv image),
        so +x is right, +z is down.
        But when displaying, to match the THOR unity's orientation, the image
        is flipped, so that in the displayed image, +x is right, +z is up.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 1)  # flip horizontally
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()

    def get_color(self, objid, default=(220, 150, 10, 255), alpha=1.0):
        color = self._colors.get(objid, default)
        if len(color) == 3:
            color = color + [int(round(alpha*255))]
        color = tuple(color)
        return color

    ### Functions to draw
    def draw_robot(self, img, x, y, th, color=(255, 150, 0), thickness=2):
        """Note: agent by default (0 angle) looks in the +z direction in Unity,
        which corresponds to +y here. That's why I'm multiplying y with cos."""
        size = self._res
        x *= self._res
        y *= self._res

        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=thickness)

        if th is not None:
            endpoint = (y+shift + int(round(shift*math.cos(th))),
                        x+shift + int(round(shift*math.sin(th))))
            cv2.line(img, (y+shift,x+shift), endpoint, color, 2)
        return img
