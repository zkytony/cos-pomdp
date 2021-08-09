# Visualization

import pygame
import cv2
import numpy as np

from thortils import convert_scene_to_grid_map

from ..utils.images import overlay
from ..framework import Visualizer
from . import constants

class ThorObjectSearchViz(Visualizer):
    def __init__(self, **config):
        self._res = config.get("res", 30)   # resolution
        self._grid_map = None
        self._linewidth = config.get("linewidth", 1)
        self._bg_path = config.get("bg_path", None)

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
