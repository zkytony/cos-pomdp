import math
import cv2
import numpy as np
import pygame

from .images import overlay, cv2shape
from .colors import lighter, lighter_with_alpha
from .math import to_rad

class Visualizer:

    def __init__(self, **config):
        self._res = config.get("res", 30)   # resolution
        self._region = config.get("region", None)
        self._linewidth = config.get("linewidth", 1)
        self._bg_path = config.get("bg_path", None)
        self._colors = config.get("colors", {})
        self._initialized = False

    @property
    def img_width(self):
        return self._region.width * self._res

    @property
    def img_height(self):
        return self._region.length * self._res

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
        self._initialized = True


    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._region.width, self._region.length
        img = np.full((w*r, l*r, 4), 255, dtype=np.uint8)

        # Make an image of grids
        if self._bg_path is not None:
            bgimg = cv2.imread(self._bg_path, cv2.IMREAD_UNCHANGED)
            bgimg = cv2.resize(bgimg, (w*r, l*r))
            img = overlay(img, bgimg, opacity=1.0)

        for x in range(w):
            for y in range(l):
                if (x, y) in self._region.obstacles:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), self._linewidth)
        return img

    def visualize(self, **kwargs):
        self.show_img(self.render())

    def render(self):
        return self._make_gridworld_image(self._res)

    def highlight(self, img, locations, color=(128,128,128)):
        r = self._res
        for loc in locations:
            x, y = loc
            cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),color, -1)
        return img

    def show_img(self, img):
        """
        Internally, the img origin (0,0) is top-left (that is the opencv image),
        so +x is right, +z is down.
        But when displaying, to match the THOR unity's orientation, the image
        is flipped, so that in the displayed image, +x is right, +z is up.
        """
        if not self._initialized:
            self.on_init()
            self._initialized = True
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
            endpoint = (y+shift + int(round(shift*math.sin(to_rad(th)))),
                        x+shift + int(round(shift*math.cos(to_rad(th)))))
            cv2.line(img, (y+shift,x+shift), endpoint, color, 2)
        return img

    def draw_object_belief(self, img, belief, color,
                           circle_drawn=None):
        """
        circle_drawn: map from pose to number of times drawn;
            Used to determine size of circle to draw at a location
        """
        if circle_drawn is None:
            circle_drawn = {}
        radius = int(round(self._res / 2))
        size = self._res // 3
        last_val = -1
        hist = belief.get_histogram()
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = lighter_with_alpha(color, 1-hist[state]/last_val)

            if len(color) == 4:
                stop = color[3]/255 < 0.1
            else:
                stop = np.mean(np.array(color[:3]) / np.array([255, 255, 255])) < 0.999

            if not stop:
                tx, ty = state['loc']
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                img = cv2shape(img, cv2.rectangle,
                               (ty*self._res,
                                tx*self._res),
                               (ty*self._res+self._res,
                                tx*self._res+self._res),
                               color, thickness=-1, alpha=color[3]/255)
                last_val = hist[state]
                if last_val <= 0:
                    break
        return img
