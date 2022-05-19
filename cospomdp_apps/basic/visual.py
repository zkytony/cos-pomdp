# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from thortils.utils.visual import Visualizer2D, GridMapVisualizer
from thortils.utils.colors import inverse_color_rgb
from thortils.utils.images import overlay, cv2shape

class BasicViz2D(Visualizer2D):

    def draw_fov(self, img, sensor, robot_state,
                 color=[233, 233, 8]):
        # We will draw what's in mean range differently from the max range.
        size = self._res // 2
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        for x in range(self._region.width):
            for y in range(self._region.length):
                if hasattr(self._region, "unknown") and (x, y) in self._region.unknown:
                    continue  # occluded (don't draw; the model doesn't care about this though but it is ok for now)

                if robot_state.loc_in_range(sensor, (x,y), use_mean=False):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.4)

                if robot_state.loc_in_range(sensor, (x,y), use_mean=True):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.7)
        return img

    def render(self, agent, objlocs, colors={}, robot_state=None, draw_fov=None,
               draw_belief=True, img=None):
        """
        Args:
            agent (CosAgent)
            robot_state (RobotState2D)
            target_belief (Histogram) target belief
            objlocs (dict): maps from object id to true object (x,y) location tuple
            colors (dict): maps from objid to [R,G,B]
        """
        if robot_state is None:
            robot_state = agent.belief.mpe().s(agent.robot_id)

        if img is None:
            img = self._make_gridworld_image(self._res)
        x, y, th = robot_state["pose"]
        for objid in sorted(objlocs):
            img = self.highlight(img, [objlocs[objid]], self.get_color(objid, colors, alpha=None))
        if draw_belief:
            target_id = agent.target_id
            target_color = self.get_color(target_id, colors, alpha=None)
            target_belief = agent.belief.b(agent.target_id)
            img = self.draw_object_belief(img, target_belief, list(target_color) + [250])
        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        if draw_fov is not None:
            if draw_fov is True:
                img = BasicViz2D.draw_fov(self, img,
                                          agent.sensor(agent.target_id),
                                          robot_state,
                                          inverse_color_rgb(target_color))
            elif hasattr(draw_fov, "__len__"):
                for objid in sorted(draw_fov):
                    img = BasicViz2D.draw_fov(self, img, agent.sensor(objid),
                                              robot_state,
                                              inverse_color_rgb(self.get_color(objid,
                                                                               colors, alpha=None)))
        return img
