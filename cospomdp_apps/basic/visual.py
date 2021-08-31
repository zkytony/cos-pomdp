from thortils.utils.visual import Visualizer2D, GridMapVisualizer
from thortils.utils.colors import inverse_color_rgb

class BasicViz2D(Visualizer2D):
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
        target_belief = agent.belief.b(agent.target_id)

        if img is None:
            img = self._make_gridworld_image(self._res)
        x, y, th = robot_state["pose"]
        for objid in sorted(objlocs):
            img = self.highlight(img, [objlocs[objid]], self.get_color(objid, colors, alpha=None))
        target_id = agent.target_id
        target_color = self.get_color(target_id, colors, alpha=None)
        if draw_belief:
            img = self.draw_object_belief(img, target_belief, list(target_color) + [250])
        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        if draw_fov is not None:
            if draw_fov is True:
                img = self.draw_fov(img,
                                    agent.sensor(agent.target_id),
                                    robot_state['pose'],
                                    inverse_color_rgb(target_color))
            elif hasattr(draw_fov, "__len__"):
                for objid in sorted(draw_fov):
                    img = self.draw_fov(img,
                                        agent.sensor(objid),
                                        robot_state['pose'],
                                        inverse_color_rgb(self.get_color(objid, colors, alpha=None)))
        return img
