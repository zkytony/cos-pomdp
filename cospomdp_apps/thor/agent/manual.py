# manual keyboard controlled agent

from thortils.utils import getch
from cospomdp.domain.action import Done
from ..common import TOS_Action, ThorAgent
from .components.action import (grid_navigation_actions2d,
                                from_grid_action_to_thor_action_params,
                                grid_camera_look_actions,
                                Move)

def print_controls(controls):
    reverse = {controls[k]:k for k in controls}
    ss =f"""
            {reverse['MoveAhead']}
        (MoveAhead)

    {reverse['RotateLeft']}                 {reverse['RotateRight']}
(RotateLeft)     (RotateRight)

    {reverse['LookUp']}
(LookUp)

    {reverse['LookDown']}
(LookDown)

    q
(quit)
    """
    print(ss)


class ThorObjectSearchKeyboardAgent(ThorAgent):
    """
    An agent whose actions are a result of keyboard control.
    """
    def __init__(self, task_config, grid_map):
        super().__init__(task_config)
        self.task_config = task_config
        self.grid_map = grid_map
        self.detectable_objects = task_config["detectables"]
        self.robot_id = task_config['robot_id']

        self.movement_params = self.task_config["nav_config"]["movement_params"]
        self._kb_mapping = {
            "w": "MoveAhead",
            "a": "RotateLeft",
            "d": "RotateRight",
            "e": "LookUp",
            "c": "LookDown",
            "q": "Done"
        }
        print_controls(self._kb_mapping)

    def act(self):
        while True:
            k = getch()
            if k == "q":
                # done
                return TOS_Action("done", {})
            if k in self._kb_mapping:
                name = self._kb_mapping[k]
                params = self.movement_params[name]
                return TOS_Action(name, params)
            print(f"{k} is not valid")

    def update(self, tos_action, tos_observation):
        # Nothing to do.
        pass
