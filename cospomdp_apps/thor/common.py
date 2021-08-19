import numpy as np
import time
from dataclasses import dataclass
from . import constants

# State, Action, Observation used in object search task
@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TOS_Action:
    name: str
    params: dict

@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TOS_State:
    agent_pose: tuple  # position, rotation, in thor coordinates
    horizon: float
    objectlocs: dict   # maps from object class (thor) to (x,y) in thor coordinates;
                       # the instance of this class is the closest one to where the robot is.

@dataclass(init=True, frozen=True)
class TOS_Observation:
    img: np.ndarray
    img_depth: np.ndarray
    detections: list    # list of (xyxy, conf, cls, loc3d)
    robot_pose: tuple

    def __str__(self):
        return ",".join(list(sorted([d[2] for d in self.detections])))

# Generic classes for task and agent in Thor environments.
class ThorEnv:
    def __init__(self, controller):
        self.controller = controller
        self._history = []  # stores the (s', a, o, r) tuples so far
        self._init_state = self.get_state(self.controller)
        self.update_history(self._init_state, None, None, 0)

    @property
    def init_state(self):
        return self._init_state

    def get_step_info(self, step):
        raise NotImplementedError

    def execute(self, agent, action):
        state = self.get_state(self.controller)
        if action.name in constants.get_acceptable_thor_actions():
            event = self.controller.step(action=action.name, **action.params)
        else:
            event = self.controller.step(action="Pass")

        next_state = self.get_state(event)
        observation = self.get_observation(event, vision_detector=agent.vision_detector)
        reward = self.get_reward(state, action, next_state)
        self.update_history(next_state, action, observation, reward)
        return (observation, reward)

    def update_history(self, next_state, action, observation, reward):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def get_state(self, event_or_controller):
        """Returns groundtruth state"""
        raise NotImplementedError

    def get_observation(self, event):
        """Returns groundtruth observation (i.e. correct object detections)"""
        raise NotImplementedError

    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def visualizer(self, **config):
        raise NotImplementedError


@dataclass(init=True, frozen=True)
class TaskArgs:
    detectables: set
    target: str = "Apple"
    max_steps: int = 100
    scene: str = 'FloorPlan1'
    task_env: str = "ThorObjectSearch"
    agent_class: str = "ThorObjectSearchOptimalAgent"
    prior: str = 'uniform'


# Make configs
def make_config(args):
    thor_config = {**constants.CONFIG, **{"scene": args.scene}}

    task_config = {
        "robot_id": "robot",
        "task_type": 'class',
        "target": args.target,
        "detectables": args.detectables,
        "nav_config": {
            "goal_distance": constants.GOAL_DISTANCE,
            "v_angles": constants.V_ANGLES,
            "h_angles": constants.H_ANGLES,
            "diagonal_ok": constants.DIAG_MOVE,
            "movement_params": thor_config["MOVEMENT_PARAMS"]
        },
        "discount_factor": 0.99,
    }

    config = {
        "thor": thor_config,
        "max_steps": args.max_steps,
        "task_env": args.task_env,
        "task_env_config": {"task_config": task_config},
        "agent_class": args.agent_class,
        "agent_config": {"task_config": task_config}
    }

    # You are expected to modify config['agent_config']
    # afterwards to tailor to your agent.
    return config
