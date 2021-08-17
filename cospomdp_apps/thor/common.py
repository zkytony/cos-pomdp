import numpy as np
from dataclasses import dataclass
from . import constants

# State, Action, Observation used in object search task
@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TOS_Action:
    name: str
    params: dict

@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TOS_State:
    agent_pose: tuple
    horizon: float

@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TOS_Observation:
    img: np.ndarray
    img_depth: np.ndarray
    detections: list
    robot_pose: tuple


# Generic classes for task and agent in Thor environments.
class ThorEnv:
    def __init__(self, controller):
        self.controller = controller
        self._history = []  # stores the (s', a, o, r) tuples so far
        self._init_state = self.get_state(self.controller)
        self._history.append((self._init_state, None, None, 0))

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
        observation = self.get_observation(event, detector=agent.detector)
        reward = self.get_reward(state, action, next_state)
        self._history.append((next_state, action, observation, reward))
        return (observation, reward)

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


@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class TaskArgs:
    detectables: set
    target: str
    max_steps: int
    scene: str = 'FloorPlan1'


# Make configs
def make_config(args):
    thor_config = {**constants.CONFIG, **{"scene": args.scene}}

    task_config = {
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
        "discount_factor": 0.99
    }

    config = {
        "thor": thor_config,
        "max_steps": args.max_steps,
        "task_env": "ThorObjectSearch",
        "task_env_config": {"task_config": task_config},
        "agent_class": "ThorObjectSearchOptimalAgent",
        "agent_config": {"task_config": task_config}
    }

    return config
