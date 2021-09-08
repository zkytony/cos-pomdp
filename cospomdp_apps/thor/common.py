import os
import numpy as np
import time
from thortils.scene import ithor_scene_type
from dataclasses import dataclass, field
from typing import List
from . import constants

MODULE_PATH = os.path.dirname(__file__)

# The path to the directory to models
YOLOV5_MODEL_DIR = os.path.abspath(os.path.join(MODULE_PATH, "../../models/"))

# The path to the directory of data
YOLOV5_DATA_DIR = os.path.abspath(os.path.join(MODULE_PATH, "../../data/"))

# The path to the yolov5 repository
YOLOV5_REPO_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../external/yolov5/"))

# The path to the saved grid maps
GRID_MAPS_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../data/thor/grid_maps"))

# The path to the saved correlational distribtutions
CORR_DISTS_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../data/thor/corr_dists"))


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
    horizon: float
    done: bool = False

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
        """Returns a string as information after each step."""
        raise NotImplementedError

    def execute(self, agent, action):
        state = self.get_state(self.controller)
        if action.name in constants.get_acceptable_thor_actions():
            event = self.controller.step(action=action.name, **action.params)
            event = self.controller.step(action="Pass")   # https://github.com/allenai/ai2thor/issues/538
        else:
            event = self.controller.step(action="Pass")

        next_state = self.get_state(event)
        observation = self.get_observation(event, action, vision_detector=agent.vision_detector)
        reward = self.get_reward(state, action, next_state)
        self.update_history(next_state, action, observation, reward)
        return (observation, reward)

    def update_history(self, next_state, action, observation, reward):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def success(self):
        raise NotImplementedError

    def get_state(self, event_or_controller):
        """Returns groundtruth state"""
        raise NotImplementedError

    def get_observation(self, event, action):
        """Returns groundtruth observation (i.e. correct object detections)"""
        raise NotImplementedError

    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def visualizer(self, **config):
        raise NotImplementedError

    def get_info(self, fields):
        """
        Given a list of field names (e.g. grid_size, grid_map),
        return a dictionary mapping from field name to value.
        """
        raise NotImplementedError


class ThorAgent:
    """
    ThorAgent's act() function outputs a TOS_Action,
    and its update() function takes in a TOS_Action and a TOS_Observation.
    This can be thought of as a wrapper for the low-level interaction
    with ai2thor; Each TOS_Action is a low-level action (e.g. MoveAhead),
    and each TOS_Observation is a low-level observation (e.g. image, object detections etc.)
    """
    AGENT_USES_CONTROLLER = False

    def act(self):
        raise NotImplementedError

    def update(self, tos_action, tos_observation):
        raise NotImplementedError

    def movement_params(self, move_name):
        """Returns the parameter dict used for ai2thor Controller.step
        for the given move_name"""
        return self.task_config["nav_config"]["movement_params"][move_name]

    @property
    def vision_detector(self):
        return None


@dataclass(init=True)
class TaskArgs:
    detectables: set
    agent_init_inputs: List = field(default_factory=lambda: [])  # inputs e.g. grid map provided at agent creation
    scene: str = 'FloorPlan1'
    target: str = "Apple"
    task_env: str = "ThorObjectSearch"
    agent_class: str = "ThorObjectSearchOptimalAgent"
    max_steps: int = constants.MAX_STEPS
    # load grid maps
    grid_maps_path: str = GRID_MAPS_PATH
    save_grid_map: bool = True
    # use & load corr dists
    save_load_corr: bool = False
    corr_dists_path: str = CORR_DISTS_PATH
    # yolo
    use_vision_detector: bool = False
    yolov5_model_dir: str = YOLOV5_MODEL_DIR,
    yolov5_data_dir: object = YOLOV5_DATA_DIR,


# Make configs
def make_config(args):
    thor_config = {**constants.CONFIG, **{"scene": args.scene}}

    task_config = {
        "robot_id": "robot",
        "task_type": 'class',
        "scene": args.scene,
        "target": args.target,
        "detectables": args.detectables,
        "nav_config": {
            "goal_distance": constants.GOAL_DISTANCE,
            "v_angles": constants.V_ANGLES,
            "h_angles": constants.H_ANGLES,
            "diagonal_ok": constants.DIAG_MOVE,
            "movement_params": thor_config["MOVEMENT_PARAMS"]
        },
        "reward_config": {
            "hi": constants.TOS_REWARD_HI,
            "lo": constants.TOS_REWARD_LO,
            "step": constants.TOS_REWARD_STEP,
        },
        "discount_factor": 0.99,
        "paths": {}
    }
    if args.use_vision_detector:
        # yolov5 model path is the path to models/directory
        scene_type = ithor_scene_type(args.scene)
        model_path = os.path.join(args.yolov5_model_dir, f"yolov5-{scene_type}", "best.pt")
        data_config = os.path.join(args.yolov5_data_dir, f"yolov5-{scene_type}", f"yolov5-{scene_type}-dataset.yaml")
        task_config["paths"]["yolov5_model_path"] = model_path
        task_config["paths"]["yolov5_data_config"] = data_config
    if "grid_map" in args.agent_init_inputs:
        task_config["paths"]["grid_maps_path"] = args.grid_maps_path
        task_config["save_grid_map"] = args.save_grid_map

    task_config["save_load_corr"] = args.save_load_corr
    if args.save_load_corr:
        task_config["paths"]["corr_dists_path"] = args.corr_dists_path

    config = {
        "thor": thor_config,
        "max_steps": args.max_steps,
        "task_config": task_config,
        "task_env": args.task_env,
        "agent_class": args.agent_class,
        "agent_config": {},
        "agent_init_inputs": args.agent_init_inputs
    }

    # You are expected to modify config['agent_config']
    # afterwards to tailor to your agent.
    return config
