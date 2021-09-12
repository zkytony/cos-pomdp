import os
import numpy as np
import time
from thortils.scene import ithor_scene_type
from cospomdp_apps.thor.detector import YOLODetector, GroundtruthDetector
from dataclasses import dataclass, field
from typing import List, Dict
from . import constants
from . import paths


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
    camera_pose: tuple
    horizon: float
    done: bool = False

    @property
    def robot_pose(self):
        return self.camera_pose

    def __str__(self):
        return ",".join(list(sorted([d[2] for d in self.detections])))

    def detections_without_locations(self):
        return [d[:3] for d in self.detections]

# Generic classes for task and agent in Thor environments.
class ThorEnv:
    def __init__(self, controller):
        self.controller = controller
        self._history = []  # stores the history so far
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
        observation = self.get_observation(event, action, detector=agent.detector)
        reward = self.get_reward(state, action, next_state)
        action_to_store, observation_to_store = agent.new_history(action, observation)
        self.update_history(next_state, action_to_store, observation_to_store, reward)
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

    def __init__(self, task_config):
        """
        If task_config['use_vision_detector'] is True, then a YOLOv5 vision detector
        will be loaded with the model_path and data_config provided
        in the task_config.
        """
        if "vision_detector" in task_config["detector_config"]:
            # detector already provided. No need to load.
            self._detector = task_config["detector_config"]["vision_detector"]
        else:
            self._detector = ThorAgent.load_detector(task_config)

    def act(self):
        raise NotImplementedError

    def update(self, tos_action, tos_observation):
        """Update belief and history"""
        raise NotImplementedError

    def new_history(self, tos_action, tos_observation):
        """Given low-level tos_action, tos_obseravtion,
        returns a tuple (action_to_store, obseravtion_to_store)
        that will be stored in the history of the environment;
        This makes it flexible if the agent doesn't only just
        plan the low-level action or directly process
        the low-level observation.
        By default, nothing is done, except that the images
        in the obseravtion will not be stored (save space)."""
        return tos_action, dict(detections=tos_observation.detections_without_locations(),
                                robot_pose=tos_observation.robot_pose,
                                horizon=tos_observation.horizon)


    def movement_params(self, move_name):
        """Returns the parameter dict used for ai2thor Controller.step
        for the given move_name"""
        return self.task_config["nav_config"]["movement_params"][move_name]

    @property
    def detector(self):
        return self._detector

    @staticmethod
    def load_detector(task_config):
        detector_config = task_config["detector_config"]
        use_vision_detector = detector_config.get('use_vision_detector', False)

        shared_kwargs = dict(
            detectables=task_config["detectables"],
            bbox_margin=detector_config['bbox_margin'],
            visualize=detector_config["plot_detections"],
            detection_sep=detector_config["detection_sep"],
            max_repeated_detections=detector_config["max_repeated_detections"],
            detection_ranges=detector_config["expected_detection_ranges"])

        if use_vision_detector:
            model_path = os.path.join(paths.YOLOV5_MODEL_DIR, detector_config["yolov5_model"])
            data_config = os.path.join(paths.YOLOV5_DATA_DIR, detector_config["yolov5_data_config"])
            keep_most_confident = detector_config["keep_most_confident"]
            conf_thres = detector_config["conf_thres"]
            detector = YOLODetector(model_path, data_config,
                                    conf_thres=conf_thres,
                                    keep_most_confident=keep_most_confident,
                                    **shared_kwargs)
        else:
            # uses groundtruth detector
            detector = GroundtruthDetector(**shared_kwargs)

        return detector


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
    save_grid_map: bool = True
    # use & load corr dists
    save_load_corr: bool = False
    # detectors
    use_vision_detector: bool = False
    bbox_margin: float = 0.3 # percentage of the bbox to exclude along each axis
    conf_thres: float = 0.25
    keep_most_confident: bool = True  # if multiple bounding boxes for an object, keep only the most confident one
    plot_detections: bool = False
    detection_sep: float = constants.GRID_SIZE
    max_repeated_detections: int = 1
    # agent detectors
    agent_detector_specs: Dict = field(default_factory=lambda: {})
    # correlations
    corr_specs: Dict = field(default_factory=lambda: {})
    # Belief update
    approx_belief: bool = True

# Make configs
def make_config(args):
    """Make config based on TaskArgs"""
    thor_config = {**constants.CONFIG, **{"scene": args.scene}}

    expected_detection_ranges = {}
    for cls in args.agent_detector_specs:
        detector_spec = args.agent_detector_specs[cls]
        if detector_spec[0] in {"fan-simplefp", "fan-nofp"}:
            expected_detection_ranges[cls] = detector_spec[1]['max_range'] * constants.GRID_SIZE
        elif detector_spec[0] in {"fan-far"}:
            # In this case, we allow detections arbitrarily far. So we set this to be a large number
            expected_detection_ranges[cls] = 100 * constants.GRID_SIZE
        else:
            print(f"WARNING: doesn't know how to deal with {detector_spec[0]}")

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
            "step": constants.TOS_REWARD_STEP
        },
        "detector_config": {
            "use_vision_detector": args.use_vision_detector,
            "bbox_margin": args.bbox_margin,
            "conf_thres": args.conf_thres,
            "keep_most_confident": args.keep_most_confident,
            "plot_detections": args.plot_detections,
            "detection_sep": args.detection_sep,
            "max_repeated_detections": args.max_repeated_detections,
            "expected_detection_ranges": expected_detection_ranges
        },
        "discount_factor": 0.95,
    }
    if task_config["detector_config"]["use_vision_detector"]:
        # yolov5 model path is the path to models/directory
        scene_type = ithor_scene_type(args.scene)
        model_to_use = os.path.join(f"yolov5-{scene_type}", "best.pt")
        data_config = os.path.join(f"yolov5-{scene_type}", f"yolov5-{scene_type}-dataset.yaml")
        task_config["detector_config"]["yolov5_model"] = model_to_use
        task_config["detector_config"]["yolov5_data_config"] = data_config

    if "grid_map" in args.agent_init_inputs:
        task_config["save_grid_map"] = args.save_grid_map

    task_config["save_load_corr"] = args.save_load_corr

    config = {
        "thor": thor_config,
        "max_steps": args.max_steps,
        "task_config": task_config,
        "task_env": args.task_env,
        "agent_class": args.agent_class,
        "agent_config": {},
        "agent_init_inputs": args.agent_init_inputs
    }
    from . import agent as agentlib
    agent_class = eval("agentlib." + config["agent_class"])
    if not (agent_class == agentlib.ThorObjectSearchOptimalAgent)\
       and not (agent_class == agentlib.ThorObjectSearchExternalAgent)\
       and not (agent_class == agentlib.ThorObjectSearchRandomAgent):
        config["agent_config"]["detector_specs"] = args.agent_detector_specs
        config["agent_config"]["corr_specs"] = args.corr_specs
    if not (agent_class == agentlib.ThorObjectSearchGreedyNbvAgent):
        config["agent_config"]["approx_belief"] = args.approx_belief

    # You are expected to modify config['agent_config']
    # afterwards to tailor to your agent.
    return config
