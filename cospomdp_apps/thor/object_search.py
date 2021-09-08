import os
import random
import math
import numpy as np
from collections import namedtuple
from pprint import pprint

import pomdp_py
from pomdp_py.utils import typ
import ai2thor
import ai2thor.util.metrics as metrics

import thortils as tt
from thortils.utils import euclidean_dist
import thortils.vision.projection as pj

from .result_types import PathResult, HistoryResult
from .common import ThorEnv, TOS_Action, TOS_State, TOS_Observation, ThorAgent
from .agent import ThorObjectSearchOptimalAgent
from .visual import ThorObjectSearchViz2D
from .detector import YOLODetector, GroundtruthDetector
from . import constants


# ------------- Task ------------- #
class TOS(ThorEnv):
    """
    TOS is short for ThorObjectSearch task.
    This represents the environment of running a single object search task.
    """
    def __init__(self, controller, task_config):
        """
        If task_type is "class", then target is an object type.
        If task_type is "object", then target is an object ID.

        detectables: a set of classes the robot can detect, or 'any'.
        """
        self.task_config = task_config
        task_type = task_config["task_type"]
        target = task_config["target"]
        self.scene = controller.scene.split("_")[0]
        self.target = target
        self.task_type = task_type
        self.goal_distance = task_config["nav_config"]["goal_distance"]
        self._detectables = self.task_config["detectables"]
        self._camera_intrinsic = pj.thor_camera_intrinsic(controller)
        if task_type not in {"class", "object"}:
            raise ValueError("Invalid target type: {}".format(task_type))
        super().__init__(controller)

    @property
    def target_id(self):
        return self.target

    @property
    def robot_id(self):
        return self.task_config.get("robot_id", "robot0")

    def get_info(self, fields):
        """
        Given a list of field names (e.g. grid_size, grid_map),
        return a dictionary mapping from a parameter name to value.
        """
        grid_size = tt.thor_grid_size_from_controller(self.controller)
        scene = tt.thor_scene_from_controller(self.controller)
        output = {}
        for item in fields:
            if item.lower() == "grid_size":
                output["grid_size"] = grid_size

            elif item.lower() == "grid_map":
                grid_maps_path = self.task_config["paths"]["grid_maps_path"]
                gmap_path = os.path.join(grid_maps_path, "{}-{}.json".format(scene, grid_size))
                if os.path.exists(gmap_path):
                    print("Loading GridMap from {}".format(gmap_path))
                    grid_map = tt.GridMap.load(gmap_path)
                else:
                    print("Converting scene to GridMap...")
                    grid_map = tt.proper_convert_scene_to_grid_map(
                        self.controller, grid_size)
                    if self.task_config["save_grid_map"]:
                        print("Saving grid map to from {}".format(gmap_path))
                        os.makedirs(grid_maps_path, exist_ok=True)
                        grid_map.save(gmap_path)
                output["grid_map"] = grid_map

            elif item.lower() == "agent_pose":
                agent_pose = tt.thor_agent_pose(self.controller, as_tuple=True)
                output["thor_agent_pose"] = agent_pose

            elif item.lower() == "camera_pose":
                camera_pose = tt.thor_camera_pose(self.controller, as_tuple=True)
                output["thor_camera_pose"] = camera_pose

            elif item.lower() == "groundtruth_prior":
                output["thor_prior"] = {self.get_object_loc(self.target) : 1e6}

            else:
                raise ValueError("Invalid field item for getting information: {}".format(item))
        return output

    def compute_results(self):
        """
        We will compute:
        1. Discounted cumulative reward
           Will save the entire trace of history.

        2. SPL. Even though our problem involves open/close,
           the optimal path should be just the navigation length,
           because the agent just needs to navigate to the container, open it
           and then perhaps look down, which doesn't change the path length.
           This metric alone won't tell the full story. Because it obscures the
           cost of actions that don't change the robot's location. So a combination with
           discounted reward is better.

           Because SPL is a metric over all trials, we will return the
           result for individual trials, namely, the path length, shortest path length,
           and success

        Note: it appears that the shortest path from ai2thor isn't snapped to grid,
        or it skips many steps. That makes it slightly shorter than the path found by
        our optimal agent. But will still use it per
        """
        # Uses the ThorObjectSearchOptimalagent to compute the path as shortest path.
        plan, poses = ThorObjectSearchOptimalAgent.plan(
            self.controller, self.init_state.agent_pose,
            self.target, self.task_type,
            **self.task_config["nav_config"])
        # Need to prepend starting pose. Get position only
        shortest_path = [tt.thor_pose_as_dict(self.init_state.agent_pose[0])]\
                        + [p[0] for p in poses]

        actual_path = self.get_current_path()
        last_reward = self._history[-1]['reward']
        success = last_reward == constants.TOS_REWARD_HI
        return [PathResult(self.scene, self.target, shortest_path, actual_path, success),
                HistoryResult(self._history, self.task_config["discount_factor"])]

    def get_current_path(self):
        """Returns a list of dict(x=,y=,z=) positions,
        using the history up to now, for computing results"""
        path = []
        for tup in self._history:
            state = tup['state']
            x, y, z = state.agent_pose[0]
            agent_position = dict(x=x, y=y, z=z)
            path.append(agent_position)
        return path

    def get_observation(self, event, action, detector):
        """
        vision_detector (cosp.vision.Detector or None): vision detector;
            If None, then groundtruth detection will be used

        Returns:
            TOS_Observation:
                img: RGB image
                img_depth: Depth image
                detections: list of (xyxy, conf, cls, pos) tuples. `cls` is
                    the detected class, `conf` is confidence, `xyxy` is bounding box,
                    'pos' is the 3D position of the detected object.
        """
        print("Getting observation")
        img = tt.vision.thor_img(event)
        img_depth = tt.vision.thor_img_depth(event)

        if isinstance(detector, GroundtruthDetector):
            detections = detector.detect_project(
                event, self._camera_intrinsic, single_loc=False)

        else:
            detections = detector.detect(img)

        # if vision_detector is None:
        #     # use groundtruth detection
        # else:
        #     import pdb; pdb.set_trace()
        #     detections = vision_detector.detect(img)
        #     for i in range(len(detections)):
        #         xyxy = detections[i][0]
        #         # TODO: COMPLETE
        #         # pos = projection.inverse_perspective(np.mean(xyxy), ..) # TODO
        return TOS_Observation(img,
                               img_depth,
                               detections,
                               tt.thor_agent_pose(event),
                               tt.thor_camera_horizon(event),
                               self.done(action))

    def get_object_loc(self, object_class):
        """Returns object location (note: in thor coordinates) for given
        object class, for the closest instance to the robot."""
        return tt.thor_closest_object_of_type_position(self.controller.last_event, object_class,
                                                       as_tuple=True)

    def get_state(self, event=None):
        # stores agent pose as tuple, for convenience.
        if event is None:
            event = self.controller.step(action="Pass")
        agent_pose = tt.thor_agent_pose(event, as_tuple=True)
        horizon = tt.thor_camera_horizon(event)
        objlocs = {}
        for cls in self._detectables:
            objlocs[cls] = self.get_object_loc(cls)
        return TOS_State(agent_pose, horizon, objlocs)

    def get_reward(self, state, action, next_state):
        """We will use a sparse reward."""
        if self.done(action):
            if self.success(action,
                            agent_pose=state.agent_pose,
                            horizon=state.horizon)[0]:
                return constants.TOS_REWARD_HI
            else:
                return constants.TOS_REWARD_LO
        else:
            return constants.TOS_REWARD_STEP

    def done(self, action):
        """Returns true if  the task is over. The object search task is over when the
        agent took the 'Done' action.
        """
        return action.name.lower() == "done"

    def success(self, action, agent_pose=None, horizon=None):
        """Returns (bool, str). For the bool:

        it is  True if the task is a success.
        The task is success if the agent takes 'Done' and
        (1) the target object is within the field of view.
        (2) the robot is close enough to the target.
        Note: uses self.controller to retrieve target object position.

        For the str, it is a message that explains failure,
        or 'Task success'"""

        if action.name.lower() != "done":
            return False, "action is not done"

        event = self.controller.step(action="Pass")
        backup_state = self.get_state(event)

        if agent_pose is not None:
            # Teleport to the given agent pose then evaluate
            position, rotation = agent_pose
            self.controller.step(action="Teleport",
                                 position=position,
                                 rotation=rotation,
                                 horizon=horizon)

        # Check if the target object is within the field of view
        event = self.controller.step(action="Pass")
        if self.task_type == "class":
            object_type = self.target
            in_fov = tt.thor_object_of_type_in_fov(event, object_type)
            p = tt.thor_closest_object_of_type(event, object_type)["position"]
            objpos = (p['x'], p['z'])
        else:
            raise NotImplementedError("We do not consider this case for now.")

        agent_position = tt.thor_agent_pose(event)[0]
        object_distance = euclidean_dist(objpos, (agent_position['x'],
                                                  agent_position['z']))
        close_enough = (object_distance <= self.goal_distance)
        success = in_fov and close_enough

        # Teleport back, if necessary (i.e. if agent_pose is provided)
        if agent_pose is not None:
            position, rotation = backup_state.agent_pose
            horizon = backup_state.horizon
            self.controller.step(action="Teleport",
                                 position=position,
                                 rotation=rotation,
                                 horizon=horizon)
        msg = "Task success"
        if not success:
            if not in_fov:
                msg = "Object not in field of view!"
            if not close_enough:
                msg = "Object not close enough! Minimum distance: {}; Actual distance: {}".\
                    format(self.goal_distance, object_distance)
        return success, msg

    def update_history(self, next_state, action, observation, reward):
        # Instead of just storing everything which takes a lot of space,
        # store the important stuff; The images in the observation don't
        # need to be saved because the image can be determined in ai2thor
        # through the actions the agent has taken. But we do want to store
        # detections in the observation.
        info = dict(state=next_state,    # the state where observation is generated
                    action=action,  # the action that lead to the state
                    detections=observation.detections if observation is not None else None,
                    observed_robot_pose=observation.robot_pose if observation is not None else None,
                    reward=reward)
        self._history.append(info)

    def get_step_info(self, step):
        info = self._history[step]
        s = info['state']
        a = info['action']
        o = set()
        for detection in info['detections']:
            cls = detection[2]
            o.add(cls)
        clses = list(sorted(o))
        o_str = ",".join(clses)
        r = info['reward']

        pose_str = "(x={:.3f}, z={:.3f}, pitch={:.3f}, yaw={:.3f}"\
            .format(s.agent_pose[0][0], s.agent_pose[0][2], s.agent_pose[1][0], s.agent_pose[1][1])

        action = a.name if not a.name.startswith("Open")\
            else "{}({})".format(a.name, a.params)
        return "Step {}: Action: {}; Reward: {}; Observation: {}; {}"\
            .format(step, action, r, o_str, pose_str)

    def visualizer(self, **config):
        return ThorObjectSearchViz2D(**config)

# Class naming aliases
ThorObjectSearch = TOS
