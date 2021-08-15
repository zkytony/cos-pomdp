import random
import math
import numpy as np
from collections import namedtuple
from pprint import pprint

import pomdp_py
from pomdp_py.utils import typ
import ai2thor
import ai2thor.util.metrics as metrics

from thortils import (thor_agent_pose,
                      thor_camera_horizon,
                      thor_object_type,
                      thor_object_of_type_in_fov,
                      thor_object_position,
                      thor_closest_object_of_type,
                      thor_pose_as_dict)

from thortils.vision import thor_img, thor_img_depth, thor_object_bboxes
from thortils.utils import (to_degrees, closest,
                            normalize_angles, euclidean_dist)

from ..vision.utils import projection
from .result_types import PathResult, HistoryResult
from .common import ThorEnv, TOS_Action, TOS_State, TOS_Observation
from .agent import ThorObjectSearchOptimalAgent
from .visual import ThorObjectSearchViz2D
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
        """
        task_type = task_config["task_type"]
        target = task_config["target"]
        if task_type not in {"class", "object"}:
            raise ValueError("Invalid target type: {}".format(task_type))
        super().__init__(controller)
        self.scene = controller.scene.split("_")[0]
        self.target = target
        self.task_type = task_type
        self.goal_distance = task_config["nav_config"]["goal_distance"]
        self.task_config = task_config

    @property
    def target_id(self):
        return self.target

    @property
    def robot_id(self):
        return self.task_config.get("robot_id", "robot0")

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
        shortest_path = [thor_pose_as_dict(self.init_state.agent_pose[0])]\
                        + [p[0] for p in poses]

        actual_path = self.get_current_path()
        last_reward = self._history[-1][-1]
        success = last_reward == constants.TOS_REWARD_HI
        return [PathResult(self.scene, self.target, shortest_path, actual_path, success),
                HistoryResult(self._history, self.task_config["discount_factor"])]

    def get_current_path(self):
        """Returns a list of dict(x=,y=,z=) positions,
        using the history up to now, for computing results"""
        path = []
        for tup in self._history:
            state = tup[0]
            x, y, z = state.agent_pose[0]
            agent_position = dict(x=x, y=y, z=z)
            path.append(agent_position)
        return path

    def get_observation(self, event=None, detector=None):
        """
        detector (cosp.vision.Detector or None): vision detector;
            If None, then groundtruth detection will be used

        Returns:
            TOS_Observation:
                img: RGB image
                img_depth: Depth image
                detections: list of (xyxy, conf, cls, pos) tuples. `cls` is
                    the detected class, `conf` is confidence, `xyxy` is bounding box,
                    'pos' is the 3D position of the detected object.
        """
        if event is None:
            event = self.controller.step(action="Pass")
        img = thor_img(event)
        img_depth = thor_img_depth(event)
        if detector is None:
            # use groundtruth detection
            detections = []
            bboxes = thor_object_bboxes(event)  # xyxy
            for objectId in bboxes:
                loc3d = thor_object_position(event, objectId, as_tuple=True)
                if loc3d is None:
                    # This is due to objectId, though provided in
                    # bounding box, is not found in event metadata.
                    # Perhaps this is a bug in ai2thor
                    pass

                cls = thor_object_type(objectId)
                if cls not in self.task_config["detectables"]:
                    continue
                conf = 1.0
                xyxy = bboxes[objectId]
                detections.append((xyxy, conf, cls, loc3d))
        else:
            detections = detector.detect(img)
            for i in range(len(detections)):
                xyxy = detections[i][0]
                # TODO: COMPLETE
                # pos = projection.inverse_perspective(np.mean(xyxy), ..) # TODO
        return TOS_Observation(img, img_depth, detections, thor_agent_pose(event))

    def get_state(self, event=None):
        # stores agent pose as tuple, for convenience.
        if event is None:
            event = self.controller.step(action="Pass")
        agent_pose = thor_agent_pose(event, as_tuple=True)
        horizon = thor_camera_horizon(event)
        return TOS_State(agent_pose, horizon)

    def get_reward(self, state, action, next_state):
        """We will use a sparse reward."""
        if self.done(action):
            if self.success(action,
                            agent_pose=state.agent_pose,
                            horizon=state.horizon):
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
        """Returns True if the task is a success.
        The task is success if the agent takes 'Done' and
        (1) the target object is within the field of view.
        (2) the robot is close enough to the target.
        Note: uses self.controller to retrieve target object position."""
        if action.name != "Done":
            return False

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
            in_fov = thor_object_of_type_in_fov(event, object_type)
            p = thor_closest_object_of_type(event, object_type)["position"]
            objpos = (p['x'], p['y'], p['z'])
        else:
            raise NotImplementedError("We do not consider this case for now.")

        agent_position = thor_agent_pose(event, as_tuple=True)[0]
        object_distance = euclidean_dist(objpos, agent_position)
        # allows 0.1 to account for small difference due to floating point instability
        close_enough = (object_distance - self.goal_distance) <= 2e-1
        success = in_fov and close_enough

        # Teleport back, if necessary (i.e. if agent_pose is provided)
        if agent_pose is not None:
            position, rotation = backup_state.agent_pose
            horizon = backup_state.horizon
            self.controller.step(action="Teleport",
                                 position=position,
                                 rotation=rotation,
                                 horizon=horizon)
        if not success:
            if not in_fov:
                print("Object not in field of view!")
            if not close_enough:
                print("Object not close enough! Minimum distance: {}; Actual distance: {}".\
                      format(self.goal_distance, object_distance))
        return success

    def get_step_info(self, step):
        sp, a, o, r = self._history[step]
        x, z, pitch, yaw = sp.agent_pose[0][0], sp.agent_pose[0][2], sp.agent_pose[1][0], sp.agent_pose[1][1]
        action = a.name if not a.name.startswith("Open") else "{}({})".format(a.name, a.params)
        return "Step {}: Action: {}, (x={}, z={}, pitch={}, yaw={}), Reward: {}"\
            .format(step, typ.blue(action), x, z, pitch, yaw, r)

    def visualizer(self, **config):
        return ThorObjectSearchViz2D(**config)

# Class naming aliases
ThorObjectSearch = TOS
