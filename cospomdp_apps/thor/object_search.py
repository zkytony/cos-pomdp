import os
import random
import math
import numpy as np
from collections import namedtuple
from pprint import pprint
import cv2

import pomdp_py
from pomdp_py.utils import typ
import ai2thor
import ai2thor.util.metrics as metrics

import thortils as tt
from thortils.utils import euclidean_dist
import thortils.vision.projection as pj
from thortils.vision.general import saveimg
from thortils.vision import thor_img, thor_img_depth, thor_topdown_img
from thortils.utils.visual import GridMapVisualizer

from .result_types import PathResult, HistoryResult
from .common import ThorEnv, TOS_Action, TOS_State, TOS_Observation, ThorAgent
from .agent import ThorObjectSearchOptimalAgent
from .visual import ThorObjectSearchViz2D
from .detector import YOLODetector, GroundtruthDetector
from . import paths
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
                grid_maps_path = paths.GRID_MAPS_PATH
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
                print("::::::::: WARNING:: Using agent_pose instead of camera_pose. Are you sure? :::::::")
                agent_pose = tt.thor_agent_pose(self.controller)
                output["thor_agent_pose"] = agent_pose

            elif item.lower() == "camera_pose":
                camera_pose = tt.thor_camera_pose(self.controller)
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
        actual_path = self.get_current_path()
        rewards = self.get_reward_sequence()
        try:
            plan, poses = ThorObjectSearchOptimalAgent.plan(
                self.controller, self.init_state.agent_pose,
                self.target, self.task_type,
                **self.task_config["nav_config"])
        except ValueError:
            # Plan not found; this trial is not "completable"; Stil save history,
            # for later replay.
            return [PathResult(self.scene, self.target, None, actual_path, None,
                               rewards, self.task_config["discount_factor"]),
                    HistoryResult(self._history, self.task_config["discount_factor"])]

        # Need to prepend starting pose. Get position only
        shortest_path = [tt.thor_pose_as_dict(self.init_state.agent_pose[0])]\
                         + [p[0] for p in poses]
        last_reward = self._history[-1]['reward']
        success = last_reward == constants.TOS_REWARD_HI
        return [PathResult(self.scene, self.target, shortest_path, actual_path, success,
                           rewards, self.task_config["discount_factor"]),
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

    def get_reward_sequence(self):
        rewards = []
        for tup in self._history:
            rewards.append(tup['reward'])
        return rewards

    def get_observation(self, event, action, detector, record_detections=True):
        """
        vision_detector (cosp.vision.Detector or None): vision detector;
            If None, then groundtruth detection will be used

        Returns:
            TOS_Observation:
                img: RGB image
                img_depth: Depth image
                detections: list of (xyxy, conf, cls, thor_positions) tuples. `cls` is
                    the detected class, `conf` is confidence, `xyxy` is bounding box,
                    'pos' is the 3D position of the detected object.
        """
        print("Getting observation")

        img = tt.vision.thor_img(event)
        img_depth = tt.vision.thor_img_depth(event)

        if detector is None:
            detections = []
        else:
            if isinstance(detector, GroundtruthDetector):
                detections = detector.detect_project(
                    event, self._camera_intrinsic, single_loc=False)
            else:
                camera_pose = tt.thor_camera_pose(event, as_tuple=True)
                detections = detector.detect_project(img, event.depth_frame,
                                                     self._camera_intrinsic,
                                                     camera_pose)
            # logging the detections
            if record_detections:
                camera_position = tt.thor_camera_position(event, as_tuple=True)
                detector.record_detections(detections, camera_position, exclude={self.target})

        return TOS_Observation(img,
                               img_depth,
                               detections,
                               tt.thor_camera_pose(event),
                               tt.thor_camera_horizon(event),
                               self.done(action))

    def get_object_loc(self, object_class):
        """Returns object location (note: in thor coordinates) for given
        object class, for the closest instance to the robot."""
        return tt.thor_closest_object_of_type_position(
            self.controller.last_event, object_class, as_tuple=True)

    def get_state(self, event=None):
        # stores agent pose as tuple, for convenience.
        if event is None:
            event = self.controller.step(action="Pass")
        agent_pose = tt.thor_agent_pose(event, as_tuple=True)
        horizon = tt.thor_camera_horizon(event)
        objlocs = {}
        if self._detectables != "any":
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
                    observation=observation,  # detections=observation.detections_without_locations if observation is not None else None,
                    action=action,
                    reward=reward)
        self._history.append(info)

    def get_step_info(self, step):
        info = self._history[step]
        s = info['state']
        a = info['action']
        o = set()
        for detection in info['observation']['detections']:
            cls = detection[2]
            o.add(cls)
        clses = list(sorted(o))
        o_str = ",".join(clses)
        r = info['reward']

        pose_str = "(x={:.3f}, z={:.3f}, pitch={:.3f}, yaw={:.3f}"\
            .format(s.agent_pose[0][0], s.agent_pose[0][2], s.agent_pose[1][0], s.agent_pose[1][1])

        if type(a) == dict:
            a = a["base"]

        action = a.name if not a.name.startswith("Open")\
            else "{}({})".format(a.name, a.params)
        return "Step {}: Action: {}; Reward: {}; Observation: {}; {}"\
            .format(step, action, r, o_str, pose_str)

    def visualizer(self, **config):
        return ThorObjectSearchViz2D(**config)

    def saver(self, save_path, agent, **kwargs):
        return ThorObjectSearchTrialSaver(self, agent, save_path, **kwargs)

# Class naming aliases
ThorObjectSearch = TOS

import os
class ThorObjectSearchTrialSaver:

    def __init__(self, task_env, agent, savedir, **kwargs):
        self.task_env = task_env
        self.agent = agent

        self.beliefsdir = os.path.join(savedir, "beliefs")
        self.fpdir = os.path.join(savedir, "first-person")
        self.tddir = os.path.join(savedir, "topdown")
        self.savedir = savedir
        self._generate_gif = kwargs.get("gif", False)
        self._frame_duration = kwargs.get("duration", 0.2)

        print(f"Will save the trial visualizations to {savedir}")
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(self.beliefsdir, exist_ok=True)
        os.makedirs(self.fpdir, exist_ok=True)
        os.makedirs(self.tddir, exist_ok=True)
        self._log = {"poses":[],
                     "object_detections": {}}

    def save_step(self, step, img, action, observation):
        # record pose and detections - the pose is already in the controller
        controller = self.task_env.controller
        agent_pose = self.task_env.get_state(controller.last_event).agent_pose
        agent_pose2d = (agent_pose[0][0], agent_pose[0][2], agent_pose[1][1])
        self._log["poses"].append(agent_pose2d)

        if observation is not None:
            for detection in observation.detections:
                xyxy, conf, cls, thor_locs = detection
                if cls not in self._log["object_detections"]:
                    self._log["object_detections"][cls] = []

                self._log["object_detections"][cls].append(
                    ((conf, self.agent.detector._avg_loc(thor_locs))))

        # First, save the img visulized by the visualizer
        belief_path = os.path.join(self.beliefsdir, f"belief_{step:0>3}.png")
        saveimg(img, belief_path)
        print(f"Saved beliefs visualization for step {step}")

        # Then, save the img from Ai2thor (both first person view and top-down view);
        # If it is the first step, then directly save the FPV from thor controller;
        # Otherwise, get the object detection visualization from the detector
        if step == 0:
            assert action is None and observation is None
            fp_img = thor_img(controller, cv2=False)
            td_img = thor_topdown_img(controller, cv2=False)

        else:
            # Get the object detection visualization
            fp_img = self.agent.detector.plot_detections(
                observation.img, observation.detections)
            fp_img = cv2.cvtColor(fp_img, cv2.COLOR_BGR2RGB)
            td_img = thor_topdown_img(controller, cv2=False)

        fp_path = os.path.join(self.fpdir, f"fpv_{step:0>3}.png")
        saveimg(fp_img, fp_path)
        print(f"Saved first person view image for step {step}")

        td_path = os.path.join(self.tddir, f"td_{step:0>3}.png")
        saveimg(td_img, td_path)
        print(f"Saved top-down view image for step {step}")


    def finish(self):
        import imageio

        self.plot_trajectory(self._log["poses"],
                             self._log["object_detections"])

        # generate gif if wanted
        if self._generate_gif:
            # belief gif
            print("Generating GIF for beliefs")
            belief_images = []
            for filename in sorted(os.listdir(self.beliefsdir)):
                if filename.endswith("png"):
                    file_path = os.path.join(self.beliefsdir, filename)
                    belief_images.append(imageio.imread(file_path))
            imageio.mimsave(os.path.join(self.savedir, "beliefs.gif"), belief_images, duration=self._frame_duration)

            print("Generating GIF for First-Person Views")
            fp_images = []
            for filename in sorted(os.listdir(self.fpdir)):
                if filename.endswith("png"):
                    file_path = os.path.join(self.fpdir, filename)
                    fp_images.append(imageio.imread(file_path))
            imageio.mimsave(os.path.join(self.savedir, "fps.gif"), fp_images, duration=self._frame_duration)

            print("Generating GIF for Top-Down views")
            td_images = []
            for filename in sorted(os.listdir(self.tddir)):
                if filename.endswith("png"):
                    file_path = os.path.join(self.tddir, filename)
                    td_images.append(imageio.imread(file_path))
            imageio.mimsave(os.path.join(self.savedir, "tds.gif"), td_images, duration=self._frame_duration)

    def plot_trajectory(self, thor_poses, thor_detected_object_locs):
        """
        Will visualize the topdown view along with the current
        trajectory as well as detected object locations. The trajectory
        will be visualized as connected nodes where each node is essentially
        a viewpoint pose
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        fig, ax = plt.subplots()
        ax.set_axis_off()
        robot_radius = self.agent.grid_map.grid_size / 3
        prev_center = None
        for thor_x, thor_z, thor_yaw in thor_poses:
            # Plot a circle as the robot and then a tick.
            circle = plt.Circle((thor_x, thor_z), robot_radius, color="lightgray",
                                edgecolor='black', linewidth=10)
            ax.add_patch(circle)
            center = (thor_x, thor_z)
            heading = (thor_x + robot_radius*math.sin(thor_yaw),
                       thor_z + robot_radius*math.cos(thor_yaw))
            line = mlines.Line2D([center[0], heading[0]],
                                 [center[1], heading[1]],
                                 color="black")
            ax.add_line(line)
            if prev_center is not None:
                line = mlines.Line2D([center[0], prev_center[0]],
                                     [center[1], prev_center[1]])
                ax.add_line(line)
            prev_center = center

        # Plot the object detections
        size = self.agent.grid_map.grid_size / 3
        for cls in thor_detected_object_locs:
            for conf, loc3d in thor_detected_object_locs[cls]:
                x, _, z = loc3d
                rect = plt.Rectangle((x - size/2,  z - size/2),
                                     size, size, facecolor="green")
                ax.add_patch(rect)

        ax.set_xlim(*np.asarray(self.agent.grid_map.ranges_in_thor[0]) * self.agent.grid_map.grid_size)
        ax.set_ylim(*np.asarray(self.agent.grid_map.ranges_in_thor[1]) * self.agent.grid_map.grid_size)
        ax.set_aspect('equal')

        plt.savefig(os.path.join(self.savedir, "trajectory.png"), transparent=True, dpi=300)
