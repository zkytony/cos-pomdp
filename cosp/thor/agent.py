import random
from thortils import (thor_closest_object_of_type,
                      thor_agent_pose,
                      thor_object_with_id,
                      thor_object_receptors)

from .utils import plt, as_tuple, as_dict
from .common import TOS_Action, ThorAgent

class ThorObjectSearchAgent(ThorAgent):
    AGENT_USES_CONTROLLER = False

######################### Optimal Agent ##################################
import time
from thortils.navigation import (get_shortest_path_to_object,
                                 get_shortest_path_to_object_type)
class ThorObjectSearchOptimalAgent(ThorObjectSearchAgent):
    """
    The optimal agent uses ai2thor's shortest path method
    to retrieve a path, and then follows this path by taking
    appropriate actions.
    """

    # Because this agent is not realistic, we permit it to have
    # access to the controller.
    AGENT_USES_CONTROLLER = True

    def __init__(self,
                 controller,
                 task_config):
        """Builds the agent and computes a plan"""
        super().__init__()
        self.controller = controller
        self.target = task_config["target"]
        self.task_type = task_config["task_type"]
        self.movement_params = task_config["nav_config"]["movement_params"]

        start_pose = thor_agent_pose(self.controller, as_tuple=True)
        overall_plan, overall_poses = ThorObjectSearchOptimalAgent.plan(
            self.controller, start_pose, self.target,
            self.task_type, **task_config["nav_config"])

        self.plan = overall_plan
        self._poses = overall_poses
        self._index = 0

    def act(self):
        """Returns action in plan"""
        if self._index < len(self.plan):
            action_name, action_params = self.plan[self._index]
            if action_name not in self.movement_params:
                action = TOS_Action(action_name, action_params)
            else:
                # TODO: should use the params in the action tuple.
                action = TOS_Action(action_name, self.movement_params[action_name])
            self._index += 1
        else:
            action = TOS_Action("Done", {})
        return action

    def update(self, action, observation):
        pass

    @classmethod
    def plan(cls, controller, start_pose, target, task_type, **nav_config):
        """
        Plans navigation + container opening actions
        where the navigation actions are movements along the shortest path
        found by A* on the grid map of environment.

        Args:
            controller (ai2thor.Controller)
            start_pose (tuple): position (x,y,z), rotation (pitch, yaw, roll).
                Both are tuples.
            target (str): Object type or ID
            task_type (str): "class" or "object"
            nav_config (dict): Navigation configurations
                (e.g. goal_distance, v_angles, h_angles, etc.), used in path finding.
                Refer to get_shortest_path_to_object for complete configuration.
        Returns:
            plan, poses

            plan: List of (action_name, params) tuples representing actions.
                The params are specific to the action; For movements, it's
                (forward, pitch, yaw). For opening, it is an object id.
            poses: List of (dict(x=,y=,z=), dict(x=,y=,z=)) (position, rotation) tuples.
                where each pose at index i corresponds to the pose resulting from taking
                action i.
        """
        start_position, start_rotation = start_pose
        if task_type == "class":
            object_type = target
            target_object = thor_closest_object_of_type(controller, object_type)
        else:
            target_object_id = target
            target_object = thor_object_with_id(controller, target_object_id)

        # Check if the target object is within openable receptacle. If so,
        # need to navigate to the receptacle, open it.
        # so receptors are also targets we want to navigate to
        ### NOTE: This method is hacky. It works for some objects, but has the following issues:
        ### 1. The robot doesn't know if it can successfully open the container
        ###    door (especially fridge).
        ### 2. An object, if placed very low, may be blocked by e.g. tabletop
        ###    and right now the robot cannot adjust its height to detect those
        ###    objects
        ### 3. (This is unrelated to this method, but it's an issue) For some
        ###    objects, their height seems wrong with respect to robot's height,
        ###    which causes the wrong pitch rotation to be computed
        ### THEREFORE, this optimal agent isn't complete; It returns a plan,
        ### which is approximately optimal (shortest), but it won't be able to
        ### execute it successfully 100% of the time because of the above issues.
        openable_receptors = thor_object_receptors(controller,
                                                   target_object["objectId"],
                                                   openable_only=True)
        openable_receptor_ids = set(r["objectId"] for r in openable_receptors if r["openable"])

        goal_objects = openable_receptors + [target_object]
        overall_poses, overall_plan = [], []
        position, rotation = start_position, start_rotation
        for obj in goal_objects:
            _start_time = time.time()
            _nav_config = dict(nav_config)
            poses, plan = get_shortest_path_to_object(
                controller, obj["objectId"],
                position, rotation,
                return_plan=True,
                **nav_config
            )
            if plan is not None:
                print("plan found in {:.3f}s".format(time.time() - _start_time))
            else:
                raise ValueError("Plan not found to {}".format(obj["objectId"]))

            if len(poses) == 0:
                # No action needed. The robot is at where it should be already.
                assert len(plan) == 0
                pose_to_extend = as_dict((position, rotation))
            else:
                pose_to_extend = poses[-1]
                # in case navigation is needed between containers (should be rare),
                # update the position and rotation for the next search.
                position, rotation = as_tuple(poses[-1])

            # Check if we need an open action
            if obj["objectId"] in openable_receptor_ids:
                open_action = ("OpenObject", dict(objectId=obj["objectId"]))
                plan.append(open_action)
                poses.append(pose_to_extend)  # just so that both lists have same length

            overall_poses.extend(poses)
            overall_plan.extend(plan)


        if plan is None:
            raise ValueError("Plan to {} not found".format(target))

        return overall_plan, overall_poses
#############################################################################



########################### COS-POMDP agent #################################
import pomdp_py
from ..framework import POMDP, Decision
from ..models import cospomdp

class HighLevelStatus:
    MOVING = "moving"
    SEARCHING = "searching"
    INITIAL = "initial"
    DONE = "done"

class MoveDecision(Decision):
    def __init__(self, dest):
        super().__init__("move")
        self.dest = dest

class SearchDecision(Decision):
    def __init__(self):
        super().__init__("search")

class DoneDecision(Decision):
    def __init__(self):
        super().__init__("done")

class HighLevelRobotState(pomdp_py.ObjectState):
    def __init__(self, pos, status):
        super().__init__("robot", dict(pos=pos, status=status))

class HighLevelObjectState(pomdp_py.ObjectState):
    """The `pos` means the location that the robot should be in
    if it wants to detect the object."""
    def __init__(self, object_class, pos):
        super().__init__(object_class, dict(pos=pos, status=status))

class HighLevelSearchRegion(cospomdp.SearchRegion):
    """This is where the high-level belief will be defined over.
    Instead of meaning 'the location of the target', it means
    'the location that the robot should be in if it wants to
    detect the target'."""
    def __init__(self, reachable_positions):
        """reachable_positions: (initial) set of 2D locations the robot can reach."""
        self.reachable_positions = set(reachable_positions)
        self._current = 0

    def __next__(self):
        if self._current < len(self.reachable_positions):
            retval = self.reachable_positions[self._current]
            self._current += 1
            return retval
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __contains__(self, item):
        return item in self.reachable_positions


class HighLevelTransitionModel(pomdp_py.TransitionModel):
    """Transition model for high-level planner.

    If the decision is moving to another position, then simply change the robot
    pose to that position (deterministic). If the decision is search, then
    update robot state status to be searching locally. If done, then done.
    """
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id

    def sample(self, state, action):
        """
        Args:
            state (cospomdp.ReducedState)
            action (Decision)
        """
        next_robot_pos = state.robot_state["pos"]
        next_robot_status = state.robot_state["status"]
        if isinstance(action, MoveDecision):
            next_robot_pos = action.dest
            next_robot_status = Status.MOVING
        elif isinstance(action, SearchDecision):
            next_robot_status = Status.SEARCHING
        elif isinstance(action, DoneDecision):
            next_robot_status = Status.DONE
        next_robot_state = HighLevelRobotState(next_robot_pos,
                                               next_robot_status)
        target_state = HighLevelObjectState(state.target_state.objclass,
                                            state.target_state["pos"])
        return cospomdp.ReducedState(
            self.robot_id, self.target_id,
            next_robot_state, target_state)


class HighLevelDetectionModel(cospomdp.DetectionModel):
    """High level detection model."""
    def __init__(self, detecting_class, true_pos, rand=random):
        """
        Detector for detecting class.
        `true_pos` is the true positive rate of detection.
        """
        self.detecting_class = detecting_class
        self._true_pos = true_pos
        self._rand = rand

    def probability(self, object_observation, object_state, robot_state):
        if object_observation.objclass != self.detecting_class\
           or object_state.objclass != self.detecting_class:
            return 0.0

        if object_observation.location is None:
            # negative; didn't observe the object
            if object_state["pos"] == robot_state["pos"]:
                # robot expects to see the object, but did. False negative
                return 1.0 - self._true_pos
            else:
                # robot isn't expecting to see the object and it didn't.
                # because we are not considering false positives so it's 100%
                return 1.0
        else:
            # positive, observed the object
            if object_state["pos"] == robot_state["pos"]:
                # robot expects to see the object, and did. True positive
                return self._true_pos
            else:
                # robot isn't expecting to see the object and it did.
                # because we are not considering false positives so it's 0%
                return 0.0

    def sample(self, object_state, robot_state):
        if object_state["pos"] == robot_state["pos"]:
            # We expect the robot to be able to detect the object,
            # subject to class-specific uncertainty
            if self._true_pos >= self._rand.uniform(0, 1):
                # True positive
                return cospomdp.ObjectObservation(object_state.objclass,
                                                  object_state["pos"])
            else:
                # False negative
                return cospomdp.ObjectObservation(object_state.objclass, None)
        else:
            # the robot is not at a location where the object
            # can be observed (of course, there could be multiple
            # locations where the robot can be to observe the target,
            # but because we will have a belief over locations to observe
            # target, this should not be a problem)
            #
            # we do not model false positives in the POMDP model at the high
            # level. It is subject to too many factors unmodelable.
            return cospomdp.ObjectObservation(object_state.objclass, None)


class HighLevelObservationModel(pomdp_py.ObservationModel):
    def __init__(self, detectable_classes):
        self.detectable_classes = detectable_objects
        self.object_observation_models =

    def sample(self, next_state, action):
        if next_state.robot_state["pos"] == next_state.target_state["pos"]:
            # Robot is at where it should be to detect the target.



class COSPOMDP(POMDP):
    """The COSPOMDP for Thor Object Search;
    Remember that it is a high-level planning framework.

    State:
        robot state: 2D reachable position
        target state: 2D reachable position; Instead of meaning
            "the location of the target", means "the location the
             robot should be in if it wants to detect the target"
    Action (Decision):
        move_to: moves the robot to another reachable position
                 (by default, nearby)
        search: decides to search within a local region (i.e. where the robot is)
        done: declares victory.
    """
    def __init__(self, target_class, search_region):
        pass
