# Generic class for experiment trial in thor
import sys
from sciex import Trial, Event
from ai2thor.controller import Controller
import thortils
import time
from pprint import pprint

from cospomdp.utils.misc import _debug
from cospomdp.utils import cfg
cfg.DEBUG_LEVEL = 0

from . import constants
from .object_search import ThorObjectSearch
from .agent import (ThorObjectSearchOptimalAgent,
                    ThorObjectSearchBasicCosAgent,
                    ThorObjectSearchExternalAgent,
                    ThorObjectSearchCompleteCosAgent,
                    ThorObjectSearchRandomAgent,
                    ThorObjectSearchGreedyNbvAgent)
from .replay import ReplaySolver
from .result_types import PathResult, HistoryResult
from .common import make_config, TaskArgs, TOS_Action, ThorAgent

class ThorTrial(Trial):

    # TODO: shared setup between trials

    RESULT_TYPES = []

    def __init__(self, name, config, verbose=False):
        if self.__class__ == ThorTrial:
            raise ValueError("ThorTrial is generic. Please create the Trial object"\
                             "for your specific task!")
        super().__init__(name, config, verbose=verbose)

    def _start_controller(self):
        controller = thortils.launch_controller(self.config["thor"])
        return controller

    def print_config(self):
        print("--- Task config ({})---".format(self.config["task_env"]))
        pprint(self.config["task_config"], width=75)
        print("--- Agent Config ({}) ---".format(self.config["agent_class"]))
        pprint(self.config["agent_config"], width=75)

    def could_provide_resource(self):
        """If using vision detector"""
        return self.config["task_config"]["detector_config"]["use_vision_detector"]

    def provide_shared_resource(self):
        """Load vision detector"""
        # Assuming it could provide resource
        assert self.could_provide_resource(), f"This trial {self.name} cannot provide shared resource."
        return ThorAgent.load_detector(self.config["task_config"])

    def setup(self):
        # If shared resource (i.e. detector is provided, use it)
        if self.shared_resource is not None:
            vision_detector = self.shared_resource
            self.config["task_config"]["detector_config"]["vision_detector"] = vision_detector

        controller = self._start_controller()
        task_env = eval(self.config["task_env"])(controller, self.config["task_config"])
        agent_class = eval(self.config["agent_class"])
        agent_init_inputs = task_env.get_info(self.config["agent_init_inputs"])
        if agent_class.AGENT_USES_CONTROLLER:
            agent = agent_class(controller,
                                self.config['task_config'],
                                **self.config['agent_config'],
                                **agent_init_inputs)
        else:
            agent = agent_class(self.config["task_config"],
                                **self.config['agent_config'],
                                **agent_init_inputs)

        # what to return
        result = dict(controller=controller,
                      task_env=task_env,
                      agent=agent)

        if self.config.get("visualize", False):
            viz = task_env.visualizer(**self.config["viz_config"])
            img = viz.visualize(task_env, agent, step=0)
            result['viz'] = viz

            if "save_path" in self.config:
                saver = task_env.saver(self.config["save_path"], agent)
                result['saver'] = saver
                saver.save_step(0, img, None, None)

        return result

    def run(self,
            logging=False,
            step_act_cb=None,
            step_act_args={},
            step_update_cb=None,
            step_update_args={}):
        """
        Functions intended for debugging purposes:
            step_act_cb: Called after the agent has determined its action
            step_update_cb: Called after the agent has executed the action and updated
                given environment observation.
        """
        # self.config["visualize"] = True
        # self.config["task_config"]["detector_config"]["plot_detections"] = True

        self.print_config()
        components = self.setup()
        agent = components['agent']
        task_env = components['task_env']
        controller = components['controller']
        viz = components.get("viz", None)
        saver = components.get("saver", None)

        _actions = []

        max_steps = self.config["max_steps"]
        for i in range(1, max_steps+1):
            # End with a Done
            if i == max_steps:
                return TOS_Action("done", {})

            action = agent.act()
            if not logging:
                a_str = action.name if not action.name.startswith("Open")\
                    else "{}({})".format(action.name, action.params)
                sys.stdout.write(f"Step {i} | Action: {a_str}; ")
                sys.stdout.flush()
            if step_act_cb is not None:
                step_act_cb(task_env, agent, viz=viz, step=i, **step_act_args)

            if cfg.DEBUG_LEVEL > 0:
                _actions.append(action)
                if _rotating_too_much(_actions):
                    import pdb; pdb.set_trace()

            observation, reward = task_env.execute(agent, action)
            agent.update(action, observation)

            if logging:
                _step_info = task_env.get_step_info(step=i)
                self.log_event(Event("Trial %s | %s" % (self.name, _step_info)))
            else:
                sys.stdout.write("Action: {}, Observation: {}; Reward: {}\n"\
                                 .format(action, observation, reward))
                sys.stdout.flush()

            if self.config.get("visualize", False):
                img = viz.visualize(task_env, agent, step=i)

                if saver is not None:
                    saver.save_step(i, img, action, observation)

            if step_update_cb is not None:
                step_update_cb(task_env, agent, viz=viz, **step_update_args)

            if task_env.done(action):
                success, msg = task_env.success(action)
                if logging:
                    self.log_event(Event("Trial %s | %s" % (self.name, msg)))
                else:
                    print(msg)
                break
        results = task_env.compute_results()
        controller.stop()
        if self.config.get("visualize", False):
            viz.on_cleanup()
        return results

    @property
    def scene(self):
        return self.config["thor"]["scene"]


# ------------- Object search trial ------------- #
class ThorObjectSearchTrial(ThorTrial):
    RESULT_TYPES = [PathResult, HistoryResult]


def _rotating_too_much(actions):
    # count the number of times, from the last one,
    # that the actions is rotating between
    # rotations
    count = 0
    for a in reversed(actions):
        if a.name.startswith("Rotate"):
            count += 1
            if count >= 8:
                return True
        else:
            break
    return False
