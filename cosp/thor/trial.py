from sciex import Trial, Event
from ai2thor.controller import Controller

# Import everything related to thor and its agents
from .task import *
from .agents import *
from .actions import ThorAction
from .result_types import PathsResult, HistoryResult
import thortils.constants as defaults
import thortils

class ThorTrial(Trial):

    RESULT_TYPES = []

    def __init__(self, name, config, verbose=False):
        if self.__class__ == ThorTrial:
            raise ValueError("ThorTrial is generic. Please create the Trial object"\
                             "for your specific task!")
        super().__init__(name, config, verbose=verbose)

    def _start_controller(self):
        controller = thortils.launch_controller(self.config["thor"])
        return controller

    def run(self, logging=False):
        controller = self._start_controller()
        task_env = eval(self.config["task_env"])(controller, **self.config["task_env_config"])
        agent_class = eval(self.config["agent_class"])
        if agent_class.AGENT_USES_CONTROLLER:
            agent = agent_class(controller, **self.config["agent_config"])
        else:
            agent = agent_class(**self.config["agent_config"])

        max_steps = self.config["max_steps"]
        for i in range(1, max_steps+1):
            action = agent.act()
            observation = task_env.execute(action)
            agent.update(action, observation)

            _step_info = task_env.get_step_info(step=i)
            if logging:
                self.log_event(Event("Trial %s | %s" % (self.name, _step_info)))
            else:
                print(_step_info)

            if task_env.done(action):
                break
        results = task_env.compute_results()
        return results

    @property
    def scene(self):
        return self.config["thor"]["scene"]


class ThorObjectSearchTrial(ThorTrial):
    RESULT_TYPES = [PathsResult, HistoryResult]


def build_object_search_trial(scene, target, task_type,
                              max_steps=100,
                              goal_distance=defaults.GOAL_DISTANCE,
                              h_angles=defaults.H_ANGLES,
                              v_angles=defaults.V_ANGLES,
                              **thor_kwargs):
    """
    Returns a ThorTrial for object search.
    """
    task_config = {
        "task_type": task_type,
        "target": target,
        "goal_distance": goal_distance,
        "v_angles": v_angles,
        "h_angles": h_angles,
    }

    thor_config = {**defaults.CONFIG, **{"scene": scene}}
    thor_config.update(thor_kwargs)
    config = {
        "thor": thor_config,
        "max_steps": max_steps,
        "task_env": "ThorObjectSearch",
        "task_env_config": {"task_config": task_config},
        "agent_class": "ThorObjectSearchOptimalAgent",
        "agent_config": {"task_config": task_config,
                         "movement_params": thor_config["MOVEMENT_PARAMS"]}
    }

    trial = ThorObjectSearchTrial("test_optimal", config, verbose=True)
    return trial


def build_object_search_movements():
    """
    Returns a list of object search movement actions
    """
    actions = []
    for move_name in constants.MOVEMENT_PARAMS:
        actions.append(ThorAction(move_name, constants.MOVEMENT_PARAMS[move_name]))
    return actions
