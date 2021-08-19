import sys
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# just so we can import mjolnir stuff
sys.path.append(os.path.join(ABS_PATH, '../../external/mjolnir/'))
from .mjolnir.datasets.glove import Glove
from .mjolnir.utils.class_finder import model_class, agent_class
from .mjolnir.utils.net_util import toFloatTensor
from .mjolnir.models.model_io import ModelInput, ModelOptions
from .mjolnir.datasets.offline_controller_with_small_rotation import ACTIONS_LIST

import torch

from .agent import ThorAgent
from .common import TOS_Action
from cospomdp.domain.action import ALL_MOVES_2D

class Args(object):
    pass

class StateInfo(object):
    # Used to pass in stuff to the NavigationAgent by-passing all the episode
    # business
    pass

def MJOLNIR_O_args(gpu_ids=[-1]):
    # Referencing models/mjolnir_o.py and utils/flag_parser.py
    args = Args()

    args.action_space = 6
    args.hidden_state_sz = 512
    args.dropout_rate = 0.25
    args.glove_file = os.path.join(ABS_PATH, "mjolnir", "data/thor_glove/glove_thorv1_300.hdf5")
    args.gpu_ids = gpu_ids
    args.gpu_id = gpu_ids[0]

    args.max_episode_length = 30
    args.episode_type = "BasicEpisode"
    args.strict_done = True

    args.rank = 0  # used to set random seed

    args.partial_reward = True   # using partial reward for parent objects
    args.eval = True
    args.seed = 1
    args.verbose = False

    args.learned_loss = False  # Only SAVN sets this to True.
    args.num_steps = -1        # This is originally set to 50; But we use our own.
    args.vis = False
    args.results_json = None   # we do not need this

    return args

class ThorObjectSearchExternalAgent(ThorAgent):
    AGENT_USES_CONTROLLER = True

    def __init__(self,
                 controller,
                 task_config,
                 model_name,
                 load_model_path,
                 args,
                 actions=ACTIONS_LIST + ["Done"]):
        """
        args (MJOLNIR_O_args): mimics the args parsed from command line
        """
        assert task_config['task_type'] == 'class',\
            "Cannot handle task type: {}".format(task_config['task_type'])
        target_class = task_config['target']
        self.gpu_ids = args.gpu_ids
        self.actions = actions

        print("Setting up {} player".format(model_name))
        if model_name.startswith("MJOLNIR"):
            model_create_fn = model_class(model_name)
            self.player = self.setup_nonadaptivea3c(
                model_create_fn,
                load_model_path,
                args)

        glove_file = args.glove_file
        glove = Glove(glove_file)
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[target_class][:], args.gpu_id
        )
        self._last_observation = None

    def setup_nonadaptivea3c(self, model_create_fn, load_model_path, args):
        # Basically following nonadaptivea3c_val, but without any offline business.
        shared_model = model_create_fn(args)

        print("Loading model...")
        try:
            saved_state = torch.load(
                load_model_path, map_location=lambda storage, loc: storage
            )
            shared_model.load_state_dict(saved_state)
        except:
            shared_model.load_state_dict(load_model_path)

        agent_type = "NavigationAgent"
        agent_create_fn = agent_class("NavigationAgent")
        player = agent_create_fn(model_create_fn,
                                 args, args.rank, args.gpu_ids[0])
        player.sync_with_shared(shared_model)
        player.reset_hidden()
        player.done = False
        return player

    def act(self):
        model_options = ModelOptions()  # this is meant to be nothing
        if self._last_observation is None:
            # This is the first time we call act. Pass in order to
            # receive observation from update.
            return TOS_Action("Pass", {})

        else:
            # last_observation should be a TOS_Observation
            state_info = StateInfo()
            state_info.objbb = {}
            for detection in self._last_observation.detections:
                xyxy = detection[0]
                cls = detection[1]
                state_info.objbb[cls] = xyxy
            state_info.frame = self._last_observation.img
            action_int, log_prob = self.player.action(
                model_options, False,
                state_info=state_info,
                glove_embedding=self.glove_embedding,
                just_return_action=True)
            action_name = self.actions[action_int.item()]
            action_params = {}
            if action_name in ALL_MOVES_2D:
                action_params = self.movement_params(action_name)
            return TOS_Action(action_name, action_params)

    def update(self, tos_action, tos_observation):
        self._last_observation = tos_observation


if __name__ == "__main__":
    #python -m cospomdp_apps.thor.external
    print("HELLO.")
    agent = ThorObjectSearchExternalAgent(
        None,
        {'target': 'Apple', 'task_type': 'class'},
        "MJOLNIR_O",
        os.path.join(ABS_PATH, "mjolnir", "trained_models", "mjolnir_train_4763_800_2021-08-19_04:03:08.dat"),
        args=MJOLNIR_O_args())
