import sys
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# just so we can import mjolnir stuff
sys.path.append(os.path.abspath('../../external/mjolnir/'))
from datasets.glove import Glove
from utils.class_finder import model_class

import torch

from .agent import ThorAgent

class Args(object):
    pass

def MJOLNIR_O_args(gpu_ids=-1):
    # Referencing models/mjolnir_o.py and utils/flag_parser.py
    args = Args()
    args.action_space = 6
    args.hidden_state_sz = 512
    args.dropout_rate = 0.25
    args.glove_file = os.path.join(ABS_PATH, "mjolnir", "data/thor_glove/glove_thorv1_300.hdf5")
    args.gpu_ids = gpu_ids
    return args


class ThorObjectSearchExternalAgent(ThorAgent):
    AGENT_USES_CONTROLLER = True

    def __init__(self,
                 controller,
                 model_name,
                 load_model_path,
                 model_args):

        if model_name.startswith("MJOLNIR"):
            model_create_fn = model_class(model_name)
            self.model = self.make_model_nonadaptivea3c(
                model_create_fn,
                load_model_path,
                model_args)


    def make_model_nonadaptivea3c(self,
                                  model_create_fn,
                                  load_model_path,
                                  model_args):
        glove_file = model_args.glove_file
        glove = Glove(glove_file)
        shared_model = model_create_fn(model_args)

        try:
            saved_state = torch.load(
                load_model_path, map_location=lambda storage, loc: storage
            )
            shared_model.load_state_dict(saved_state)
        except:
            shared_model.load_state_dict(load_model_path)




if __name__ == "__main__":
    print("HELLO.")
    agent = ThorObjectSearchExternalAgent(
        None, "MJOLNIR_O",
        os.path.join(ABS_PATH, "mjolnir", "trained_models", "mjolnir_train_4763_800_2021-08-19_04:03:08.dat"),
        model_args=MJOLNIR_O_args())
