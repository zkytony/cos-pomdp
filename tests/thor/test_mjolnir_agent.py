import os

from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from cospomdp_apps.thor.external import ThorObjectSearchExternalAgent, MJOLNIR_O_args
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.join(ABS_PATH, "../../external/mjolnir", "pretrained_models")

MODEL_NAME = "MJOLNIR_O"
LOAD_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "mjolnir_o_pretrain.dat")#train_4763_800_2021-08-19_04:03:08.dat")


def _test_mjolnir_agent():
    args = TaskArgs(detectables='any',
                    scene='FloorPlan329',
                    target="Painting",
                    agent_class="ThorObjectSearchExternalAgent",
                    task_env="ThorObjectSearch")
    config = make_config(args)
    config['agent_config']['model_name'] = MODEL_NAME
    config['agent_config']['load_model_path'] = LOAD_MODEL_PATH
    config['agent_config']['args'] = MJOLNIR_O_args(gpu_ids=[0])
    config['visualize'] = False
    trial = ThorObjectSearchTrial("test_mjolnir", config, verbose=True)
    print("Trial created")
    trial.run()

if __name__ == "__main__":
    _test_mjolnir_agent()
