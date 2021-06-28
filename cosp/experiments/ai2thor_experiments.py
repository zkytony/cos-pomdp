from ..config.ai2thor_config import load_config
import pprint

def generate_experiment_trials():
    config = load_config()
    pprint.pprint(config)
