import pomdp_py
import os
import pickle
import yaml
import argparse

class ReplaySolver(pomdp_py.Planner):
    def __init__(self, history):
        self.history = history
        self.index = 1

    def plan(self, agent):
        info = self.history[self.index]
        return info['action']

    def update(self, agent, action, observation):
        self.index += 1


def main():
    parser = argparse.ArgumentParser(description="Replay")
    parser.add_argument("trial_path", type=str, help="path to trial directory")
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)
    with open(os.path.join(args.trial_path, "history.yaml"), "rb") as f:
        history = yaml.load(f, Loader=yaml.Loader)

    trial.config['agent_config']['solver'] = "ReplaySolver"
    trial.config['agent_config']['solver_args'] = {"history": history['history']}
    trial.config['visualize'] = True
    trial.config['viz_config'] = {"res": 30}
    trial.run()


if __name__ == "__main__":
    main()
