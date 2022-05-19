# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pomdp_py
import os
import pickle
import yaml
import argparse
from .common import TOS_Action

class ReplaySolver(pomdp_py.Planner):
    def __init__(self, history):
        self.history = history
        self.index = 1

    def plan(self, agent):
        if self.index >= len(self.history):
            print("No more!")
            return None

        info = self.history[self.index]
        if type(info["action"]) == TOS_Action:
            return info["action"]
        else:
            return (info["action"]["goal"],
                    info["action"]["goal_done"],
                    info["action"]["base"])

    def update(self, *args):
        self.index += 1
        print("::::::::::::::::::::::::::::", self.index)


def main():
    parser = argparse.ArgumentParser(description="Replay")
    parser.add_argument("trial_path", type=str, help="path to trial directory")
    parser.add_argument("--save", help="save the sequence of images",
                        action='store_true')
    parser.add_argument("--gif", help="save the sequence of images",
                        action='store_true')
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)
    with open(os.path.join(args.trial_path, "history.yaml"), "rb") as f:
        history = yaml.load(f, Loader=yaml.Loader)

    # this works assuming other components are deterministic - this should be the case.
    trial.config['agent_config']['solver'] = "ReplaySolver"
    trial.config['agent_config']['solver_args'] = {"history": history['history']}
    trial.config['visualize'] = True
    trial.config['viz_config'] = {"res": 30}

    if args.save:
        trial.config['save_path'] = os.path.join(args.trial_path, "vis")
        trial.config['save_opts'] = {'gif': args.gif}
    trial.config['task_config']['detector_config']['plot_detections'] = True
    trial.run()


if __name__ == "__main__":
    main()
