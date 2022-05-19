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

import random
import pomdp_py
from cospomdp.utils.math import normalize, euclidean_dist
from tqdm import tqdm
from cospomdp.domain.state import CosState

def initialize_target_belief_2d(target, search_region, belief_type, prior, *args):
    def _prob(prior, loc):
        return prior.get(loc, 1.0)

    target_id, target_class = target
    if belief_type.startswith("histogram"):
        hist = normalize({
            search_region.object_state(target_id, target_class, loc): _prob(prior, loc)
            for loc in search_region.locations
        })
        return pomdp_py.Histogram(hist)

    else:
        raise NotImplementedError("belief_type {} is not yet implemented".format(belief_type))

def update_target_belief_2d(current_btarget,
                            next_srobot,
                            observation,
                            observation_model,
                            belief_type,
                            bu_args={}):
    """
    current_btarget: current target belief
    srobot: robot state corresponding to the observation.
    """
    random_starget = current_btarget.random()
    Starget_class = random_starget.__class__
    target_id = random_starget.id
    target_class = random_starget.objclass

    if belief_type.startswith("histogram"):
        assert isinstance(current_btarget, pomdp_py.Histogram)

        all_target_states = set(current_btarget.get_histogram().keys())
        if belief_type.endswith("approx"):
            bu_samples = min(len(all_target_states), 200)

            target_states_subset = set(random.sample(all_target_states,
                                                     bu_args.get("belief_samples", bu_samples)))
            # Include target states at locations in the observation
            for zi in observation:
                if zi.loc is not None:
                    target_states_subset.add(Starget_class(
                        target_id, target_class, zi.loc))
        else:
            target_states_subset = all_target_states

        new_btarget_hist = {}
        for starget in tqdm(all_target_states, desc=f"Belief Update ({belief_type})"):
            if starget in target_states_subset:
                state = CosState({target_id: starget,
                                  next_srobot.id: next_srobot})
                pr_z = observation_model.probability(observation, state)
                new_btarget_hist[starget] = pr_z * current_btarget[starget]

        for starget in all_target_states:
            if starget not in new_btarget_hist:
                nnstarget = min(target_states_subset,
                                key=lambda s: euclidean_dist(s.loc, starget.loc))
                new_btarget_hist[starget] = new_btarget_hist[nnstarget]
        new_btarget_hist = normalize(new_btarget_hist)
        new_btarget = pomdp_py.Histogram(new_btarget_hist)
    return new_btarget
