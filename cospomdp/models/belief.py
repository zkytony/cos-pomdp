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
from ..domain.state import CosState, ObjectState

class CosJointBelief(pomdp_py.OOBelief):
    def __init__(self, object_beliefs):
        super().__init__(object_beliefs)

    def random(self, rnd=random):
        return CosState(super().random(rnd=random, return_oostate=False))

    def mpe(self):
        # import pdb; pdb.set_trace()
        return CosState(super().mpe(return_oostate=False))

    def b(self, objid):
        return self.object_beliefs[objid]

    def set_b(self, objid, belief):
        self.object_beliefs[objid] = belief
