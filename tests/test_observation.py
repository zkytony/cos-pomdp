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

from cospomdp.domain.observation import *


def test_observation():
    o1 = Loc(0, (5,5))
    o2 = Loc(0, (5,5))
    assert o1 == o2
    assert hash(o1) == hash(o2)

    o = CosObservation(None, {o1.id:o1, o2.id:o2})
    assert len(o) == 1
