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

from cospomdp.models.search_region import SearchRegion2D
import numpy as np

def test_sr():
    locations = np.unique(np.random.randint(0, 5, size=(1000, 2)), axis=0)
    sr = SearchRegion2D(locations)
    assert sr.width == 5
    assert sr.length == 5
    assert sr.dim == (5,5)
