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

# This is a toy domain for 2D COS-POMDP
from cospomdp_apps.basic.common import solve

WORLD =\
"""
### map
R........
.x..xG...
.x.Tx....

### robotconfig
th: 0

### corr
T around G: d=2

### detectors
T: fan-nofp | fov=45, min_range=0, max_range=2 | (0.6, 0.1)
G: fan-nofp | fov=45, min_range=0, max_range=3 | (0.8, 0.1)

### goal
nav: G

### colors
T: [0, 22, 120]
G: [0, 210, 20]

### END
"""

if __name__ == "__main__":
    solve(WORLD, nsteps=50,
          solver="pomdp_py.POUCT",
          solver_args=dict(max_depth=50,
                           num_sims=1000,
                           discount_factor=0.95,
                           exploration_const=100))
