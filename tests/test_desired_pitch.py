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

from cospomdp.models.sensors import pitch_facing
import random
from tqdm import tqdm

def _test():

    for i in tqdm(range(100)):
        r = (random.randint(0, 8), random.randint(0, 8), random.randint(0, 8))
        t = (random.randint(0, 8), random.randint(0, 8), random.randint(0, 8))
        pitch = pitch_facing(r, t)
        try:
            if t[2] < r[2]:
                assert pitch > 0
            else:
                assert pitch < 0
        except:
            import pdb; pdb.set_trace()
    print("Passed.")

if __name__ == "__main__":
    _test()
