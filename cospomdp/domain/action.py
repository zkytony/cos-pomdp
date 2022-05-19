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


class Motion(pomdp_py.SimpleAction):
    """Motion moves the robot.
    The specific definition is domain-dependent"""

    def __repr__(self):
        return str(self)

class Done(pomdp_py.SimpleAction):
    """Declares the task to be over"""
    def __init__(self):
        super().__init__("done")

    def __repr__(self):
        return str(self)
