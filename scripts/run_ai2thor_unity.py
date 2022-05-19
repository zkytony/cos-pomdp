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

from ai2thor.controller import Controller
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

LOAD_EXECUTABLE_PATH = os.path.join("/home/kaiyuzh/.ai2thor/releases/thor-201903131714-Linux64/thor-201903131714-Linux64")#repo/ai2thor/unity/builds/thor-Linux64-local/thor-Linux64-local")

controller = Controller(
    local_executable_path=LOAD_EXECUTABLE_PATH
)
controller.start()
