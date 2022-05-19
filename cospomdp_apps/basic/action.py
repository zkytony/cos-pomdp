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

# This instantiates the action space used in the COS-POMDP
# created for this domain.
from cospomdp.domain.action import Motion, Done

############################
# Actions
############################
STEP_SIZE=1
class Move2D(Motion):

    # Assumes (0,0) is at bottom-left, 0 degree is +x, counterclockwise.
    FORWARD = (STEP_SIZE, 0)
    LEFT_45 = (0, 45.0)
    LEFT_90 = (0, 90.0)
    RIGHT_45 = (0, -45.0)
    RIGHT_90 = (0, -90.0)
    LEFT = LEFT_45
    RIGHT = RIGHT_45

    def __init__(self, name, delta):
        """
        delta: (forward, angle)
        """
        self.name = name
        self.delta = delta
        super().__init__(name)

    def __repr__(self):
        return str(self)

    @property
    def dyaw(self):
        return self.delta[1]

MoveAhead = Move2D("MoveAhead", Move2D.FORWARD)
RotateLeft = Move2D("RotateLeft", Move2D.LEFT)
RotateRight = Move2D("RotateRight", Move2D.RIGHT)
ALL_MOVES_2D = {MoveAhead, RotateLeft, RotateRight}
