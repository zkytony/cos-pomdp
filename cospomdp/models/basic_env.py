import pomdp_py
from ..domain.state import (CosState2D, ObjectState2D)
from .transition_model import (RobotTransition2D,
                               FullTransitionModel2D)

class BasicEnv2D(pomdp_py.Environment):
    """This is meant for providing a basic simulation environment
    for COS-POMDP. If you are using e.g. Thor, you may not need this."""
    def __init__(self, init_robot_state, objlocs, target_id,
                 reachable_positions, reward_model):
        objstates = {objid: ObjectState2D(objid, objid, objlocs[objid])
                     for objid in objlocs}
        init_state = CosState2D({**{init_robot_state.id:init_robot_state},
                                 **objstates})
        robot_trans_model = RobotTransition2D(init_robot_state.id,
                                              reachable_positions)
        transition_model = FullTransitionModel2D(robot_trans_model)
        super().__init__(init_state, transition_model=transition_model,
                         reward_model=reward_model)
