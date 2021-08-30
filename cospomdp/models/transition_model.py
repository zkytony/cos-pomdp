import math
from pomdp_py import TransitionModel

from ..utils.math import indicator, to_rad, fround
from ..domain.state import RobotState, CosState, ObjectState, RobotStatus

class RobotTransition(TransitionModel):
    """Models Pr(sr' | s, a); Likely domain-specific"""
    def __init__(self, robot_id):
        self.robot_id = robot_id


class CosTransitionModel(TransitionModel):
    """Cos-POMDP transition model Pr(s' | s, a)
    where the state s and s' are factored into robot and target states."""
    def __init__(self, target_id, robot_trans_model):
        self.target_id = target_id
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        starget = state.s(self.target_id)
        next_target_state = ObjectState(starget.id, starget.objclass, starget["loc"])
        robot_id = self.robot_trans_model.robot_id
        return CosState({robot_id: next_robot_state,
                         self.target_id: next_target_state})


class FullTransitionModel(TransitionModel):
    """F-POMDP transition model Pr(s' | s, a)
    where the state s and s' are factored into robot and n object states."""
    def __init__(self, robot_trans_model):
        self.robot_trans_model = robot_trans_model

    def sample(self, state, action):
        next_robot_state = self.robot_trans_model.sample(state, action)
        objstates = {next_robot_state.id:next_robot_state}
        for objid in state.object_states:
            if objid == next_robot_state.id:
                continue
            next_object_state = ObjectState(objid,
                                            state.s(objid).objclass,
                                            state.s(objid)['loc'])
            objstates[objid] = next_object_state
        return CosState(objstates)
