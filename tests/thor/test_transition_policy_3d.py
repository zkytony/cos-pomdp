import cospomdp
import cospomdp_apps.thor.agent as agentlib
from cospomdp_apps.thor.agent import components
from cospomdp_apps.thor.agent.components.action import grid_pitch, grid_navigation_actions, grid_camera_look_actions
from cospomdp_apps.thor.agent.components.state import RobotState3D, ObjectState3D
from cospomdp_apps.thor.constants import V_ANGLES, MOVEMENT_PARAMS

STARGET = ObjectState3D("target", "target", (4,4), 3)

def _s(srobot):
    return cospomdp.CosState({"robot": srobot,
                              "target": STARGET})

def _test():
    reachable_positions = [(x,y) for x in range(10)
                           for y in range(10)]

    v_angles = [grid_pitch(v) for v in V_ANGLES]
    robot_trans_model = components.transition_model.RobotTransition3D(
        "robot", reachable_positions, v_angles
    )
    movement_params = MOVEMENT_PARAMS
    camera_look_actions = grid_camera_look_actions(movement_params)
    navigation_actions = grid_navigation_actions(movement_params, 0.25)

    sensor = cospomdp.FanSensor3D(min_range=1, max_range=5, fov=50, v_angles=v_angles)

    reward_model = cospomdp.ObjectSearchRewardModel(sensor,
                                                    1.0 / 0.25,
                                                    "robot",
                                                    "target")
    policy_model = components.policy_model.PolicyModel3D(robot_trans_model,
                                                         reward_model,
                                                         navigation_actions,
                                                         camera_look_actions)
    # A normal robot state - expecting all actions to be available
    state = _s(RobotState3D("robot", (0, 0, 0), 5, 0))
    assert policy_model.valid_moves(state)\
        == set(navigation_actions) | set(camera_look_actions)

    # Take LookDown a couple of times -- too many times
    ll = {a.name: a for a in grid_camera_look_actions(movement_params)}
    lookdown = ll["LookDown"]
    lookup = ll["LookUp"]

    ns = _s(robot_trans_model.sample(state, lookdown))
    assert ns.s("robot").pitch == lookdown.delta[2] % 360,\
        f"{ns.s('robot').pitch} != {lookdown.delta[2]}"
    print("Lookdown:", lookdown.delta[2])
    print("LookUp:", lookup.delta[2])

    print(policy_model.get_all_actions(ns))
    print(state.s("robot"))
    ns = _s(robot_trans_model.sample(state, lookdown))
    print(policy_model.get_all_actions(ns))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookdown))
    print(policy_model.get_all_actions(ns))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookdown))
    print(policy_model.get_all_actions(ns))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookdown))
    print(policy_model.get_all_actions(ns))
    print(ns.s("robot"))
    assert policy_model.valid_moves(ns)\
        == set(navigation_actions) | (set(camera_look_actions) - {lookdown})

    # Do the same for look up
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(state, lookup))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookup))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookup))
    print(ns.s("robot"))
    ns = _s(robot_trans_model.sample(ns, lookup))
    print(ns.s("robot"))
    assert policy_model.valid_moves(ns)\
        == set(navigation_actions) | (set(camera_look_actions) - {lookup})


if __name__ == "__main__":
    _test()
