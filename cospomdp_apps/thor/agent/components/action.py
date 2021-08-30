# Actions for ai2thor
from cospomdp.domain.action import Motion
from thortils.navigation import get_navigation_actions
from cospomdp_apps.basic.action import Move2D

class Move(Motion):
    """
    Different from Move2D, this action also models tilt (i.e. pitch).
    See thortils.navigation for action definition.

    Note that the forward is defined over the unit of a grid cell,
    and angles are in degrees.
    """
    def __init__(self, name, delta):
        """
        delta: (forward, h_angle, v_angle)
        self.name = name
        self.delta = delta
        """
        self.name = name
        self.delta = delta
        super().__init__(name)

    def __repr__(self):
        return str(self)

def navigation_actions(movement_params, grid_size):
    """
    movement_params (name -> {params})
    see thortils.navigation.convert_movement_to_action

    Returns navigation action suitable for GridMap coordinate system.

    The GridMap coordinate system looks like:

    y
    ^
    |
    + ---->x  0 deg, ccw
    z

    Rotation around z axis is yaw (look left/right).
    Rotation around x axis is pitch (look up/down)
    """
    # the action tuples here are in thor units
    action_tuples = get_navigation_actions(movement_params)
    actions = []
    for name, delta in action_tuples:
        forward, h_angle, v_angle = delta
        forward = forward / grid_size
        if name == "RotateLeft":
            h_angle = abs(h_angle)

        elif name == "RotateRight":
            h_angle = -abs(h_angle)

        elif name == "LookUp":
            v_angle = abs(v_angle)

        elif name == "LookDown":
            v_angle = -abs(v_angle)
        else:
            assert h_angle == 0.0
            assert v_angle == 0.0
        delta = (forward, h_angle, v_angle)
        actions.append(Move(name, delta))
    return actions

def navigation_actions2d(movement_params, grid_size):
    """
    movement_params (name -> {params})
    see thortils.navigation.convert_movement_to_action
    """
    actions = navigation_actions(movement_params, grid_size)
    actions2d = []
    for a in actions:
        if a.name not in {"LookUp", "LookDown"}:
            actions2d.append(Move2D(a.name, a.delta[:2]))
    return actions2d

def thor_action_params(action, grid_size):
    if len(action.delta) == 2:
        forward, h_angle = action.delta
        v_angle = 0
    else:
        forward, h_angle, v_angle = action.delta

    if action.name == "MoveAhead" or action.name == "MoveBack":
        return {"moveMagnitude": forward * grid_size}

    elif action.name == "RotateLeft":
        return {"degrees": abs(h_angle)}

    elif action.name == "RotateRight":
        return {"degrees": abs(h_angle)}

    elif action.name == "LookUp":
        return {"degrees": abs(v_angle)}

    elif action.name == "LookDown":
        return {"degrees": abs(v_angle)}

    else:
        raise ValueError("Unrecognized action {}".format(action.name))
