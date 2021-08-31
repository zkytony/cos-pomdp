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


class MoveTopo(Motion):
    def __init__(self, src_nid, dst_nid, gdist=None,
                 cost_scaling_factor=1.0):
        """
        Moves the robot from src node to dst node
        """
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.gdist = gdist
        self._cost_scaling_factor = cost_scaling_factor
        super().__init__("move({}->{})".format(self.src_nid, self.dst_nid))

    @property
    def step_cost(self):
        return -(self.gdist * self._cost_scaling_factor)


def grid_navigation_actions2d(movement_params, grid_size):
    """
    movement_params (name -> {params})
    see thortils.navigation.convert_movement_to_action
    """
    actions = grid_navigation_actions(movement_params, grid_size)
    actions2d = []
    for a in actions:
        if a.name not in {"LookUp", "LookDown"}:
            actions2d.append(Move2D(a.name, a.delta[:2]))
    return actions2d


def grid_navigation_actions(movement_params, grid_size):
    """
    movement_params (name -> {params})
    e.g. {"MoveAhead": {"moveMagnitude": 0.25}}
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


def from_grid_action_to_thor_action_params(action, grid_size):
    """Returns a dictionary used to pass in Controller.step()
    as parameters for the action with name `action.name`."""
    if len(action.delta) == 2:
        forward, h_angle = action.delta
        v_angle = 0
    else:
        forward, h_angle, v_angle = action.delta

    if action.name == "MoveAhead" or action.name == "MoveBack":
        return {"moveMagnitude": forward * grid_size}

    elif action.name == "RotateLeft" or action.name == "RotateRight":
        return {"degrees": abs(h_angle)}

    elif action.name == "LookUp" or action.name == "LookDown":
        return {"degrees": abs(v_angle)}

    else:
        raise ValueError("Unrecognized action {}".format(action.name))


def from_grid_action_to_thor_action_delta(action, grid_size):
    """Given a Move action (in grid map coordinates),
    Returns a tuple (forward, h_angles, v_angles).

    The ai2thor coordinate system looks like:

    z 0 deg cw
    ^
    |
    + ---->x  0
    y

    """
    if len(action.delta) == 2:
        forward, h_angle = action.delta
        v_angle = 0
    else:
        forward, h_angle, v_angle = action.delta

    if action.name == "MoveAhead":
        return (forward*grid_size, 0.0, 0.0)

    elif action.name == "RotateLeft":
        return (0.0, -abs(h_angle), 0.0)

    elif action.name == "RotateRight":
        return (0.0, abs(h_angle), 0.0)

    elif action.name == "LookUp":
        return (0.0, 0.0, -abs(v_angle))

    elif action.name == "LookDown":
        return (0.0, 0.0, abs(v_angle))

    else:
        raise ValueError("Unrecognized action {}".format(action.name))
