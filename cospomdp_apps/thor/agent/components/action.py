# Actions for ai2thor
from cospomdp.domain.action import Motion
from thortils.navigation import get_navigation_actions
from cospomdp_apps.basic.action import Move2D
from cospomdp_apps.thor.common import TOS_Action

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

    @property
    def dyaw(self):
        return self.delta[1]

    @property
    def dpitch(self):
        return self.delta[2]


class MoveTopo(Motion):
    def __init__(self, src_nid, dst_nid, gdist=None,
                 cost_scaling_factor=1.0, atype="move"):
        """
        Moves the robot from src node to dst node
        """
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.gdist = gdist
        self._cost_scaling_factor = cost_scaling_factor
        super().__init__("{}({}->{})".format(atype, self.src_nid, self.dst_nid))

    @property
    def step_cost(self):
        return min(-(self.gdist * self._cost_scaling_factor), -1)

class Stay(MoveTopo):
    def __init__(self, nid, cost_scaling_factor=1.0):
        super().__init__(nid, nid, gdist=0.0, cost_scaling_factor=1.0, atype="stay")

class MoveViewpoint(Motion):
    def __init__(self, dst_pose):
        self.dst_pose = dst_pose
        super().__init__("move_to_viewpoint({})".format(dst_pose))


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

def thor_camera_look_actions(movement_params):
    looks = {}
    for a in movement_params:
        if a in {"LookUp", "LookDown"}:
            looks[a] = (TOS_Action(a, movement_params[a]))
    return looks

def grid_camera_look_actions(movement_params):
    """In ai2thor, "Since the agent looks up and down in 30 degree
    increments, it most common for the horizon to be in { -30, 0, 30,
    60 }.

    Negative camera horizon values correspond to agent looking up, whereas
    positive horizon values correspond to the agent looking down.
    """
    actions = []
    for action_name in movement_params:
        if action_name in {"LookUp", "LookDown"}:
            degrees = abs(movement_params[action_name]["degrees"])
            if action_name == "LookDown":
                degrees = -degrees
            delta = (0, 0, degrees)
            actions.append(Move(action_name, delta))
    return actions

def grid_pitch(thor_pitch):
    return thor_pitch % 360

def grid_h_angles(thor_h_angles):
    """accepted h angles; should be identical with thor, in fact.
    See thortils.GridMap fro details."""
    return [(90 - thor_yaw) % 360
            for thor_yaw in thor_h_angles]

def grid_height_range(thor_height_range, grid_size):
    """thor_height_range (min_height, max_height);
    Returns this range in grid coordinates; Note that
    height range is continuous."""
    return (thor_height_range[0] / grid_size,
            thor_height_range[1] / grid_size)

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
            continue  # will be handled by grid_camera_look_actions

        elif name == "LookDown":
            continue  # will be handled by grid_camera_look_actions

        else:
            assert h_angle == 0.0
            assert v_angle == 0.0
        delta = (forward, h_angle, v_angle)
        actions.append(Move(name, delta))
    return actions

def from_thor_delta_to_thor_action_params(name, delta):
    return _to_thor_action_params(name, delta)

def from_grid_action_to_thor_action_params(action, grid_size):
    """Returns a dictionary used to pass in Controller.step()
    as parameters for the action with name `action.name`."""
    if len(action.delta) == 2:
        forward, h_angle = action.delta
        v_angle = 0
    else:
        forward, h_angle, v_angle = action.delta
    delta = list(action.delta)
    delta[0] *= grid_size  # delta[0] is forward
    return _to_thor_action_params(action.name, tuple(delta))

def _to_thor_action_params(name, delta):
    if len(delta) == 2:
        forward, h_angle = delta
        v_angle = 0
    else:
        forward, h_angle, v_angle = delta
    if name == "MoveAhead" or name == "MoveBack":
        return {"moveMagnitude": forward}

    elif name == "RotateLeft" or name == "RotateRight":
        return {"degrees": abs(h_angle)}

    elif name == "LookUp" or name == "LookDown":
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
