# Utility functions for thor & its experimentation

from ai2thor.controller import Controller


def spl_ratio(li, pi, Si):
    """spl ratio for a single trial.
    li, pi, Si stands for shortest_path_length, actual_path_length, success for trial i.
    """
    pl_ratio = max(pi, li) if max(pi, li) > 0 else 1.0
    return float(Si) * li / pl_ratio


def compute_spl(episode_results):
    """
    Reference: https://arxiv.org/pdf/1807.06757.pdf

    Args:
        episode_results (list) List of tuples
            (shortest_path_length, actual_path_length, success),
             as required by the formula. `actual_path_length` and
            `shortest_path_length` are floats; success is boolean.
    Return:
        float: the SPL metric
    """
    # li, pi, Si stands for shortest_path_length, actual_path_length, success for trial i
    return sum(spl_ratio(li, pi, Si)
               for li, pi, Si in episode_results) / len(episode_results)


def _resolve(event_or_controller):
    """Returns an event, whether the given parameter is an event (already)
    or a controller"""
    if isinstance(event_or_controller, Controller):
        return event_or_controller.step(action="Pass")
    else:
        return event_or_controller  # it's just an event


def thor_visible_objects(event_or_controller):
    event = _resolve(event_or_controller)
    thor_objects = thor_get(event, "objects")
    result = []
    for obj in thor_objects:
        if obj["visible"]:
            result.append(obj)
    return result


def thor_get(event, *keys):
    """Get the true environment state, which is the metadata in the event returned
    by the controller. If you would like a particular state variable's value,
    pass in a sequence of string keys to retrieve that value.
    For example, to get agent pose, you call:

    env.state("agent", "position")"""
    if len(keys) > 0:
        d = event.metadata
        for k in keys:
            d = d[k]
        return d
    else:
        return event.metadata


def thor_agent_pose2d(event_or_controller):
    """Returns a tuple (x, y, th), a 2D pose
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    rotation = thor_get(event, "agent", "rotation")
    return position["x"], position["z"], rotation["y"]

def thor_agent_pose(event_or_controller):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    rot: dict (x=, y=, z=)
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    rotation = thor_get(event, "agent", "rotation")
    return position, rotation

def thor_agent_position(event_or_controller):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    return position

def thor_camera_pose(event_or_controller, get_tuples=False):
    """
    This is exactly the same as thor_agent_pose
    except that the pitch of the rotation is set
    to camera horizon. Everything else is the same.
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    rotation = thor_get(event, "agent", "rotation")
    assert abs(rotation["z"]) < 1e-3  # assert that there is no roll
    cameraHorizon = thor_get(event, "agent", "cameraHorizon")
    if get_tuples:
        return (position["x"], position["y"], position["z"]),\
            (cameraHorizon, rotation["y"], 0)
    else:
        return position, dict(x=cameraHorizon, y=rotation["y"], z=0)
