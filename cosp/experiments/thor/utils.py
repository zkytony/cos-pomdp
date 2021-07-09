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
