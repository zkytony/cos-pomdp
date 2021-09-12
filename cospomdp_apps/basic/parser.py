# This is convenient code for specifying
# a COS-POMDP instance and parsing the spec.

import numpy as np

from cospomdp.domain.state import RobotState2D
from cospomdp.models.agent import CosAgent
from cospomdp.models.search_region import SearchRegion2D
from cospomdp.models.observation_model import FanModelNoFP
from cospomdp.models.correlation import CorrelationDist
from cospomdp.models.reward_model import ObjectSearchRewardModel, NavRewardModel
from cospomdp.utils.math import euclidean_dist
from .transition_model import RobotTransition2D
from .policy_model import PolicyModel2D
from .belief import initialize_target_belief_2d

from cospomdp.utils.corr_funcs import *

################ World spec & Parsing #################
def parse_worldstr(worldstr):
    """
    worldstr spec see example below:

    ### map
    R....
    .x.Tx
    .xG.x

    ### robotconfig
    th: 0

    ### corr
    T around G: d=2

    ### detectors
    T: fan-nofp | fov=45, min_range=0, max_range=2 | (0.6, 0.1)
    G: fan-nofp | fov=45, min_range=0, max_range=3 | (0.8, 0.1)

    ### goal
    find: T, 2.0

    ### colors
    T: [225, 20, 20]
    G: [0, 210, 20]

    ### END

    Notes:
    - blank lines are skipped
    - '### END' is important
    - The specification of sensor parameters must be separated by commas,
      so does the correlation parameters
    """
    # Parse world string
    mode = None
    lines = {
        "map": [],
        "corr": [],
        "detectors": [],
        "goal": [],
        "robotconfig": [],
        "colors": []
    }
    sofar = {
        'robot_id': 'R'
    }
    for line in worldstr.split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("###"):
            new_mode = line.split("###")[1].strip()
            if mode is not None:
                parsed_stuff = eval(f"_handle_{mode}")(lines[mode], sofar)
                sofar.update(parsed_stuff)
            mode = new_mode
            if mode == "END":
                break
        else:
            lines[mode].append(line)
    if mode != "END":
        raise ValueError("worldstr syntax error: does not contain '### END'")
    return sofar

def create_instance(worldstr):
    """Given worldstr (see parse_worldstr for format), parse it
    and then construct a CosAgent accordingly with uniform prior,
    then return that agent.

    Returns: CosAgent, {objid: (x,y)}"""
    spec = parse_worldstr(worldstr)

    robot_id = spec["robot_id"]
    init_robot_pose = (*spec["init_robot_loc"], spec["robotconfig"]["th"])
    objects = spec["objects"]
    if isinstance(spec["reward_model"], ObjectSearchRewardModel):
        target_id = spec["reward_model"].target_id
    elif isinstance(spec["reward_model"], NavRewardModel):
        target_id = spec["dest_symbol"]
    else:
        raise ValueError("Cannot deal with reward model {}".format(type(spec["reward_model"])))
    search_region = spec["search_region"]
    reachable_positions = search_region.locations

    corr_dists = {}
    for key in spec['corr_funcs']:
        # Only care if it is related to the target
        obj1, obj2 = key
        if obj1 == target_id:
            other = obj2
        elif obj2 == target_id:
            other = obj1
        else:
            continue

        if other not in corr_dists:
            corr_func = eval(spec['corr_funcs'][key][0])
            corr_func_args = spec['corr_funcs'][key][1]
            corr_dists[other] = CorrelationDist(objects[other],
                                                objects[target_id],
                                                search_region,
                                                corr_func,
                                                corr_func_args=corr_func_args)
    detectors = spec["detectors"]
    reward_model = spec["reward_model"]
    init_robot_state = RobotState2D(robot_id, init_robot_pose)
    robot_trans_model = RobotTransition2D(robot_id, reachable_positions)
    policy_model = PolicyModel2D(robot_trans_model, reward_model)
    agent = CosAgent(objects[target_id],
                     init_robot_state,
                     search_region,
                     robot_trans_model,
                     policy_model,
                     corr_dists,
                     detectors,
                     reward_model,
                     initialize_target_belief_2d)
    if "colors" in spec:
        return agent, spec['objectlocs'], spec["colors"]
    else:
        return agent, spec['objectlocs']


def _handle_map(lines, *args):
    lines = list(reversed(lines))
    locations = []
    objects = {}
    objectlocs = {}
    init_robot_loc = None
    for y, line in enumerate(lines):
        for x, c in enumerate(line.strip()):
            if c != "x":
                locations.append((x,y))

                if c.isalpha() and c != "R":
                    objclass = c
                    objid = c  # id is same as class
                    objects[objid] = (objid, objclass)
                    objectlocs[objid] = (x,y)

                if c == "R":
                    init_robot_loc = (x,y)
    search_region = SearchRegion2D(locations)
    return dict(search_region=search_region,
                init_robot_loc=init_robot_loc,
                objects=objects,
                objectlocs=objectlocs)


def _handle_robotconfig(lines, *args):
    cfg = {}
    for line in lines:
        k, v = line.split(":")
        if v.strip().replace('.', ',', 1).isdigit():  #https://stackoverflow.com/a/23639915/2893053
            v = float(v.strip())
        cfg[k] = v
    return dict(robotconfig=cfg)

def _handle_corr(lines, *args):
    corr_funcs = {}
    for line in lines:
        line = line.strip()
        correl = line.split(":")[0].strip().split(" ")
        assert len(correl) == 3, f"Can't interpret relation {line}"
        params = line.split(":")[1].strip()

        obj1, rel, obj2 = correl
        rel = rel.strip()

        corr_funcs[(obj1, obj2)] = (rel, eval(f"dict({params})"))
        if rel.strip() == "around":
            corr_funcs[(obj2, obj1)] = (rel, eval(f"dict({params})"))
    return dict(corr_funcs=corr_funcs)

def _handle_detectors(lines, *args):
    detectors = {}
    for line in lines:
        line = line.strip()
        obj, config = line.split(":")
        obj = obj.strip()
        detector_type, sensor_params, quality_params = config.split("|")
        if detector_type.strip() == "fan-nofp":
            sensor_params = eval(f"dict({sensor_params.strip()})")
            quality_params = eval(quality_params.strip())
            detector = FanModelNoFP(obj, sensor_params, quality_params)
            detectors[obj] = detector
    return dict(detectors=detectors)

def _handle_goal(lines, sofar):
    assert len(lines) == 1, "only one goal."
    line = lines[0].strip()
    verb, args = line.split(":")
    adds = {}
    if verb.strip() == "find":
        target_obj = args.split(",")[0].strip()
        goal_dist = float(args.split(",")[1].strip())
        sensor = sofar['detectors'][target_obj].sensor
        reward_model = ObjectSearchRewardModel(sensor, goal_dist,
                                               sofar['robot_id'],
                                               target_obj)
    elif verb.strip() == "nav":
        parts = args.split(",")
        dest_symbol = parts[0].strip()
        # recognize destination specified by an object id,
        if dest_symbol in sofar["objects"]:
            th = 0
            if len(parts) == 2:
                th = eval(parts[1].strip())
            goal_pose = (*sofar["objectlocs"][dest_symbol], th)
        reward_model = NavRewardModel(goal_pose, sofar['robot_id'])
        adds["dest_symbol"] = dest_symbol
    return dict(reward_model=reward_model, **adds)

def _handle_colors(lines, *args):
    colors = {}
    for line in lines:
        line = line.strip()
        obj, color = line.split(":")
        colors[obj.strip()] = eval(color.strip())
    return dict(colors=colors)
