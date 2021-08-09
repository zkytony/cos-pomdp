import datetime
import time
import pytz
from pytz import reference as pytz_reference
from pomdp_py.utils import typ

# COS-POMDP specific
def resolve_robot_target_args(robot_id, target_id, *args):
    """
    args can be {robot_id: robotthing, target_id: targetthing}
    or (robotthing, targetthing) or robotthing, targetthing.
    Returns robothting, targetthing
    """
    if len(args) == 1:
        things = args[0]
        if type(things) == dict:
            assert len(things) == 2\
                and (robot_id in things and target_id in things)
            return things[robot_id], things[target_id]

        elif hasattr(things, "__len__") and type(things) != str:
            assert len(things) == 2
            return things

        else:
            raise ValueError("cannot handle argument {}".format(args[0]))

    elif len(args) == 2:
        return args

    else:
        raise ValueError("cannot handle argument(s) {}".format(args))



# Printing
def json_safe(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {json_safe(key):json_safe(value) for key, value in obj.items()}
    else:
        return str(obj)
    return obj


def diff(rang):
    return rang[1] - rang[0]

def in_range(x, rang):
    return x >= rang[0] and x < rang[1]

def in_range_inclusive(x, rang):
    return x >= rang[0] and x <= rang[1]

def in_region(p, ranges):
    return in_range(p[0], ranges[0]) and in_range(p[1], ranges[1]) and in_range(p[2], ranges[2])

def remap(oldval, oldmin, oldmax, newmin, newmax, enforce=False):
    newval = (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
    if enforce:
        return min(max(newval, newmin), newmax)
    else:
        return newval


# Others
def safe_slice(arr, start, end):
    true_start = max(0, min(len(arr)-1, start))
    true_end = max(0, min(len(arr)-1, end))
    return arr[true_start:true_end]

# Others
def discounted_cumulative_reward(rewards, discount_factor):
    total = 0
    d = 1.0
    for r in rewards:
        total += r*d
        d *= discount_factor
    return total



########## Python utils
def nice_timestr(dtobj=None):
    """pass in a datetime.datetime object `dt`
    and get a nice time string. If None is passed in,
    then get string for the current time"""
    if dtobj is None:
        dtobj = datetime.datetime.now()

    localtime = pytz_reference.LocalTimezone()
    return dtobj.strftime("%a, %d-%b-%Y %I:%M:%S, " + localtime.tzname(dtobj))
