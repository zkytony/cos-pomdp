import datetime
import time
import pytz
from pytz import reference as pytz_reference
from pomdp_py.utils import typ
from . import cfg

__all__ = ['_debug',
           'resolve_robot_target_args',
           'discounted_cumulative_reward']

def _debug(content, p="yellow", lev=1, c=None):
    """p: a string making function (e.g. typ.blue),
    or more conveniently, just a string 'blue'.
    If you want to bold, then do 'bold-blue'.
    """
    if c is not None:
        p = c
    bold = False
    if type(p) == str:
        if p.startswith("bold"):
            bold = True
            p = p.split("-")[1]
        p = eval(f"typ.{p}")
    if cfg.DEBUG_LEVEL >= lev:
        if bold:
            print(typ.bold(p(content)))
        else:
            print(p(content))


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



def discounted_cumulative_reward(rewards, discount_factor):
    total = 0
    d = 1.0
    for r in rewards:
        total += r*d
        d *= discount_factor
    return total
