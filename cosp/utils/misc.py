import datetime
import time
import pytz
from pytz import reference as pytz_reference

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

# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def disable():
        bcolors.WHITE   = ''
        bcolors.CYAN    = ''
        bcolors.MAGENTA = ''
        bcolors.BLUE    = ''
        bcolors.GREEN   = ''
        bcolors.YELLOW  = ''
        bcolors.RED     = ''
        bcolors.ENDC    = ''

    @staticmethod
    def s(color, content):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        return color + content + bcolors.ENDC

def print_info(content):
    print(bcolors.s(bcolors.BLUE, content))

def print_note(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_error(content):
    print(bcolors.s(bcolors.RED, content))

def print_warning(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_success(content):
    print(bcolors.s(bcolors.GREEN, content))

def print_info_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.BLUE, content))

def print_note_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))

def print_error_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.RED, content))

def print_warning_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.YELLOW, content))

def print_success_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))
# For your convenience:
# from moos3d.util import print_info, print_error, print_warning, print_success, print_info_bold, print_error_bold, print_warning_bold, print_success_bold

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
