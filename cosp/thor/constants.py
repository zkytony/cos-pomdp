# Refer to thortils/constants.py for full (default) configuration
#-------------------------------------------------------------------------------
GRID_SIZE = 0.25
MOVE_STEP_SIZE = GRID_SIZE

H_ROTATION = 45   # Only 90 won't stuck
V_ROTATION = 30

FOV = 90   # from official doc: The default field of view when agentMode="default" is 90 degrees.

VISIBILITY_DISTANCE = 1.5
INTERACTION_DISTANCE = 1.5   # objects farther away than this cannot be interacted with.
AGENT_MODE = "default"   # from official doc: For iTHOR, it is often safest to stick with the default agent.

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

RENDER_DEPTH = True
RENDER_INSTANCE_SEGMENTATION = True

# Need in order to not stuck the agent for sub-90 degree rotation.
# BUT, it actually DOES NOT WORK
CONTINUOUS = True
SNAP_TO_GRID = not CONTINUOUS

#------------------------------------------------------------------------------
# Ai2thor parameters related to object search
GOAL_DISTANCE = 1.0
MAX_STEPS = 100

SCATTER_GRANULARITY = GRID_SIZE*2

#------------------------------------------------------------------------------

# Create config defined in this file as a dictionary
def _load_config():
    config = {}
    for k, v in globals().items():
        if k.startswith("__"):
            continue
        if callable(eval(k)):
            continue
        config[k] = v
    return config
CONFIG = _load_config()
