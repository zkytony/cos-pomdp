import os

# These paths will be relative to the local computer

MODULE_PATH = os.path.dirname(__file__)

# The path to the directory to models
YOLOV5_MODEL_DIR = os.path.abspath(os.path.join(MODULE_PATH, "../../models/"))

# The path to the directory of data
YOLOV5_DATA_DIR = os.path.abspath(os.path.join(MODULE_PATH, "../../data/"))

# The path to the yolov5 repository
YOLOV5_REPO_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../external/yolov5/"))

# The path to the saved grid maps
GRID_MAPS_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../data/thor/grid_maps"))

# The path to the saved correlational distribtutions
CORR_DISTS_PATH = os.path.abspath(os.path.join(MODULE_PATH, "../../data/thor/corr_dists"))
