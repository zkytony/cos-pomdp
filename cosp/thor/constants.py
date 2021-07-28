from thortils.constants import *

# If you want to update any configuration,
# define them here.

TOS_REWARD_HI = 100
TOS_REWARD_LO = -100
TOS_REWARD_STEP = -1

# Moves diagonally
DIAG_MOVE = False

GOAL_DISTANCE = 1.0


KITCHEN_OBJECT_CLASSES = [
    # Small objects
    "PepperShaker",
    "ButterKnife",
    "Cup",
    "Fork",
    "Ladle",
    "SaltShaker",
    "ScrubBrush",
    "Spatula"
    # Medium objects
    "Bread",
    "Book",
    "Bowl",
    "Toaster",
    "Vase",
    "Statue"
    "CoffeeMachine",
    # Large objects
    "Cabinet",
    "Drawer",
    "DiningTable",
    "CoffeeTable",
    "Fridge",
    "Shelf",
    "CounterTop",
    "Oven"
]

LIVING_ROOM_OBJECT_CLASSES = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Box",
    "Bowl",
]

BATHROOM_OBJECT_CLASSES = [
    "HousePlant",
    "Lamp",
    "Book",
    "AlarmClock"
]

BEDROOM_OBJECT_CLASSES = [
    "Sink",
    "ToiletPaper",
    "SoapBottle",
    "LightSwitch"
]

ALL_OBJECT_CLASSES = KITCHEN_OBJECT_CLASSES +\
                     LIVING_ROOM_OBJECT_CLASSES +\
                     BATHROOM_OBJECT_CLASSES +\
                     BEDROOM_OBJECT_CLASSES
