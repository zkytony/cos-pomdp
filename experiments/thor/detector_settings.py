# Configuration for the THOR experiment.
# Uses constants defined in cospomdp_apps/thor
# by default

# Classes whose true positive rates are low indicating noisy detection
################# NOTE THESE NEED TO BE UPDATED ############
CLASSES = {
    "kitchen": {
        "targets": [
            ("PepperShaker", 0.22),  # Class, True Positive Rate
            ("Bowl", 0.48),
            ("Tomato", 0.52),
            ("Pot", 0.59),
            ("Bread", 0.56)
        ],
        "supports": [
            ("Fridge", 0.72),
            ("Toaster", 0.74),
            ("StoveBurner", 0.72),
            ("Microwave", 0.70),
            ("CoffeeMachine", 0.86)
        ]
    },
    "living_room": {
        "targets": [
            ("KeyChain", 0.14),
            ("CreditCard", 0.28),
            ("Laptop", 0.63),
            ("GarbageCan", 0.64)
        ],
        "supports": [
            ("FloorLamp", 0.72),
            ("Painting", 0.74),
            ("RemoteControl", 0.72),
            ("Television", 0.85),
            ("Sofa", 0.75)
        ]
    },
    "bedroom": {
        "targets": [
            ("CellPhone", 0.53),
            ("Book", 0.60),
            ("CD", 0.44)
        ],
        "supports": [
            ("AlarmClock", 0.79),
            ("Laptop", 0.77),
            ("Pillow", 0.88),
            ("DeskLamp", 0.83),
            ("Mirror", 0.85)
        ]
    },
    "bathroom": {
        "targets": [
            ("Candle", 0.51),
            ("SprayBottle", 0.49),
            ("ScrubBrush", 0.34),
            ("Plunger", 0.45)
        ],
        "supports": [
            ("Towel", 0.79),
            ("Mirror", 0.78),
            ("HandTowel", 0.86),
            ("Sink", 0.76),
            ("Toilet", 0.92)
        ]
    }
}

if __name__ == "__main__":
    print("Verifying configuration correctness (no typos etc.)")
    import thortils as tt
    from pprint import pprint

    scenes = [
        ("kitchen", "FloorPlan21"),
        ("living_room", "FloorPlan221"),
        ("bedroom", "FloorPlan321"),
        ("bathroom", "FloorPlan421")
    ]
    for catg, scene in scenes:
        controller = tt.launch_controller({"scene": scene})
        all_object_types = tt.thor_all_object_types(controller)
        for cls, rate in CLASSES[catg]["targets"]:
            assert cls in all_object_types,\
                "{} is not in objects in {}, {}\n"\
                "These objects are available:\n{}"\
                .format(cls, catg, scene, pprint(all_object_types))
        for cls, rate in CLASSES[catg]["supports"]:
            assert cls in all_object_types,\
                "{} is not in objects in {}, {}\n"\
                "These objects are available:\n{}"\
                .format(cls, catg, scene, pprint(all_object_types))
