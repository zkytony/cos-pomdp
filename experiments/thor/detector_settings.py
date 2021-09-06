# Configuration for the THOR experiment.
# Uses constants defined in cospomdp_apps/thor
# by default

# Classes whose true positive rates are low indicating noisy detection
################# NOTE THESE NEED TO BE UPDATED ############

CLASSES = {
    "kitchen": {
        "targets": [
            # Class, True Positive Rate (from confusion matrix), avg detection range (from table below)
            ("PepperShaker", 0.410959, 1.712714),
            ("DishSponge", 0.520000, 1.868989)
            ("Apple", 0.615385, 1.704362),
        ],
        "supports": [
            ("Toaster", 0.873786, 2.197788),
            ("StoveBurner", 0.945312, 1.732698),
            ("Microwave", 0.70, 2.449375),
            ("CoffeeMachine", 0.86, 2.266664)
        ]
    },
    "living_room": {
        "targets": [
            ("CreditCard", 0.28, 1.641105),
            ("Laptop", 0.63, 2.694615),
            ("GarbageCan", 0.64, 3.641513)
        ],
        "supports": [
            ("FloorLamp", 0.72, 3.586244),
            ("Painting", 0.74, 3.353564),
            ("RemoteControl", 0.72, 1.680573),
            ("Television", 0.85, 2.474153),
            ("Sofa", 0.75, 2.880409)
        ]
    },
    "bedroom": {
        "targets": [
            ("CellPhone", 0.53, 1.649831),
            ("Book", 0.60, 2.042013),
            ("CD", 0.44, 1.592458)
        ],
        "supports": [
            ("AlarmClock", 0.79, 2.712512),
            ("Laptop", 0.77, 2.096662),
            ("Pillow", 0.88, 2.706392),
            ("DeskLamp", 0.83, 2.444946),
            ("Mirror", 0.85, 2.187369)
        ]
    },
    "bathroom": {
        "targets": [
            ("Candle", 0.51, 1.420458),
            ("SprayBottle", 0.49, 1.755695),
            ("ScrubBrush", 0.34, 1.963239),
            ("Plunger", 0.45, 2.046129)
        ],
        "supports": [
            ("Towel", 0.79, 1.674387),
            ("Mirror", 0.78, 1.918279),
            ("HandTowel", 0.86, 2.015952),
            ("Sink", 0.76, 1.450576),
            ("Toilet", 0.92, 1.786077)
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
