# Configuration for the THOR experiment.
# Uses constants defined in cospomdp_apps/thor
# by default

# Classes whose true positive rates are low indicating noisy detection
################# NOTE THESE NEED TO BE UPDATED ############
TARGET_CLASSES = {
    "kitchen": [
        ("Apple", 0.28),  # Class, True Positive Rate
        ("Mug", 0.28),
        ("Pot", 0.33),
        ("PepperShaker", 0.36),
        ("Sink", 0.46),
        ("Bread", 0.48),
        ("Potato", 0.50),
        ("Tomato", 0.61),
        ("Faucet", 0.67)
    ],
    "living_room": [
        ("DeskLamp", 0.29),
        ("CreditCard", 0.30),
        ("KeyChain", 0.42),
        ("FloorLamp", 0.43),
        ("GarbageCan", 0.56),
        ("Box", 0.58),
        ("Pillow", 0.60)
    ],
    "bedroom": [
        ("CreditCard", 0.29),
        ("CellPhone", 0.36),
        ("Book", 0.44),
        ("CD", 0.46),
        ("Painting", 0.53),
        ("AlarmClock", 0.65)
    ],
    "bathroom": [
        ("Candle", 0.35),
        ("TowelHolder", 0.41),
        ("SprayBottle", 0.43),
        ("ScrubBrush", 0.50),
        ("HandTowel", 0.67),
        ("Plunger", 0.67)
    ]
}

# Classes whose true positive rates are high, suitable as correlated objects to
# help search for the target objects.
CORRELATED_CLASSES = {
    "kitchen": [
        ("Fridge", 0.73),
        ("Toaster", 0.80),
        ("SoapBottle", 0.80),
        ("Microwave", 0.83),
        ("Lettuce", 0.86),
        ("StoveKnob", 0.93)
    ],
    "living_room": [
        ("Laptop", 0.71),
        ("Painting", 0.78),
        ("RemoteControl", 0.76),
        ("Television", 0.86),
        ("HousePlant", 0.82)
    ],
    "bedroom": [
        ("HousePlant", 0.70),
        ("Laptop", 0.79),
        ("Pillow", 0.81),
        ("DeskLamp", 0.87),
        ("Mirror", 0.93)
    ],
    "bathroom": [
        ("Towel", 0.74),
        ("Mirror", 0.74),
        ("Faucet", 0.83),
        ("SoapBottle", 0.88),
        ("Toilet", 0.92)
    ]
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
        for cls, rate in TARGET_CLASSES[catg]:
            assert cls in all_object_types,\
                "{} is not in objects in {}, {}\n"\
                "These objects are available:\n{}"\
                .format(cls, catg, scene, pprint(all_object_types))
        for cls, rate in CORRELATED_CLASSES[catg]:
            assert cls in all_object_types,\
                "{} is not in objects in {}, {}\n"\
                "These objects are available:\n{}"\
                .format(cls, catg, scene, pprint(all_object_types))
