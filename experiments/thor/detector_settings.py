# Configuration for the THOR experiment.
# Uses constants defined in cospomdp_apps/thor
# by default

# Classes whose true positive rates are low indicating noisy detection
################# NOTE THESE NEED TO BE UPDATED ############
CLASSES = {
    "kitchen": {
        "targets": [
            # Class, True Positive Rate (from confusion matrix), avg detection range (from table below)
            ("PepperShaker", 0.22, 1.712714),
            ("Bowl", 0.48, 1.789045),
            ("Tomato", 0.52, 1.548525),
            ("Pot", 0.59, 1.831025),
            ("Bread", 0.56, 2.113790)
        ],
        "supports": [
            ("Fridge", 0.72, 2.117456),
            ("Toaster", 0.74, 2.197788),
            ("StoveBurner", 0.72, 1.732698),
            ("Microwave", 0.70, 2.449375),
            ("CoffeeMachine", 0.86, 2.266664)
        ]
    },
    "living_room": {
        "targets": [
            ("KeyChain", 0.14, 1.692580),
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


#######################
# --- Average TP Detection Distance to Agent (kitchen) ---
#                    conf  agent_dist
# class
# Apple          0.514675    1.704362
# Bowl           0.584905    1.789045
# Bread          0.753566    2.113790
# ButterKnife    0.269287         NaN
# Cabinet        0.738433    2.127686
# CoffeeMachine  0.739963    2.266664
# CounterTop     0.572546    1.777759
# Cup            0.456179    1.370557
# DishSponge     0.750188    1.868989
# Egg            0.461310    1.239266
# Faucet         0.611865    1.805268
# Floor          0.692701    2.857364
# Fork           0.313639         NaN
# Fridge         0.707805    2.117456
# GarbageCan     0.708028    2.146102
# Knife          0.374021    1.266382
# Lettuce        0.775978    1.990603
# LightSwitch    0.771444    2.498457
# Microwave      0.733685    2.449375
# Mug            0.458949    1.722366
# Pan            0.437905    2.128699
# PepperShaker   0.371662    1.712714
# Plate          0.694175    1.758784
# Pot            0.630622    1.831025
# Potato         0.573553    1.883032
# SaltShaker     0.410324    2.029399
# Sink           0.534758    1.798008
# SoapBottle     0.602826    2.115424
# Spatula        0.370509    1.070613
# Spoon          0.306501    1.912021
# StoveBurner    0.711078    1.732698
# StoveKnob      0.738261    1.907450
# Toaster        0.756345    2.197788
# Tomato         0.718066    1.548525
#######################
# --- Average TP Detection Distance to Agent (living_room) ---
#                    conf  agent_dist
# class
# Box            0.786729    2.188811
# CreditCard     0.565715    1.641105
# Floor          0.607458    2.803557
# FloorLamp      0.750061    3.586244
# GarbageCan     0.744133    3.641513
# HousePlant     0.761207    3.261840
# KeyChain       0.422840    1.692580
# Laptop         0.818030    2.694615
# LightSwitch    0.729957    2.938222
# Painting       0.819409    3.353564
# Pillow         0.889551    2.749840
# RemoteControl  0.633211    1.680573
# Sofa           0.741703    2.880409
# Television     0.811412    2.474153
#############################
# --- Average TP Detection Distance to Agent (bedroom) ---
#                  conf  agent_dist
# class
# AlarmClock   0.697623    2.712512
# Bed          0.709187    2.422100
# Book         0.683029    2.042013
# CD           0.751475    1.592458
# CellPhone    0.643272    1.649831
# CreditCard   0.666992    1.577886
# DeskLamp     0.843819    2.444946
# Floor        0.607262    2.078250
# GarbageCan   0.747788    2.552600
# KeyChain     0.593924    1.616284
# Laptop       0.785645    2.096662
# LightSwitch  0.778797    2.312341
# Mirror       0.807523    2.187369
# Pen          0.436731    1.499158
# Pencil       0.431795    1.753464
# Pillow       0.808403    2.706392
# Window       0.738548    2.673544
#############################
# --- Average TP Detection Distance to Agent (bathroom) ---
#                        conf  agent_dist
# class
# Candle             0.635133    1.420458
# Cloth              0.652630    1.670997
# Faucet             0.713005    1.643623
# Floor              0.661479    2.582650
# GarbageCan         0.732545    1.772476
# HandTowel          0.814438    2.015952
# HandTowelHolder    0.875222    2.073436
# LightSwitch        0.821500    1.905857
# Mirror             0.715110    1.918279
# Plunger            0.651194    2.046129
# ScrubBrush         0.593970    1.963239
# Sink               0.716122    1.450576
# SoapBar            0.664647    1.436793
# SoapBottle         0.740175    1.677307
# SprayBottle        0.607558    1.755695
# Toilet             0.804277    1.786077
# ToiletPaper        0.747924    1.689702
# ToiletPaperHanger  0.742238    1.964909
# Towel              0.803889    1.674387
# TowelHolder        0.549228    1.639580
