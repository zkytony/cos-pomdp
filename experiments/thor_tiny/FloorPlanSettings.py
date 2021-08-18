from cospomdp.utils.corr_funcs import around

SETTINGS = {
    ####################################################################################33
    "FloorPlan1": {
        'combos': [
            ("Apple", "Book", (around, dict(d=3))),
            ("PepperShaker", "StoveBurner", (around, dict(d=2))),
            ("DishSponge", "SinkBasin", (around, dict(d=2))),
        ],

        'detectors': {
            "Apple": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "Book": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "PepperShaker": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "StoveBurner": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "DishSponge": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "SinkBasin": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
        }
    },
    ####################################################################################33
    "FloorPlan2": {
        'combos': [
            ("DishSponge", "SinkBasin", (around, dict(d=2))),
            ("PepperShaker", "StoveBurner", (around, dict(d=2))),
            ("Egg", "Bread", (around, dict(d=2))),
        ],

        'detectors': {
            "DishSponge": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "SinkBasin": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "PepperShaker": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "StoveBurner": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "Egg": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "Bread": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
        }
    },
    ####################################################################################33
    "FloorPlan3": {
        'combos': [
            ("DishSponge", "SinkBasin", (around, dict(d=2))),
            ("PepperShaker", "StoveBurner", (around, dict(d=2))),
            ("Cup", "CoffeeMachine", (around, dict(d=2))),
        ],

        'detectors': {
            "DishSponge": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "SinkBasin": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "PepperShaker": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "StoveBurner": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
            "Cup": ("fan-nofp", dict(fov=90, min_range=1, max_range=3), (0.7, 0.1)),
            "CoffeeMachine": ("fan-nofp", dict(fov=90, min_range=1, max_range=6), (0.8, 0.1)),
        }
    }
    ####################################################################################33
}
