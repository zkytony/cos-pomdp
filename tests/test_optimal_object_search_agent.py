# Note:
from cosp.thor.trial import build_object_search_trial


def test_out_optimal_agent():
    trial = build_object_search_trial("PepperShaker", "class")
    trial.run(logging=True)


if __name__ == "__main__":
    test_out_optimal_agent()
