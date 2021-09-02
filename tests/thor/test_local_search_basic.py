from test_cosagent_complete import _test_complete_search
from cospomdp_apps.thor.agent.components.action import Stay



if __name__ == "__main__":
    components = _test_complete_search("PepperShaker", "StoveBurner",
                                       scene="FloorPlan1",
                                       num_sims=100,
                                       setup_only=True)
    agent = components['agent']
    handler = agent.handle(Stay(agent.robot_state().nid))
    print(handler.step())
