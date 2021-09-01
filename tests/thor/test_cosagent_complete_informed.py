from test_cosagent_complete import _test_complete_search

def step_act_cb(task_env, agent, **kwargs):
    pass


if __name__ == "__main__":
    _test_complete_search("Bowl", "Book",
                          scene="FloorPlan1",
                          num_sims=40,
                          prior='informed')
