from test_cosagent_complete import _test_complete_search

def step_act_cb(task_env, agent, **kwargs):
    # if agent.belief.b(agent.robot_id).mpe().nid == 1:
    #     import pdb; pdb.set_trace()
    pass


if __name__ == "__main__":
    _test_complete_search("PepperShaker", "StoveBurner",
                          scene="FloorPlan1",
                          num_sims=40,
                          prior='informed',
                          step_act_cb=step_act_cb,
                          local_search_params={"num_sims": 100,
                                               "max_depth": 20,
                                               "discount_factor": 0.95,
                                               "exploration_const": 100,
                                               "show_progress": True})
