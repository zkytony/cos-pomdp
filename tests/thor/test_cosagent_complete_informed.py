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
                          target_false_pos=0.05,
                          local_search_type="3d",
                          local_search_params={'pouct': {"num_sims": 200,
                                                         "max_depth": 30,
                                                         "discount_factor": 0.95,
                                                         "exploration_const": 100,
                                                         "show_progress": True}})
