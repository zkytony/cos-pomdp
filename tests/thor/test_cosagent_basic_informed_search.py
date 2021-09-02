from pomdp_py.utils import TreeDebugger
from test_cosagent_basic_search import _test_basic_search

def step_act_cb(task_env, agent, **kwargs):
    dd = TreeDebugger(agent.cos_agent.tree)
    if kwargs.get("block", True):
        import ipdb; ipdb.set_trace()
    else:
        dd.mbp

def _test_basic_informed_search(target, other, block=True, **params):
    _test_basic_search(target, other,
                       prior='informed',
                       step_act_cb=step_act_cb,
                       step_act_args={'block': block},
                       **params)


if __name__ == "__main__":
    _test_basic_informed_search('Bowl', 'Book',
                                num_sims=200,
                                show_progress=True,
                                exploration_const=100,
                                local_search_params={"num_sims": 100,
                                                     "max_depth": 20,
                                                     "discount_factor": 0.95,
                                                     "exploration_const": 100,
                                                     "show_progress": True})
