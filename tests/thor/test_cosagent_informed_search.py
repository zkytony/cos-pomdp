import thortils

from pomdp_py.utils import TreeDebugger
from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from test_cosagent_basic_search import _test_basic_search

def step_act_cb(task_env, agent):
    dd = TreeDebugger(agent.cos_agent.tree)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    _test_basic_search('Bowl', 'Book', prior='informed',
                       step_act_cb=step_act_cb,
                       num_sims=200,
                       show_progress=True,
                       exploration_const=100)
