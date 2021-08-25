import thortils

from cospomdp.utils.corr_funcs import around
from cospomdp_apps.thor.common import TaskArgs, make_config
from cospomdp_apps.thor.agent import ThorObjectSearchCosAgent
from cospomdp_apps.thor.trial import ThorObjectSearchTrial
from test_cosagent_basic_search import _test_basic_search

if __name__ == "__main__":
    _test_basic_search('Bowl', 'Book', prior='informed')
