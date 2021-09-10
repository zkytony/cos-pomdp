# Run all tests here in this folder; This is convenient
# to verify that all code still runs.
from test_cosagent_basic_search import _test_basic_search
from test_cosagent_basic_informed_search import _test_basic_informed_search
from test_analyze_search_tree import _test_analyze_cosagent_basic_search_tree
from test_mjolnir_agent import _test_mjolnir_agent
from test_optimal_object_search_agent import _test_optimal_search_agent
from test_visualizer import _test_visualizer
from test_sample_topo_maps import _test_topo_map_sampling_multiple

def main():
    _test_visualizer(sleep=2)
    _test_optimal_search_agent({"FloorPlan1": ["Vase", "Book"]})
    _test_mjolnir_agent(max_steps=10)
    _test_basic_search('Bowl', 'Book',
                       num_sims=50,
                       max_depth=15,
                       max_steps=10)
    _test_basic_informed_search('Bowl', 'Book',
                                num_sims=50,
                                max_depth=15,
                                max_steps=10,
                                block=False)
    _test_analyze_cosagent_basic_search_tree(
        'Bowl', 'Book', num_sims=100, num_trajs=10, max_steps=1,
        show_progress=True, max_depth=30, exploration_const=0,
        interactive=False)
    _test_topo_map_sampling_multiple()

if __name__ == "__main__":
    main()
