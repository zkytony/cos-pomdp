from test_cosagent_complete import _test_complete_search


if __name__ == "__main__":
    _test_complete_search("Bowl", "Book",
                          scene="FloorPlan1",
                          num_sims=100,
                          prior='informed')
