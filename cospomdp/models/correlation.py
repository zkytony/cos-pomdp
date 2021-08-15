from ..probability import JointDist, Event, TabularDistribution

class CorrelationDist(JointDist):
    def __init__(self, objclass, target_class, search_region, corr_func):
        """
        Models Pr(Si | Starget) = Pr(objclass | target_class)
        Args:
            objclass (str): class corresponding to the state variable Si
            target_class (str): target object class
            corr_func: can take in a target location, and an object location,
                and return a value, the greater, the more correlated.
        """
        self.objclass = objclass
        self.target_class = target_class
        self.search_region = search_region
        super().__init__([objclass, target_class])

        # calculate weights
        self.dists = {}  # maps from target state to
        for target_loc in search_region:
            target_state = search_region.object_state(target_class, target_loc)
            weights = {}
            for object_loc in search_region:
                object_state = search_region.object_state(objclass, object_loc)
                prob = corr_func(target_loc, object_loc,
                                 target_class, objclass)
                weights[Event({self.objclass: object_state})] = prob
            self.dists[target_state] =\
                TabularDistribution([self.objclass], weights, normalize=True)

    def marginal(self, variables, evidence):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence, i.e. evidence (if supplied);

        NOTE: Only supports variables = [objclass]
        with evidence being a specific target state

        variables (array-like);
        evidence (dict) mapping from variable name to value"""
        assert variables == [self.objclass],\
            "CorrelationDist can only be used to infer distribution"\
            "over the correlated object's state"
        assert self.target_class in evidence\
            and evidence[self.target_class].objclass == self.target_class,\
            "When inferring Pr(Si | Starget), you must provide a value for Starget"\
            "i.e. set evidence = <some target state>"
        target_state = evidence[self.target_class]
        if target_state not in self.dists:
            raise ValueError("Unexpected value for target state in evidence: {}".format(target_state))
        return self.dists[target_state]

    def valrange(self, var):
        if var != self.target_class and var != self.objclass:
            raise ValueError("Unable to return value range for {} because it is not modeled"\
                             .format(var))
        # For either object, the value range is the search region.
        return [self.search_region.object_state(var, loc)
                for loc in self.search_region]
