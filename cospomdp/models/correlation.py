from ..probability import JointDist, Event, TabularDistribution
from tqdm import tqdm
import pickle

class CorrelationDist(JointDist):
    def __init__(self, corr_object, target, search_region,
                 corr_func_or_dict, corr_func_args={}):
        """
        Models Pr(Si | Starget) = Pr(corr_object_id | target_id)
        Args:
            corr_object (tuple): (ID, class) of correlated object
            target (tuple): (ID, class) of target object
            search_region (SearchRegion): where the objects can be located.
            corr_func_or_dict: Either: a function that can take in a target location,
                and an object location, and return a value, the greater, the more correlated.
                Or: a dictionary that maps (target_loc, corr_object_loc) to a float,
                by default 1e-12.


        """
        self.corr_object_id, self.corr_object_class = corr_object
        self.target_id, self.target_class = target
        self.search_region = search_region
        super().__init__([self.corr_object_id, self.target_id])

        # calculate weights
        self.dists = {}  # maps from target state to conditional distributions
        for target_loc in tqdm(search_region, total=len(search_region.locations),
                               desc="Creating Pr({} | {})".format(corr_object[1], target[1])):
            target_state = search_region.object_state(
                self.target_id, self.target_class, target_loc)
            weights = {}
            for object_loc in search_region:
                object_state = search_region.object_state(
                    self.corr_object_id, self.corr_object_class, object_loc)
                if type(corr_func_or_dict) == dict:
                    prob = corr_func_or_dict.get((target_loc, object_loc), 1e-12)
                else:
                    # it's a function
                    prob = corr_func_or_dict(target_loc, object_loc,
                                             self.target_id, self.corr_object_id,
                                             **corr_func_args)
                weights[Event({self.corr_object_id: object_state})] = prob
            self.dists[target_state] =\
                TabularDistribution([self.corr_object_id], weights, normalize=True)

    def save(self, savepath):
        with open(savepath, "wb") as f:
            pickle.dump(self)

    def load(self, loadpath):
        with open(loadpath, "rb") as f:
            return pickle.load(f)

    def marginal(self, variables, evidence):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence, i.e. evidence (if supplied);

        NOTE: Only supports variables = [corr_object_id]
        with evidence being a specific target state

        variables (array-like);
        evidence (dict) mapping from variable name to value"""
        assert variables == [self.corr_object_id],\
            "CorrelationDist can only be used to infer distribution"\
            "over the correlated object's state"
        assert self.target_id in evidence\
            and evidence[self.target_id].id == self.target_id,\
            "When inferring Pr(Si | Starget), you must provide a value for Starget"\
            "i.e. set evidence = <some target state>"
        target_state = evidence[self.target_id]
        if target_state not in self.dists:
            raise ValueError("Unexpected value for target state in evidence: {}".format(target_state))
        return self.dists[target_state]

    def valrange(self, var):
        if var != self.target_id and var != self.corr_object_id:
            raise ValueError("Unable to return value range for {} because it is not modeled"\
                             .format(var))
        # For either object, the value range is the search region.
        if var == self.target_id:
            cls = self.target_class
        else:
            cls = self.corr_object_class
        return [self.search_region.object_state(var, cls, loc)
                for loc in self.search_region]
