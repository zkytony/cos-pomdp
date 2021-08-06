"""
Factor graph representation
of joint distribution, using pgmpy.
Custom interface
"""
import random
import numpy as np
from pgmpy.models import FactorGraph as PGMFactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools
from corrsearch.probability.dist import JointDist
from corrsearch.probability.tabular_dist import Event, TabularDistribution

class FactorGraph(JointDist):

    def __init__(self,
                 variables,
                 factors,
                 compute_joint=True):
        """
        Args:
            variables (list): List of variables
            factors (list): List of TabularDistribution representation of factors.
        """
        self.fg = PGMFactorGraph()
        self.fg.add_nodes_from(variables)
        self.variables = variables
        # Create factors using pgmpy's interface
        pgmpy_factors = []
        edges = []
        all_state_names = {}
        for factor in factors:
            # cardinality
            cardinality = [len(factor.valrange(var)) for var in factor.variables]
            # obtain state names
            state_names = {}
            for var in factor.variables:
                state_names[var] = list(sorted(factor.valrange(var)))
                                               # key=lambda si: si["loc"]))
                if var not in all_state_names:
                    all_state_names[var] = state_names[var]
                else:
                    assert all_state_names[var] == state_names[var],\
                        "Expected same value ranges for different factors,"\
                        "for variable %s" % var

            values = []
            for setting in itertools.product(*[state_names[var]
                                               for var in factor.variables]):
                setting_dict = {factor.variables[i]:setting[i]
                                for i in range(len(factor.variables))}
                prob = factor.prob(setting_dict)
                values.append(prob)

            phi = DiscreteFactor(factor.variables,
                                 cardinality,
                                 values,
                                 state_names=state_names)
            pgmpy_factors.append(phi)
            for var in factor.variables:
                edges.append((var, phi))
        self.fg.add_factors(*pgmpy_factors)
        self.fg.add_edges_from(edges)
        self.fg.check_model()
        self.bp = BeliefPropagation(self.fg)

        # For efficiency
        self._ranges = all_state_names  # maps from variable name to ranges (list)
        self.joint = None
        if compute_joint:
            self._compute_joint()

    def _compute_joint(self):
        if self.joint is None:
            print("Computing joint probability table..")
            self.joint = PGMFactorDist(self.bp.query(self.variables, show_progress=True))

    def prob(self, setting):
        """
        Args:
            setting (dict): Mapping from variable to value.
                Does not have to specify the value for every variable
        """
        self._compute_joint()
        return self.joint.prob(setting)

    def sample(self, rnd=random):
        self._compute_joint()
        return self.joint.sample(rnd=rnd)

    def marginal(self, variables, observation=None):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence (if supplied)"""
        dist = self.bp.query(variables, evidence=observation, show_progress=False)
        return PGMFactorDist(dist)

    def valrange(self, var):
        """Returns an enumerable that contains the possible values
        of the given variable var"""
        return self._ranges[var]


class PGMFactorDist(JointDist):
    def __init__(self, discrete_factor):
        """discrete_factor (DiscreteFactor)"""
        self.factor = discrete_factor
        self.variables = discrete_factor.variables

    def _idx_to_setting(self, idx):
        """Given an index `idx` that refers to an entry in the
        factor probability table, return """
        indices = np.unravel_index(idx, self.factor.values.shape)
        setting = {}
        for i in range(len(self.factor.variables)):
            var = self.factor.variables[i]
            validx = indices[i]
            val = self.valrange(var)[validx]
            setting[var] = val
        return setting

    def prob(self, setting):
        """
        Args:
            setting (dict): Mapping from variable to value.
                Does not have to specify the value for every variable
        """
        return self.factor.get_value(setting)

    def sample(self, rnd=random):
        weights = self.factor.values.flatten()
        idx = rnd.choices(np.arange(0, len(weights)),
                          weights=weights, k=1)[0]
        return self._idx_to_setting(idx)

    def marginal(self, variables, observation=None):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence (if supplied)"""
        dist = self.bp.query(variables, evidence=observation, show_progress=False)
        return PGMFactorDist(dist)

    def valrange(self, var):
        """Returns an enumerable that contains the possible values
        of the given variable var"""
        return self.factor.state_names[var]

    def to_tabular_dist(self):
        weights = {}
        for combo in itertools.product(*[self.valrange(var)
                                           for var in self.variables]):
            setting = {self.variables[i]:combo[i]
                       for i in range(len(self.variables))}
            event = Event(setting)
            weights[event] = self.prob(setting)
        return TabularDistribution(self.variables, weights)
