# COS-POMDP: Correlation Object Search POMDP
import pomdp_py
from ..framework import Agent, Decision
from ..utils.misc import resolve_robot_target_args

class SearchRegion:
    # DOMAIN-SPECIFIC
    """domain-specific / abstraction-specific host of a set of locations. All that
    it needs to support is enumerability (which could technically be implemented
    by sampling)
    """
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __contains__(self):
        pass


class DetectionModel:
    # DOMAIN-SPECIFIC
    """Interface for Pr(zi | si, srobot'); Domain-specific"""
    def __init__(self):
        pass

    def probability(self, zi, si, srobot, a=None):
        """
        zi: object observation
        si: object state
        srobot: robot state
        a (optional): action taken
        """
        raise NotImplementedError

    def sample(self, si, srobot, a=None):
        raise NotImplementedError


class ReducedState(pomdp_py.OOState):
    """Reduced state that only contains robot and target states.
    Both robot_state and target_state are pomdp_py.ObjectState"""
    def __init__(self, robot_id, target_id, *args):
        self.robot_id = robot_id
        self.target_id = target_id
        robot_state, target_state = resolve_robot_target_args(robot_id, target_id, *args)
        super().__init__({self.robot_id: robot_state,
                          self.target_id: target_state})

    def __str__(self):
        return\
            "{}(\n"\
            "    {},\n"\
            "    {})".format(type(self), self.robot_state, self.target_state)

    def __repr__(self):
        return str(self)

    @property
    def robot_state(self):
        return self.object_states[self.robot_id]

    @property
    def target_state(self):
        return self.object_states[self.target_id]


class Observation(pomdp_py.Observation):
    def __init__(self, object_observations):
        """
        object_observations (tuple): vector of observations of each object
        """
        self.object_observations = object_observations
        self._hash = hash(self.object_observations)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.object_observations == other.object_observations
        return False

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return str(self)

    def __str__(self):
        obzstr = ["    {}".format(o)
                  for o in self.object_observations]
        return\
            "{}(\n"\
            "    {})".format("\n".join(obzstr))

    def __iter__(self):
        return iter(self.object_observations)

    def __len__(self):
        return len(self.object_observations)


class ObjectObservation(pomdp_py.SimpleObservation):
    """Keep it simple for now"""
    def __init__(self, objclass, location):
        """
        Args:
            objclass (str): Object class
            location (object): Object location, hashable; None if not observed
        """
        self.objclass = objclass
        self.location = location
        super().__init__((objclass, location))


class ObjectObservationModel(pomdp_py.ObservationModel):
    """This is the model for Pr( zi | starget, srobot' ),
    a result of the conditional independence assumption. It is
    the observation model in COS-POMDP.

    The model is (unless the object is the target itself)

    Pr(zi | starget, srobot' ) = sum_si Pr(zi | si, srobot') * Pr(si | starget)

    The term Pr(zi | si, srobot') is a detection_model
    The term Pr(si | starget) is a given distribution, which specifies the
    correlation.
    """
    def __init__(self, objclass, target_class,
                 detection_model, corr_dist):
        """
        Args:
            objclass (str): Class for object i
            target_class (str): Target class
            corr_dist (JointDist): Distribution of Pr(Si | Starget); The interface
                of JointDist allows for this to be either Pr(Si, Starget) or Pr(Si | Starget)
                underneath the hood.

                If objclass = target_class, then corr_dist is optional.

                TODO: maybe we need to be more creative here for the joint distribution,
                because feeding in absolute coordinates won't generalize
            detection_model (DetectionModel): model for Pr(zi | si, srobot')
        """
        self.objclass = objclass
        self.target_class = target_class
        self.detection_model = detection_model

        # Compute the conditional distribution for every value of Starget
        if self.objclass != self.target_class:
            self._cond_dists = {}
            for starget in corr_dist.valrange(target_class):
                # Obtain Pr(Si | S_target = starget)
                self._cond_dists[starget] =\
                    corr_dist.marginal([self.objclass], evidence={self.target_class: starget})

    def corr_cond_dist(self, starget):
        return self._cond_dists[starget]

    def probability(self, object_observation, next_state, *args):
        # action doesn't matter here
        """
        Args:
            object_observation (ObjectObservation): observation of an object
            next_state (State): state where the observation is made
            action (Action): action that led to the next state
        """
        zi = object_observation
        starget = next_state.target_state
        srobot = next_state.robot_state
        dist_si = self._cond_dists[starget]  # Pr(Si | S_target = starget)
        if self.objclass == self.target_class:
            # Only the detection model matters, if both classes are the same
            return self.detection_model.probability(zi, starget, srobot)

        pr_total = 0.0
        for si in dist_si.valrange(self.objclass):
            pr_detection = self.detection_model.probability(zi, si, srobot)
            pr_corr = dist_si.prob({self.objclass: si})  # compute Pr(Si = si | S_target = starget)
            pr_total += pr_detection * pr_corr
        return pr_total

    def sample(self, next_state, *args):
        # action doesn't matter here
        starget = next_state.target_state
        srobot = next_state.robot_state
        if self.objclass == self.target_class:
            zi = self.detection_model.sample(starget, srobot)
        else:
            dist_si = self._cond_dists[starget]  # Pr(Si | S_target = starget)
            si = dist_si.sample()[self.objclass]
            zi = self.detection_model.sample(si, srobot)
        return zi
