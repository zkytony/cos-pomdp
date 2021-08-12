import random
from pomdp_py.utils import typ
from pomdp_py import Observation, ObservationModel, OOObservation, Gaussian
from .fansensor import FanSensor
from .frustum_camera import FrustumCamera
from ..utils.math import roundany

class LocDetection(Observation):
    """
    Note for ai2thor domain: The POMDP model always uses the GridMap,
       i.e. 0-based coordinate indices, for locations. The underlying
       state space and observation space are discretized.
    """
    def __init__(self, objclass, loc):
        self.objclass = objclass
        self.loc = loc
    def __eq__(self, other):
        return isinstance(self, LocDetection)\
            and ((self.objclass == other.objclass)\
                 and (self.loc == other.loc))
    def __hash__(self):
        return hash((self.objclass, self.loc))
    def __str__(self):
        return typ.blue("o({}, {})".format(self.objclass, self.loc))
    def __repr__(self):
        return "loc({}, {})".format(self.objclass, self.loc)

class ObjectDetection2D(LocDetection):
    def __init__(self, objclass, loc):
        assert loc is None or len(loc) == 2, "2D object detection needs 2D object location"
        super().__init__(objclass, loc)

class Voxel(LocDetection):
    """3D object observation"""
    FREE = "free"
    OTHER = "other"
    UNKNOWN = "unknown"
    def __init__(self, label, loc):
        """
        label (objid or FREE or UNKNOWN)
        loc (x, y, z) 3D location
        """
        self.label = label
        super().__init__(label, loc)


class JointObservation(Observation):
    def __init__(self, detections):
        """
        detections: a tuple of object detections; each is a LocDetection.
            We assume that detections has been sorted in some way so
            there is consistency when comparing two JointObservations.
        """
        self.detections = detections
        self._hashcode = hash(self.detections)

    def __eq__(self, other):
        return isinstance(other, JointObservation)\
            and self.detections == other.detections

    def __hash__(self):
        return self._hashcode

    def __repr__(self):
        obzstr = ["{}:{}".format(o.objclass, o.loc)
                  for o in self.detections]
        return "Obz({})".format(obzstr)

    def __str__(self):
        obzstr = ["{}:{}".format(o.objclass, o.loc)
                  for o in self.detections]
        return typ.blue("Obz({})".format(obzstr))


### Observation models
class CorrObservationModel(ObservationModel):
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
        if self.objclass == self.target_class:
            # Only the detection model matters, if both classes are the same
            return self.detection_model.probability(zi, starget, srobot)

        dist_si = self._cond_dists[starget]  # Pr(Si | S_target = starget)
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



def _round(round_to, loc_cont):
    if round_to == "int":
        return tuple(map(lambda x: int(round(x)), loc_cont))
    elif type(round_to) == float:
        return tuple(map(lambda x: roundany(x, round_to),
                         loc_cont))
    else:
        return loc_cont

class DetectionModel:
    """Interface for Pr(zi | si, srobot'); Domain-specific"""
    def __init__(self, objclass, round_to="int"):
        self.objclass = objclass
        self._round_to = round_to

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

class DetectionModelFull:
    """Interface for Pr(zi | s1, ..., sn, srobot') """
    def __init__(self, objclass, round_to="int"):
        self.objclass = objclass
        self._round_to = round_to

    def probability(self, zi, s, a=None):
        """
        a (optional): action taken
        """
        raise NotImplementedError

    def sample(self, s, a=None):
        raise NotImplementedError



class FanModelYoonseon(DetectionModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')
    Yoonseon's model proposed in the OOPOMDP paper;

    Pros: parameterization is simplistic;
          simulates both false positive and false negatives.

    Cons: false positive assumption is unrealistic;
          False positive and false negative rates are the same;
          parameter values is too harsh and don't have good semantics
          (e.g. epsilon=0.9 is a pretty bad sensor already).
    """
    def __init__(self, objclass, fan_params,
                 quality_params, round_to="int"):
        """
        objclass: the class detected by this model
        fan_params: initialization params for FanSensor
        quality_params: a (sigma, epsilon) tuple
            See the definition of the 2D MOS sensor model in
            the OO-POMDP paper.
        round_to: Round the sampled observation locations to,
            either a float, 'int', or None
        """
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params
        super().__init__(objclass, round_to)

    def _compute_params(self, object_in_sensing_region, epsilon):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = epsilon
            beta = (1.0 - epsilon) / 2.0
            gamma = (1.0 - epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - epsilon) / 2.0
            beta = (1.0 - epsilon) / 2.0
            gamma = epsilon
        return alpha, beta, gamma

    def probability(self, zi, si, srobot, a=None):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        sigma, epsilon = self.params
        alpha, beta, gamma = self._compute_params(
            self.sensor.in_range(si["loc"], srobot["pose"]), epsilon)

        # Requires Python >= 3.6
        prob = 0.0
        # Event A:
        # object in sensing region and observation comes from object i
        if zi.loc is None:
            # Even though event A occurred, the observation is NULL.
            # This has 0.0 probability.
            prob += 0.0 * alpha
        else:
            gaussian = Gaussian(list(si["loc"]),
                                [[sigma**2, 0],
                                 [0, sigma**2]])
            prob += gaussian[zi.loc] * alpha

        # Event B
        prob += (1.0 / self.sensor.sensor_region_size) * beta

        # Event C
        pr_c = 1.0 if zi.loc is None else 0.0  # indicator zi == NULL
        prob += pr_c * gamma
        return prob

    def sample(self, si, srobot, a=None, return_event=False):
        sigma, epsilon = self.params
        alpha, beta, gamma = self._compute_params(
            self.sensor.in_range(si["loc"], srobot["pose"]), epsilon)
        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        if event_occured == "A":
            gaussian = Gaussian(list(si["loc"]),
                                [[sigma**2, 0],
                                 [0, sigma**2]])
            # Needs to discretize otherwise MCTS tree cannot handle this.
            loc = _round(self._round_to, gaussian.random())

        elif event_occured == "B":
            # Sample from field of view
            loc_cont = self.sensor.uniform_sample_sensor_region(srobot["pose"])
            loc = _round(self._round_to, loc_cont)
        else:  # event == C
            loc = None
        zi = ObjectDetection2D(si.objclass, loc)
        if return_event:
            return zi, event_occured
        else:
            return zi


class FanModelNoFP(DetectionModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')

    Model without involving false positives

    Pros: semantic parameter;
    Cons: no false positives modeled
    """
    def __init__(self, objclass, fan_params, quality_params, round_to="int"):
        """
        objclass: the class detected by this model
        fan_params: initialization params for FanSensor
        quality_params: (detection probability, sigma)
        round_to: Round the sampled observation locations to,
            either a float, 'int', or None
        """
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params  # calling it self.params to have consistent interface
        super().__init__(objclass, round_to)

    @property
    def detection_prob(self):
        return self.params[0]

    @property
    def sigma(self):
        return self.params[1]

    def probability(self, zi, si, srobot, a=None):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        in_range = self.sensor.in_range(si["loc"], srobot["pose"])
        if in_range:
            if zi.loc is None:
                # false negative
                return 1.0 - self.detection_prob
            else:
                # True positive; gaussian centered at object loc
                gaussian = Gaussian(list(si["loc"]),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                return self.detection_prob * gaussian[zi.loc]
        else:
            if zi.loc is None:
                # True negative; we are not modeling false positives
                return 1.0
            else:
                return 0.0


    def sample(self, si, srobot, a=None, return_event=False):
        in_range = self.sensor.in_range(si["loc"], srobot["pose"])
        if in_range:
            if random.uniform(0,1) <= self.detection_prob:
                # sample according to gaussian
                gaussian = Gaussian(list(si["loc"]),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                loc = _round(self._round_to, gaussian.random())
                zi = ObjectDetection2D(si.objclass, loc)
                event = "detected"

            else:
                zi = ObjectDetection2D(si.objclass, None)
                event = "missed"
        else:
            zi = ObjectDetection2D(si.objclass, None)
            event = "out_of_range"
        if return_event:
            return zi, event
        else:
            return zi


class FrustumModelFull(DetectionModelFull):
    """
    Pr(zi | s1, ..., sn, srobot');
    Low-level Frustum Camera Model with full state;
    considers occlusion. Currently uses mos3d model.

    quality_params: Parameters in the MOS3D observation model.
    """
    def __init__(self, objclass, camera_params, quality_params, log=False):
        self.sensor = FrustumCamera(**camera_params)
        self.alpha, self.beta, self.gamma = quality_params
        self.log = log
        super().__init__(objclass)

    def probability(self, zi, s, a=None):
        """
        zi: voxel
        s (JointState3D)
        a (optional): action taken
        """
        if zi.label == Voxel.UNKNOWN:
            return self.gamma  # not in FOV
        elif zi.label == self.objclass:
            return self.alpha  # i
        else:
            return self.beta   # not i

    def sample(self, s, a=None):
        si = s.object_states[self.objclass]
        srobot = s.robot_state["pose"]
        voxel = Voxel(si["loc"], Voxel.UNKNOWN)
        if self.sensor.observable(self.objclass, s):
            if FrustumCamera.sensor_functioning(self.alpha, self.beta, self.log):
                voxel.label = self.objclass
            else:
                voxel.label = Voxel.OTHER
            return voxel
        else:
            # Object not in FOV. The label is UNKNOWN.
            return voxel


class JointObservationModel(ObservationModel):
    def __init__(self, target_class, zi_models):
        """
        models: maps from objclass to ObservationModel;
        each objclass is a detectable class
        """
        self.zi_models = zi_models
        self._classes = list(sorted(self.zi_models.keys()))
        self.target_class = target_class

    def sample(self, next_state, action):
        return JointObservation(tuple(self.zi_models[objclass].sample(next_state, action)
                                      for objclass in self._classes))


    def probability(self, observation, next_state, action):
        pr_total = 1.0
        for zi in observation.detections:
            pr_zi = self.zi_models[zi.objclass].probability(zi, next_state)
            pr_total *= pr_zi
        return pr_total
