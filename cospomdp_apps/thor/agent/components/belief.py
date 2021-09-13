import pomdp_py
import random
from .state import ObjectState3D
from cospomdp_apps.thor.common import Height
from cospomdp_apps.basic.belief import initialize_target_belief_2d, update_target_belief_2d
from cospomdp.utils.math import roundany

class TargetBelief3D(pomdp_py.GenerativeDistribution):
    """
    Height range: The range of height the objects could be
        within the grid map's notion of z coordinates.

    Precisely, here is the difference:

        In GridMap, +x is right, +y is up. angle on the xy-plane rotates from +x to +y
            +z is, by right and rule, therefore up.
            angles on the yz-plane rotates from +y to +z
            angles on the xz-plane rotates from +x to +z

        In ai2thor, +x is right, +z is up. angle on the xz-plane rotates from +z to +x
            +y is, by left hand rule, up.  (Unity uses left-hand rule);
            angles on the zy-plane rotates from +y to +z,
            angles on the zx-plane rotates from +y to +x (left-hand-rule)
    """
    def __init__(self, target, loc_belief, height_belief, robot_height, grid_size):
        self.loc_belief = loc_belief
        self.height_belief = height_belief
        self.target_id, self.target_class = target
        self._robot_height = robot_height
        self._grid_size = grid_size

    @property
    def loc_stargets(self):
        return self.loc_belief.histogram.keys()

    def __iter__(self):
        """Will iterate over the object location states"""
        return iter(self.loc_belief)

    def __getitem__(self, starget):
        if isinstance(starget, ObjectState3D):
            starget2d = starget.to_2d
            thor_htarget = starget.height * self.grid_size
            thor_hrobot = self._robot_height*self._grid_size
            hstr = Height.to_str(thor_hrobot, thor_htarget)
            return self.loc_belief[starget2d] * self.height_belief[hstr]
        else:
            assert hasattr(starget, "loc")
            return self.loc_belief[starget]

    def random(self, rnd=random):
        loc = self.loc_belief.random(rnd=rnd).loc
        height_str = self.height_belief.random(rnd=rnd)
        thor_hrobot = self._robot_height*self._grid_size
        height = roundany(Height.to_val(thor_hrobot, height_str), self._grid_size)
        return ObjectState3D(self.target_id,
                             self.target_class,
                             loc, height)

    def mpe(self):
        loc = self.loc_belief.mpe().loc
        height_str = self.height_belief.mpe()
        thor_hrobot = self._robot_height*self._grid_size
        height = roundany(Height.to_val(thor_hrobot, height_str), self._grid_size)
        return ObjectState3D(self.target_id,
                             self.target_class,
                             loc, height)

    def get_histogram(self):
        return self.loc_belief.get_histogram()


def initialize_target_belief_3d(target, search_region,
                                belief_type, prior,
                                init_robot_state, **binit_args):
    prior_loc, prior_height = prior

    # Initialize the same way as 2d, and keep a separate belief for height
    target_loc_belief = initialize_target_belief_2d(target, search_region,
                                                    belief_type, prior_loc)
    # belief about target height
    target_height_belief = prior_height

    return TargetBelief3D(target, target_loc_belief, target_height_belief,
                          init_robot_state.height, binit_args["grid_size"])


def update_target_belief_3d(current_btarget,
                            next_srobot,
                            observation,
                            observation_model,
                            belief_type,
                            bu_args={}):
    assert isinstance(current_btarget, TargetBelief3D)

    # First update target location belief
    current_btarget_loc = current_btarget.loc_belief
    next_btarget_loc = update_target_belief_2d(current_btarget_loc,
                                               next_srobot,
                                               observation,
                                               observation_model,
                                               belief_type,
                                               bu_args=bu_args)

    # then, update belief about height
    # This is simple - if the robot's pitch
    # currently is > 0 then the robot is looking up,
    # otherwise, it is looking down. Depending on
    # whether we received an observation, we update
    # the belief accordingly
    current_btarget_height = current_btarget.height_belief
    next_btarget_height_hist = dict(current_btarget_height.histogram)
    if next_srobot.pitch > 0:
        # Looking up
        if observation.z(current_btarget.target_id).loc is None:
            next_btarget_height_hist[Height.ABOVE] *= 0.5
        else:
            next_btarget_height_hist[Height.ABOVE] *= 10.0

    elif next_srobot.pitch == 0:
        if observation.z(current_btarget.target_id).loc is None:
            next_btarget_height_hist[Height.SAME] *= 0.5
        else:
            next_btarget_height_hist[Height.SAME] *= 10.0

    else:
        if observation.z(current_btarget.target_id).loc is None:
            next_btarget_height_hist[Height.BELOW] *= 0.5
        else:
            next_btarget_height_hist[Height.BELOW] *= 10.0

    return TargetBelief3D((current_btarget.target_id, current_btarget.target_class),
                          next_btarget_loc,
                          pomdp_py.Histogram(next_btarget_height_hist),
                          next_srobot.height,
                          current_btarget._grid_size)
