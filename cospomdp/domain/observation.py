import pomdp_py
from .state import RobotStatus, RobotState2D

class Loc2D(pomdp_py.SimpleObservation):
    """Observation of an object's 2D location"""
    NULL = None  # empty
    def __init__(self, objid, loc):
        self.objid = objid
        self.loc = loc
        super().__init__((objid, loc))
    def __str__(self):
        return f"({self.objid}, {self.loc})"
    @property
    def id(self):
        return self.objid

class CosObservation2D(pomdp_py.Observation):
    def __init__(self, robotobz, objobzs):
        """
        objobzs (dict): maps from objid to Loc2D or NULL
        """
        self._hashcode = hash(frozenset(objobzs.items()))
        if isinstance(robotobz, RobotState2D):
            robotobz = RobotObservation2D(robotobz.id,
                                          robotobz['pose'],
                                          robotobz['status'].copy())
        self._robotobz = robotobz
        self._objobzs = objobzs

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self._objobzs == other._objobzs

    def __str__(self):
        robotstr = ",".join(map(str, self._robotobz.pose))
        objzstr = ""
        for objid in self._objobzs:
            if self._objobzs[objid].loc is not None:
                objzstr += "{}{}".format(objid, self._objobzs[objid].loc)
        return "CosObservation(r:{};o:{})".format(robotstr, objzstr)

    def __repr__(self):
        return f"{self.__class__}, {self._objobzs}"

    def __len__(self):
        # Only care about object observations here
        return len(self._objobzs)

    def __iter__(self):
        # Only care about object observations here
        return iter(self._objobzs.values())

    def __getitem__(self, objid):
        # objid can be either object id or robot id.
        return self.z(objid)

    def z(self, objid):
        if objid == self._robotobz.robot_id:
            return self._robotobz
        elif objid in self._objobzs:
            return self._objobzs[objid]
        else:
            raise ValueError("Object ID {} not in observation".format(objid))

    @property
    def z_robot(self):
        return self._robotobz

class RobotObservation2D(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose, status):
        self.robot_id = robot_id
        self.pose = robot_pose
        self.status = status
        super().__init__((self.robot_id, self.pose, self.status))

    def __str__(self):
        return f"({self.robot_pose, self.status})"

    def to_state(self):
        return RobotState2D(self.robot_id, self.pose, self.status)

class Voxel(pomdp_py.SimpleObservation):
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
        self.loc = loc
        super().__init__((label, loc))

    def __str__(self):
        return f"({self.loc}, {self.label})"
