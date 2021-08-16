import pomdp_py

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
    def __init__(self, objlocs):
        """
        objlocs (dict): maps from objid to Loc2D or NULL
        """
        self._hashcode = hash(frozenset(objlocs.items()))
        self.objlocs = objlocs

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self.objlocs == other.objlocs

    def __str__(self):
        return "{ %s }" % (",".join(sorted(map(str, self.objlocs))))

    def __repr__(self):
        return f"{self.__class__}, {self.objlocs}"

    def __len__(self):
        return len(self.objlocs)

    def __iter__(self):
        return iter(self.objlocs.values())

    def __getitem__(self, objid):
        return self.objlocs[objid]

    def z(self, objid):
        return self[objid]

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
