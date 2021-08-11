import pomdp_py

class LocDetection(pomdp_py.Observation):
    def __init__(self, objclass, loc):
        self.objclass = objclass
        self.loc = loc
    def __eq__(self, other):
        return isinstance(self, LocDetection)\
            and ((self.objclass == other.objclass)\
                 and (self.loc == other.loc))
    def __hash__(self):
        return hash((self.objclass, self.loc))

class ObjectDetection2D(LocDetection):
    def __init__(self, objclass, loc):
        assert len(loc) == 2, "2D object detection needs 2D object location"
        super().__init__(objclass, loc)


class ObjectDetection3D(LocDetection):
    def __init__(self, objclass, loc):
        assert len(loc) == 3, "3D object detection needs 3D object location"
        super().__init__(objclass, loc)
