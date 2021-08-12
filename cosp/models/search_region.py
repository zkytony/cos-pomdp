from .state import ObjectState2D, ObjectState3D

class SearchRegion:
    # DOMAIN-SPECIFIC
    """domain-specific / abstraction-specific host of a set of locations. All that
    it needs to support is enumerability (which could technically be implemented
    by sampling)
    """
    def __init__(self, locations):
        self.locations = locations

    def __iter__(self):
        return iter(self.locations)

    def __contains__(self, item):
        return item in self.locations

    def object_state(self, objclass, loc):
        raise NotImplementedError

class SearchRegion2D(SearchRegion):
    def object_state(self, objclass, loc):
        return ObjectState2D(objclass, dict(loc=loc))

class SearchRegion3D(SearchRegion):
    def object_state(self, objclass, loc):
        return ObjectState3D(objclass, dict(loc=loc))
