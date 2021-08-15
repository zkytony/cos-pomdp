from ..domain.state import ObjectState2D

class SearchRegion:
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

    def object_state(self, objid, objclass, loc):
        raise NotImplementedError

class SearchRegion2D(SearchRegion):
    def object_state(self, objid, objclass, loc):
        return ObjectState2D(objid, objclass, loc)
