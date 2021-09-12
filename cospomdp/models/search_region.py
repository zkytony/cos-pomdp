from ..domain.state import ObjectState, ObjectState3D

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
    def __init__(self, locations):
        """
        locations should be 2D tuples of integers.
        """
        super().__init__(locations)
        self._w = max(locations, key=lambda l: l[0])[0] - min(locations, key=lambda l: l[0])[0] + 1
        self._l = max(locations, key=lambda l: l[1])[1] - min(locations, key=lambda l: l[1])[1] + 1
        self._obstacles = {(x,y)
                           for x in range(self._w)
                           for y in range(self._l)
                           if (x,y) not in locations}

    def object_state(self, objid, objclass, loc):
        return ObjectState(objid, objclass, loc)

    @property
    def dim(self):
        return (self._w, self._l)

    @property
    def width(self):
        return self._w

    @property
    def length(self):
        return self._l

    @property
    def obstacles(self):
        return self._obstacles


class SearchRegion3D(SearchRegion):
    def __init__(self, locations, height_range):
        """
        locations should be 2D tuples of integers.
        height_range: Height should also be with respect to in GridMap coordinates,
            but it does not need to be integers.
        """
        super().__init__(locations)
        self._w = max(locations, key=lambda l: l[0])[0] - min(locations, key=lambda l: l[0])[0] + 1
        self._l = max(locations, key=lambda l: l[1])[1] - min(locations, key=lambda l: l[1])[1] + 1
        self._obstacles = {(x,y)
                           for x in range(self._w)
                           for y in range(self._l)
                           if (x,y) not in locations}
        self._height_range = height_range

    @property
    def height_range(self):
        return self._height_range

    def object_state(self, objid, objclass, loc, height):
        return ObjectState3D(objid, objclass, loc, height)

    @property
    def dim(self):
        return (self._w, self._l)

    @property
    def width(self):
        return self._w

    @property
    def length(self):
        return self._l

    @property
    def obstacles(self):
        return self._obstacles
