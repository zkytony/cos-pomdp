from pomdp_py import GenerativeDistribution, Histogram


class LocBelief2D(Histogram):
    """Intended for high-level belief over an object's locations."""
    def __init__(self, objclass, search_region, prior="uniform"):
        """
        Args:
            objclass: Class of object belief is formed
            search_region: locations the object could be in
            prior: maps from location to a probability, or "uniform"
        """
        hist = {}
        for loc in search_region:
            if prior == "uniform":
                hist[loc] = 1.0
            else:
                hist[loc] = prior[loc]
        super().__init__(hist)
