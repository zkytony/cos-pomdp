from cospomdp.models.search_region import SearchRegion2D
import numpy as np

def test_sr():
    locations = np.unique(np.random.randint(0, 5, size=(1000, 2)), axis=0)
    sr = SearchRegion2D(locations)
    assert sr.width == 5
    assert sr.length == 5
    assert sr.dim == (5,5)
