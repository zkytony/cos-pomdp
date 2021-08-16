from cospomdp.domain.observation import *


def test_observation():
    o1 = Loc2D(0, (5,5))
    o2 = Loc2D(0, (5,5))
    assert o1 == o2
    assert hash(o1) == hash(o2)

    o = CosObservation2D(None, {o1.id:o1, o2.id:o2})
    assert len(o) == 1
