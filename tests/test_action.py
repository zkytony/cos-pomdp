from cospomdp.domain.action import *


def test_action():
    assert MoveAhead.delta == Move2D.FORWARD
    assert RotateLeft.delta == Move2D.LEFT
    assert RotateRight.delta == Move2D.RIGHT
    assert MoveAhead == Move2D(MoveAhead.name, MoveAhead.delta)
    assert str(MoveAhead) == MoveAhead.name
    assert str(Done()) == Done().name
