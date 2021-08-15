from cospomdp.domain.action import *


def test_action():
    assert MoveAhead.delta == Move2D.FORWARD
    assert MoveLeft.delta == Move2D.LEFT
    assert MoveRight.delta == Move2D.RIGHT
    assert MoveAhead == Move2D(MoveAhead.name, MoveAhead.delta)
    assert str(MoveAhead) == MoveAhead.name
    assert str(Done()) == Done().name
