import pomdp_py

STEP_SIZE=1
class Move2D(pomdp_py.SimpleAction):

    # Assumes (0,0) is at bottom-left, 0 degree is +x, counterclockwise.
    FORWARD = (STEP_SIZE, 0)
    LEFT_45 = (0, 45.0)
    LEFT_90 = (0, 90.0)
    RIGHT_45 = (0, -45.0)
    RIGHT_90 = (0, -90.0)
    LEFT = LEFT_45
    RIGHT = RIGHT_45

    def __init__(self, name, delta):
        """
        delta: (forward, angle)
        """
        self.name = name
        self.delta = delta
        super().__init__(name)

    def __repr__(self):
        return str(self)

MoveAhead = Move2D("MoveAhead", Move2D.FORWARD)
RotateLeft = Move2D("RotateLeft", Move2D.LEFT)
RotateRight = Move2D("RotateRight", Move2D.RIGHT)
ALL_MOVES_2D = {MoveAhead, RotateLeft, RotateRight}

class Done(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("done")

    def __repr__(self):
        return str(self)
