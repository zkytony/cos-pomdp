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

MoveAhead = Move2D("MoveAhead", Move2D.FORWARD)
MoveLeft = Move2D("MoveLeft", Move2D.LEFT)
MoveRight = Move2D("MoveRight", Move2D.RIGHT)
ALL_MOVES_2D = [MoveAhead, MoveLeft, MoveRight]

class Done(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("done")
