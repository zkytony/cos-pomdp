from pomdp_py import SimpleAction


class Action(SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class Decision(Action):
    """A Decision is a High-level action;
    can be thought of as an option, but not really.
    Because a decision can be converted into a POMDP,
    which has a different interpretation than an option."""
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return "Decis(%s)" % self.name

    def __str__(self):
        return self.name

class Move(Action):
    """
    name is the name of the move, e.g. MoveAhead (in Ai2Thor)
    delta is the underlying change in robot pose ( (forward, h_angle, v_angle) in Ai2Thor)
    """
    def __init__(self, name, delta):
        self.name = name
        self.delta = delta
        super().__init__(name)

class Interact(Action):
    """Interacts with an object (objid) with interaction type (name),
    with parameters"""
    def __init__(self, name, objid, params):
        self.name = name
        self.objid = objid
        self.params = params
        super().__init__("{}-{}-{}".format(name, objid, params))
    def __str__(self):
        return f"{self.name}({self.objid})"

class Done(Action):
    def __init__(self):
        super().__init__("done")

class MoveDecision(Decision):
    """Move Decision moves the robot to a destination,
    specified by a 2D robot pose (x,y,th), where th is the
    yaw of the robot base."""
    def __init__(self, dest):
        self.dest = dest

class DoneDecision(Decision):
    def __init__(self):
        super().__init__("DONE")
