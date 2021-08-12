from pomdp_py import SimpleAction


class Action(SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class Move(Action):
    """
    name is the name of the move, e.g. MoveAhead (in Ai2Thor)
    delta is the underlying change in robot pose ( (forward, h_angle, v_angle) in Ai2Thor);
    or it could be (forward, angle) for 2D
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

class SearchDecision(Decision):
    def __init__(self):
        super().__init__("SEARCH")
