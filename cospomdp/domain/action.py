import pomdp_py


class Motion(pomdp_py.SimpleAction):
    """Motion moves the robot.
    The specific definition is domain-dependent"""

    def __repr__(self):
        return str(self)


class Done(pomdp_py.SimpleAction):
    """Declares the task to be over"""
    def __init__(self):
        super().__init__("done")

    def __repr__(self):
        return str(self)
