import pomdp_py

class ThorAction(pomdp_py.Action):
    def __init__(self, name, params={}):
        self.name = name
        self.params = params
    def __eq__(self, other):
        if isinstance(other, ThorAction):
            return self.name == other.name
        return False
    def __hash__(self):
        return hash(self.name)
