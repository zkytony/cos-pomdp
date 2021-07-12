import pomdp_py
import json


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

    def to_json(self):
        """Returns json string of this action"""
        return json.dumps({"name": self.name, "params": self.params})

    @classmethod
    def from_json(self, string):
        obj = json.loads(string)
        return ThorAction(obj["name"], obj["params"])
