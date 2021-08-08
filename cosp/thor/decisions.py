from ..framework import Decision

class MoveDecision(Decision):
    def __init__(self, dest):
        super().__init__("move-to-{}".format(dest))
        self.dest = dest

class SearchDecision(Decision):
    def __init__(self):
        super().__init__("search")

class DoneDecision(Decision):
    def __init__(self):
        super().__init__("done")
