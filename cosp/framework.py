import pomdp_py

class TaskEnv:
    """
    A TaskEnv maintains the state of a task instance and tracks the progress.
    """
    def execute(self, action):
        pass

    def done(self):
        pass

    def compute_results(self):
        """
        Returns a list of sciex.Result objects that
        correspond to results for this task.
        """
        pass


class Agent:
    """Agent acting in the world."""
    def act():
        pass

    def update(self, action, observation):
        pass


class Method:
    """Decision Making Method that produces actions."""
    def act(self, agent):
        pass


class Action(pomdp_py.SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class Decision(Action):
    """A Decision is a High-level action;
    can be thought of as an option, but not really.
    Because a decision can be converted into a POMDP,
    which has a different interpretation than an option."""
    def __init__(self, name):
        super().__init__(name)
