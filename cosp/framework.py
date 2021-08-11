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

class Visualizer:
    def visualize(self, task_env, agent):
        raise NotImplementedError
