# Utility functions for thor & its experimentation

from ai2thor.controller import Controller
import matplotlib
import matplotlib.pyplot as plt
import thortils


def spl_ratio(li, pi, Si):
    """spl ratio for a single trial.
    li, pi, Si stands for shortest_path_length, actual_path_length, success for trial i.
    """
    pl_ratio = max(pi, li) if max(pi, li) > 0 else 1.0
    return float(Si) * li / pl_ratio


def compute_spl(episode_results):
    """
    Reference: https://arxiv.org/pdf/1807.06757.pdf

    Args:
        episode_results (list) List of tuples
            (shortest_path_length, actual_path_length, success),
             as required by the formula. `actual_path_length` and
            `shortest_path_length` are floats; success is boolean.
    Return:
        float: the SPL metric
    """
    # li, pi, Si stands for shortest_path_length, actual_path_length, success for trial i
    return sum(spl_ratio(li, pi, Si)
               for li, pi, Si in episode_results) / len(episode_results)


def plot_path(path, controller, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    all_positions = controller.step(action="GetReachablePositions")\
                              .metadata["actionReturn"]
    x = [pos["x"] for pos in all_positions]
    z = [pos["z"] for pos in all_positions]
    plt.scatter(x, z, s=3)

    xpath = [pos["x"] for pos in path]
    zpath = [pos["z"] for pos in path]
    plt.plot(xpath, zpath, "o-", linewidth=3)
