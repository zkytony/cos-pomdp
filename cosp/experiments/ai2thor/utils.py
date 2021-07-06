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
