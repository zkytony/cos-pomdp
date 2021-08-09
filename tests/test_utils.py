from cosp.utils.math import euclidean_dist

# Hard coded correlation
CORR_MATRIX = {
    ("Apple", "CounterTop"): 0.7,
    ("Apple", "Bread"): 0.8,
    ("Bread", "CounterTop"): 0.7,
}
for k in list(CORR_MATRIX.keys()):
    CORR_MATRIX[tuple(reversed(k))] = CORR_MATRIX[k]

def corr_func(target_pos, object_pos,
              target_class, objclass):
    """
    Returns a float value to essentially mean
    Pr(Si = object_pos | Starget = target_pos)
    """
    # This is a rather arbitrary function for now.
    corr = CORR_MATRIX[(target_class, objclass)]
    distance = euclidean_dist(target_pos, object_pos)
    if corr > 0:
        return (1.0 - corr) / (distance + 0.2)
    else:
        # 7.5 is a distance threshol
        return (1.0 - abs(corr)) / (-distance - 7.5)
