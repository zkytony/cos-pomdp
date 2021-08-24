import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR
import scipy.stats as stats
import math
import scipy.stats
import pandas as pd
from thortils.utils.math import *

def indicator(cond, epsilon=0.0):
    return 1.0 - epsilon if cond else epsilon

def normalize_log_prob(likelihoods):
    """Given an np.ndarray of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                        (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return normalized

def normalize(ss):
    """
    ss (dict or array-like):
        If dict, maps from key to float; If array-like, all floats.
    Returns:
        a new object (of the same kind as input) with normalized entries
    """
    if type(ss) == dict:
        total = sum(ss.values())
        return {k:ss[k]/total for k in ss}
    else:
        total = sum(ss)
        return type(ss)(ss[i]/total for i in range(len(ss)))

def uniform(size, ranges):
    return tuple(random.randrange(ranges[i][0], ranges[i][1])
                 for i in range(size))

def roundany(x, base):
    """
    rounds the number x (integer or float) to
    the closest number that increments by `base`.
    """
    return base * round(x / base)

def fround(round_to, loc_cont):
    """My own 'float'-rounding (hence fround) method.

    round_to can be 'int', 'int-' or any float,
    and will output a value that is the result of
    rounding `loc_cont` to integer, or the `round_to` float;
    (latter uses roundany).

    If 'int-', will floor `loc_cont`
    """
    if round_to == "int":
        return tuple(map(lambda x: int(round(x)), loc_cont))
    elif round_to == "int-":
        return tuple(map(lambda x: int(math.floor(x)), loc_cont))
    elif type(round_to) == float:
        return tuple(map(lambda x: roundany(x, round_to),
                         loc_cont))
    else:
        return loc_cont

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True

## Geometry
def intersect(seg1, seg2):
    """seg1 and seg2 are two line segments each represented by
    the end points (x,y). The algorithm comes from
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect"""
    # Represent each segment (p,p+r) where r = vector of the line segment
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]

    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s != 0:
        t = np.cross(q-p, s) / r_cross_s
        u = np.cross(q-p, r) / r_cross_s
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Two lines meet at point
            return True
        else:
            # Are not parallel and not intersecting
            return False
    else:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
            else:
                # colinear and disjoint
                return False
        else:
            # two lines are parallel and non intersecting
            return False

def overlap(seg1, seg2):
    """returns true if line segments seg1 and 2 are
    colinear and overlapping"""
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]

    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s == 0:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
    return False


## Statistics
# confidence interval
def ci_normal(series, confidence_interval=0.95):
    series = np.asarray(series)
    tscore = stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)
    y_error = stats.sem(series)
    ci = y_error * tscore
    return ci

def mean_ci_normal(series, confidence_interval=0.95):
    ### CODE BY CLEMENT at LIS ###
    """Confidence interval for normal distribution with unknown mean and variance.

    Interpretation:
    An easy way to remember the relationship between a 95%
    confidence interval and a p-value of 0.05 is to think of the confidence interval
    as arms that "embrace" values that are consistent with the data. If the null
    value is "embraced", then it is certainly not rejected, i.e. the p-value must be
    greater than 0.05 (not statistically significant) if the null value is within
    the interval. However, if the 95% CI excludes the null value, then the null
    hypothesis has been rejected, and the p-value must be < 0.05.
    """
    series = np.asarray(series)
    # this is the "percentage point function" which is the inverse of a cdf
    # divide by 2 because we are making a two-tailed claim
    tscore = scipy.stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)

    y_mean = np.mean(series)
    y_error = scipy.stats.sem(series)

    half_width = y_error * tscore
    return y_mean, half_width

def perplexity(p, base=2):
    """Measures how well a probability model predicts a sample (from the same
    distribution). On Wikipedia, given a probability distribution p(x), this
    distribution has a notion of "perplexity" defined as:

        2 ^ (- sum_x p(x) * log_2 p(x))

    The exponent is also called the "entropy" of the distribution.
    These two words, "perplexity" amd "entropy" are designed
    to be vague by Shannon himself (according to Tomas L-Perez).

    The interpretation of entropy is "level of information",
    or "level of uncertainty" inherent in the distribution.
    Higher entropy indicates higher "randomness" of the
    distribution. You can actually observe the scale of

    "p(x) * log_2 p(x)" in a graph tool (e.g. Desmos).

    I made the probability distribution of x to be P(x) = 1/2 * (sin(x)+1).
    (It's not normalized, but it's still a distribution.)
    You observe that when P(x) approaches 1, the value of p(x) * log_2 p(x)
    approaches zero. The farther P(x) is away from 1, the more negative
    is the value of p(x) * log_2 p(x). You can understand it has,
    if p(x) * log_2 p(x) --> 0, then the value of x here does not
    contribute much to the uncertainty since P(x) --> 1.
    Thus, if we sum up these quantities

    "sum_x p(x) * log_2 p(x)"
    we can understand it as: the lower this sum, the more uncertainty there is.
    If you take the negation, then it becomes the higher this quantity,
    the more the uncertainty, which is the definition of entropy.

    Why log base 2? It's "for convenience and intuition" (by Shannon) In fact
    you can do other bases, like 10, or e.

    Notice how KL-divergence is defined as:

    "-sum_x p(x) * log_2 ( p(x) / q(x) )"

    The only difference is there's another distribution q(x). It measures
    how "different" two distributions are. KL divergence of 0 means identical.

    How do you use perplexity to compare two distributions? You compute the
    perplexity of both.

    Also refer to: https://www.cs.rochester.edu/u/james/CSC248/Lec6.pdf

    Parameters:
        p: A sequence of probabilities
    """
    H = scipy.stats.entropy(p, base=base)
    return base**H

def kl_divergence(p, q, base=2):
    return scipy.stats.entropy(p, q, base=base)

def normal_pdf_2d(point, variance, domain, normalize=True):
    """
    returns a dictionary that maps a value in domain to a probability
    such that the probability distribution is a 2d gaussian with mean
    at the given point and given variance.
    """
    dist = {}
    total_prob = 0.0
    for val in domain:
        prob = scipy.stats.multivariate_normal.pdf(np.array(val),
                                                   np.array(point),
                                                   np.array(variance))
        dist[val] = prob
        total_prob += prob
    if normalize:
        for val in dist:
            dist[val] /= total_prob
    return dist

def dists_to_seqs(dists, avoid_zero=True):
    """Convert dictionary distributions to seqs (lists) such
    that the elements at the same index in the seqs correspond
    to the same key in the dictionary"""
    seqs = [[] for i in range(len(dists))]
    vals = []
    d0 = dists[0]
    for val in d0:
        for i, di in enumerate(dists):
            if val not in di:
                raise ValueError("Value %s is in one distribution but not another" % (str(val)))
            if avoid_zero:
                prob = max(1e-12, di[val])
            else:
                prob = di[val]
            seqs[i].append(prob)
        vals.append(val)
    return seqs, vals

def compute_mean_ci(results):
    """Given `results`, a dictionary mapping "result_type" to a list of values
    for this result_type, compute the mean and confidence intervals for each
    of the result type. It will add a __summary__ key to the given dictionary.x"""
    results["__summary__"] = {}
    for restype in results:
        if restype.startswith("__"):
            continue
        mean, ci = mean_ci_normal(results[restype], confidence_interval=0.95)
        results["__summary__"][restype] = {
            "mean": mean,
            "ci-95": ci,
            "size": len(results[restype])
        }
    return results

def entropy(p, base=2):
    """
    Parameters:
        p: A sequence of probabilities
    """
    return scipy.stats.entropy(p, base=base)
