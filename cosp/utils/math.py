import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR
import scipy.stats as stats
import math
import scipy.stats
import pandas as pd


def indicator(cond, epsilon=0.0):
    return 1.0 - epsilon if cond else epsilon

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def normalize_log_prob(likelihoods):
    """Given an np.ndarray of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                        (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return normalized

def uniform(size, ranges):
    return tuple(random.randrange(ranges[i][0], ranges[i][1])
                 for i in range(size))

# Math
def to_radians(th):
    return th*np.pi / 180

def to_rad(th):
    return th*np.pi / 180

def to_degrees(th):
    return th*180 / np.pi

def to_deg(th):
    return th*180 / np.pi

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

def proj(vec1, vec2, scalar=False):
    # Project vec1 onto vec2. Returns a vector in the direction of vec2.
    scale = np.dot(vec1, vec2) / np.linalg.norm(vec2)
    if scalar:
        return scale
    else:
        return vec2 * scale

def R_x(th):
    return np.array([
        1, 0, 0, 0,
        0, np.cos(th), -np.sin(th), 0,
        0, np.sin(th), np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_y(th):
    return np.array([
        np.cos(th), 0, np.sin(th), 0,
        0, 1, 0, 0,
        -np.sin(th), 0, np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_z(th):
    return np.array([
        np.cos(th), -np.sin(th), 0, 0,
        np.sin(th), np.cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_between(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Only applicable to 3D vectors!")
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vX = np.array([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    ]).reshape(3,3)
    R = I + vX + np.matmul(vX,vX) * ((1-c)/(s**2))
    return R

def R_euler(thx, thy, thz, affine=False):
    """
    Obtain the rotation matrix of Rz(thx) * Ry(thy) * Rx(thz); euler angles
    """
    R = scipyR.from_euler("xyz", [thx, thy, thz], degrees=True)
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def R_to_euler(R):
    """
    Obtain the thx,thy,thz angles that result in the rotation matrix Rz(thx) * Ry(thy) * Rx(thz)
    Reference: http://planning.cs.uiuc.edu/node103.html
    """
    return R.as_euler('xyz', degrees=True)
    # # To prevent numerical errors, avoid super small values.
    # epsilon = 1e-9
    # matrix[abs(matrix - 0.0) < epsilon] = 0.0
    # thz = to_degrees(math.atan2(matrix[1,0], matrix[0,0]))
    # thy = to_degrees(math.atan2(-matrix[2,0], math.sqrt(matrix[2,1]**2 + matrix[2,2]**2)))
    # thx = to_degrees(math.atan2(matrix[2,1], matrix[2,2]))
    # return thx, thy, thz

def R_to_quat(R):
    return R.as_quat()

def euler_to_quat(thx, thy, thz):
    return scipyR.from_euler("xyz", [thx, thy, thz], degrees=True).as_quat()

def quat_to_euler(x, y, z, w):
    return scipyR.from_quat([x,y,z,w]).as_euler("xyz", degrees=True)

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True

def R2d(th):
    return np.array([
        np.cos(th), -np.sin(th),
        np.sin(th), np.cos(th)
    ]).reshape(2,2)


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
