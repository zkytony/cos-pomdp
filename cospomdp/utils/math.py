# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def law_of_cos(a, b, angle):
    """Given length of edges a and b and angle (degrees)
    in between, return the length of the edge opposite to the angle"""
    return math.sqrt(a**2 + b**2 - 2*a*b*math.cos(to_rad(angle)))

def inverse_law_of_cos(a, b, c):
    """Given three edges, a, b, c, figure out
    the angle between a and b (i.e. opposite of c), in degrees"""
    costh = (a**2 + b**2 - c**2) / (2*a*b)
    return to_deg(math.acos(costh))

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

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def indicies2d(m, n):
    # reference: https://stackoverflow.com/a/44230705/2893053
    return np.indices((m,n)).transpose(1,2,0)


def tind_test(sample1, sample2):
    """Performs a two-sample independent t-test.  Note that in statistics, a sample
    is a set of individuals (observations) or objects collected or selected from
    a statistical population by a defined procedure.

    references:
    https://www.reneshbedre.com/blog/ttest.html#two-sample-t-test-unpaired-or-independent-t-test.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    Formula:

        t = (m1 - m2)  / sqrt ( s^2 (1/n1 + 1/n2) )

    where m1, m2 are the means of two independent samples, and s^2 is the "pooled"
    sample variance, calculated as:

        s^2 = [ (n1-1)s1^2 + (n2-1)s2^2 ] / (n1 + n2 - 2)

    and n1, n2 are the sample sizes

    Note: Independent samples are samples that are selected randomly so that its
    observations do not depend on the values other observations.

    Args:
        sample1 (list or numpy array)
        sample2 (list or numpy array)
    Returns:
        (tstatistic, pvalue)
    """
    res = stats.ttest_ind(a=sample1, b=sample2, equal_var=True)
    return res.statistic, res.pvalue


def pval2str(pval):
    """Converts p value to ns, *, **, etc.
    Uses the common convention."""
    if math.isnan(pval):
        return "NA"
    if pval > 0.05:
        return "ns"
    else:
        if pval <= 0.0001:
            return "****"
        elif pval <= 0.001:
            return "***"
        elif pval <= 0.01:
            return "**"
        else:
            return "*"

def wilcoxon_test(sample1, sample2):
    """This is a nonparametric test (i.e. no need to compute statistical
    parameter from the sample like mean).

    From Wikipedia:
    When applied to test the location of a set of samples, it serves the same
    purpose as the one-sample Student's t-test.

    On a set of matched samples, it is a paired difference test like the paired
    Student's t-test

    Unlike Student's t-test, the Wilcoxon signed-rank test does not assume that
    the differences between paired samples are normally distributed.

    On a wide variety of data sets, it has greater statistical power than
    Student's t-test and is more likely to produce a statistically significant
    result. The cost of this applicability is that it has less statistical power
    than Student's t-test when the data are normally distributed.
    """
    def _all_zeros(sample):
        return all([abs(s) <= 1e-12 for s in sample])
    if _all_zeros(sample1) and _all_zeros(sample2):
        # the test cannot be performed; the two samples have no difference
        return float('nan'), float('nan')

    res = stats.wilcoxon(x=sample1, y=sample2)
    return res.statistic, res.pvalue


def test_significance_pairwise(results, sigstr=False, method="t_ind"):
    """
    Runs statistical significance tests for all pairwise combinations.
    Returns result as a table. Uses two-sample t-test. Assumes independent sample.

    Args:
        results (dict): Maps from method name to a list of values for the result.
        sigstr (bool): If True, then the table entries will be strings like *, **, ns etc;
             otherwise, they will be pvalues.
    Returns:
        pd.DataFrame: (X, Y) entry will be the statistical significance between
            method X and method Y.
    """
    method_names = list(results.keys())
    rows = []
    for meth1 in method_names:
        row = []
        for meth2 in method_names:
            if meth1 == meth2:
                row.append("-");
            else:
                if method == "t_ind":
                    _, pval = tind_test(results[meth1], results[meth2])
                elif method == "wilcoxon":
                    _, pval = wilcoxon_test(results[meth1], results[meth2])
                else:
                    raise ValueError("Unable to perform significance test {}".format(method))

                pvalstr = "{0:.4f}".format(pval)
                if sigstr:
                    pvalstr += " ({})".format(pval2str(pval))
                row.append(pvalstr)
        rows.append(row)
    df = pd.DataFrame(rows, method_names, method_names)
    return df
