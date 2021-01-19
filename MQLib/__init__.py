"""
The MQLib module provides a python interface to the MQLib C++ library, as
described in https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0798 .
It can be used to access dozens of heuristics for Max-Cut and Quadratic
Unconstrained Binary Optimization (QUBO), two famous NP-hard problems.
Additionally, it provides access to a hyperheuristic, which predicts for a
given problem instance what the best-performing heuristic will be, and then
runs that one.
"""

import networkx as nx
from numbers import Number
import numpy as np
import os
import scipy
import _MQLib

class Instance:
    """
    Instance describes a problem instance for either Max-Cut or
    QUBO.
    """
    def __init__(self, problem, dat):
        """
        Construct a new 'Instance' object given the problem data. For
        Max-Cut, this is either a networkx graph or a matrix containing
        the (weighted) adjacency matrix for the graph. For QUBO, this
        is a square matrix Q. IMPORTANT: both the adjacency matrix and Q
        are symmetric. When inputting a matrix, you can either input the
        full symmetric matrix or you can input either the upper or lower
        triangular portion of the matrix. If an upper triangular matrix is
        inputted, then it is assumed that the exact same values will
        appear in the lower triangular piece. Similarly, if a lower
        triangular matrix is inputted, then it is assumed that the exact
        same values will appear in the upper triangular piece.

        :param problem: 'M' for Max-Cut or 'Q' for QUBO
        :param dat: a networkx graph (Max-Cut only) or something that can
        be converted to a scipy sparse matrix (e.g. a numpy array or list of
        lists).
        """
        if problem != "M" and problem != "Q":
            raise ValueError("Instance problem should be \"M\" or \"Q\"")

        # Convert dat to a COO sparse matrix
        if isinstance(dat, list):
            dat = np.array(dat, dtype=np.double)
        if isinstance(dat, np.ndarray):  # includes np.matrix
            if len(dat.shape) != 2 or dat.shape[0] != dat.shape[1]:
                raise ValueError("Need a 2d, square input matrix")
            mat = scipy.sparse.coo_matrix(dat, dtype=np.double)
        elif isinstance(dat, nx.classes.graph.Graph):
            if problem == "Q":
                raise ValueError("Network input is only for Max-Cut instances")
            if dat.is_directed():
                raise ValueError("Network should be undirected")
            mat = nx.to_scipy_sparse_matrix(dat, dtype=np.double)
        elif scipy.sparse.issparse(dat):
            if len(dat.shape) != 2 or dat.shape[0] != dat.shape[1]:
                raise ValueError("Need a 2d, square input matrix")
            mat = dat
            if not isinstance(mat, scipy.sparse.coo.coo_matrix):
                mat = scipy.sparse.coo_matrix(mat)
        else:
            raise TypeError("Unsupported problem data type")
        if mat.nnz == 0:
            raise ValueError("Input matrix has no non-zero elements")

        # Extract positions of non-zero elements and their values from
        # the sparse matrix
        left = mat.nonzero()[0] + 1
        if left.dtype != 'int32':
            left = left.astype(np.int32)
        right = mat.nonzero()[1] + 1
        if right.dtype != 'int32':
            right = right.astype(np.int32)
        vals = mat.data
        if vals.dtype != 'double':
            vals = vals.astype(np.double)

        # Don't send two copies of each element for a symmetric matrix;
        # instead confirm that we're indeed symmetric and then only send
        # a single version of each.
        if sum(left < right) > 0 and sum(left > right) > 0:
            # Check if symmetric: https://stackoverflow.com/a/48800071/3093387
            if (abs(mat-mat.T) > 1e-10).nnz > 0:
                raise ValueError("Matrix had entries above and below the main diagonal but was not symmetric")
            if problem == "M":
                keep = left < right
            else:
                keep = left <= right
            left = left[keep]
            right = right[keep]
            vals = vals[keep]

        # Instantiate our object and store it as a class variable
        self.inst = _MQLib._Inst(problem, left, right, vals, mat.shape[0])

    def getMetrics(self):
        """
        Compute a number of problem instance metrics for this problem instance,
        returning the metrics and the amount of wall clock time it took to compute
        them. All metrics are graph metrics, so QUBO instance are converted to
        Max-Cut graphs for the purposes of this computation.

        :return: A dict where key 'metrics' contains a dict of metric values
        and key 'runtimes' contains a dict of runtimes.
        """
        metrics = _MQLib.instanceMetrics(self.inst)
        return {"metrics": {x: y for x, y in zip(metrics[0], metrics[1])},
                "runtimes": {x: y for x, y in zip(metrics[2], metrics[3])}}

# Cached random forest models for use in the hyperheuristic. Since loading
# these takes a non-trivial amount of time, we'll only load them the first
# time the hyperheuristic is requested, and we will store them for
# subsequent calls to the hyperheuristic. We store this data as a length 1
# list so we can modify it from within a function.
_HHData = [None]

def runHeuristic(heuristic, instance, rtsec, cb_fun, seed=-1):
    """
    Run a specified heuristic on a given problem instance. Briefly, Max-Cut
    involves finding a partition of the nodes in the graph with the largest
    sum of edge weights between the two groups. Meanwhile, QUBO defines a
    square matrix Q and seeks to find the binary-valued column vector q that
    maximizes q'Qq.

    Heuristics are run until the specified runtime limit is reached. If the
    heuristic and instance are of different types (e.g. the heuristic solves
    Max-Cut but the instance is a QUBO instance), then the instance will be
    converted to the other problem type, solved with the heuristic, and then
    the solution will be converted back to the original problem.

    :param heuristic: The code of the heuristic, as returned by the
    MQLib.getHeuristics() function. The special code of "HH" runs a
    hyperheuristic, which uses machine learning to predict the best-performing
    heuristic based on instance characteristics and then runs that one.
    :param instance: A MQLib.Instance object to be optimized.
    :param rtsec: The runtime limit (wall clock time for heuristic execution),
    in seconds
    :param seed: A seed to ensure reproducible heuristic results. -1 indicates
    the current time should be used to seed the PRNG.
    :return: Returns a dict with a number of pieces of information. Key
    "heuristic" indicates the heuristic that was run (only informative for
    the hyperheuristic). "instance" is the passed MQLib.Instance object.
    "objval" is the best objective value found (sum of edge weights for Max-Cut
    or value of q'Qq for QUBO), and "solution" is a solution that achieved
    that best objective value (1/0 valued for QUBO and 1/-1 valued for Max-Cut).
    "bestsolhistory_objvals" and "bestsolhistory_runtimes" indicate the
    chronological list of all best objective values found by the heuristic,
    along with the wall clock times taken to achieve each of these values.
    """
    global _HHData
    if not isinstance(heuristic, str):
        raise TypeError("heuristic argument should be a string")
    if not isinstance(instance, Instance):
        raise TypeError("instance argument should be the Instance class")
    if not isinstance(rtsec, Number):
        raise TypeError("rtsec argument should be a number")
    if not isinstance(seed, int):
        raise TypeError("seed argument should be an integer")

    # Load the random forest models if this is our first call to the
    # hyperheuristic
    if heuristic == "HH" and _HHData[0] is None:
        datloc = os.path.abspath(os.path.join(os.path.dirname(__file__), "hhdata"))
        if not os.path.isdir(datloc) or len([x for x in os.listdir(datloc) if ".rf" in x]) == 0:
            raise RuntimeError("Invalid hyperheuristic data directory " + datloc)
        _HHData[0] = _MQLib._HHData(datloc)

    # Run the heuristic and return
    ret = _MQLib.runHeuristic(heuristic, instance.inst, rtsec, cb_fun, seed)
    return {"heuristic": ret[0],
            "instance": instance,
            "objval": ret[1],
            "solution": ret[2],
            "bestsolhistory_objvals": ret[3],
            "bestsolhistory_runtimes": ret[4]}

def getHeuristics():
    """
    Lists all heuristics available through the MQLib.

    :return: Returns a dict that separately lists heuristics for
    Max-Cut (key "MaxCut") and QUBO (key "QUBO"). For each problem,
    it returns a dict mapping heuristic code to a brief description.
    """
    ret = _MQLib.getHeuristics()
    return {"MaxCut": {x: y for x, y in zip(ret[0], ret[1])},
            "QUBO": {x: y for x, y in zip(ret[2], ret[3])}}
