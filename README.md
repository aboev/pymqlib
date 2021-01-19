# MQLib Python interface

Repository contains Python interface for [MQLib](https://github.com/MQLib/MQLib) library.
MQLib is a collection of well-known heuristics for solving combinatorial problems based on paper [What Works Best When? A Systematic Evaluation of Heuristics for Max-Cut and QUBO](https://www.semanticscholar.org/paper/What-Works-Best-When-A-Systematic-Evaluation-of-for-Dunning-Gupta/ab4b91cc3339727fe061cad5d23f4d65c30fb1ce).
Original source was obtained from [Pythonhosted](https://files.pythonhosted.org/packages/a1/8a/6bc95618d24e317988f463e095191a6d33475c820f89ba0c0445a188b0d6/MQLib-0.1.tar.gz) and refactored in accordance with MQLib.

Main difference is instrumentation of custom callback functions (see example below) and small fixes.

## Installation
```
pip install .
```

## Quick-Start
```
from MQLib import Instance, runHeuristic
import numpy as np

Q = np.random.randn(10,10)
Q = (Q + Q.T)/2

i = Instance('Q', Q)

def cb_fun(spins):
    print("New solution = %s" % spins)
    return 1

runHeuristic("BURER2002", i, 10, cb_fun, 100)
```
