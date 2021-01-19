from glob import glob
from setuptools import setup, Extension
import numpy

description = """
Python interface to the MQLib, a C++ library of heuristics for Max-Cut
and Quadratic Unconstrained Binary Optimization (QUBO). Also includes a
hyperheuristic, which uses machine learning to predict the best-performing
heuristic for a given problem instance and then runs that heuristic.

This library and the related systematic heuristic evaluation strategy are described in [the paper](https://github.com/MQLib/MQLib/blob/master/paper/SECM_final.pdf). To cite the MQLib, please use:
```
@article{DunningEtAl2018,
  title={What Works Best When? A Systematic Evaluation of Heuristics for Max-Cut and {QUBO}},
  author={Dunning, Iain and Gupta, Swati and Silberholz, John},
  year={2018},
  journal={{INFORMS} Journal on Computing},
  volume={30},
  number={3}
}
```
"""

setup(name = "MQLib",
      version = "0.1",
      author = "Iain Dunning, Swati Gupta, and John Silberholz",
      author_email = "john.silberholz@gmail.com",
      description = "Heuristics for the Max-cut and QUBO combinatorial optimization problems",
      long_description = description,
      long_description_content_type="text/markdown",
      url = "https://github.com/MQLib/MQLib",
      ext_modules = [Extension("_MQLib", glob("src/**/*.cpp", recursive=True),
                               include_dirs = ["include", numpy.get_include()],
                               extra_compile_args=['-std=c++0x', '-O2'])],
      packages = ["MQLib"],
      package_data = {"": ["**/*.rf"]},
      install_requires = ['numpy', 'scipy', 'networkx'],
      python_requires='>=3.5',
      classifiers = [
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: C++",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
      )
