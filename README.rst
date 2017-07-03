pyscgm
======

.. image:: https://img.shields.io/pypi/v/pyscgm.svg
    :target: https://pypi.python.org/pypi/pyscgm
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/dseuss/pyscgm.png
   :target: https://travis-ci.org/dseuss/pyscgm
   :alt: Latest Travis CI build status

An implementation of sketchy CGM from [1] for low-rank matrix recovery.
The module `pyscgm.extmath` provides implementations of the randomized
algorithms from [2] for partial eigenvalue and singular value decompositions
for matrix-free linear algebra.
Both work for real and complex matrices.
The interface is based on `scipy.sparse.linalg.LinearOperator`.

Usage
-----

Installation
------------

Requirements
^^^^^^^^^^^^

Compatibility
-------------

References
---------
__[1]__ Yurtsever, A., Udell, M., TroJ. A., &; Cevher, V. (2017). Sketchy Decisions: Convex Low-Rank Matrix Optimization with Optimal Storage.  *arXiv:1702.06838 [math, Stat]*. Retrieved from http://arxiv.org/abs/1702.06838

__[2]__ Halko, N., Martinsson, P.-G., & TroJ. A. (2009). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *arXiv:0909.4061 [math]*. Retrieved from http://arxiv.org/abs/0909.4061


Licence
-------

This work is distributed under the terms of the BSD 3-clause license (see
[LICENSE](LICENSE)).

Authors
-------

`pyscgm` was written by `Daniel Suess <daniel@dsuess.me>`_.
