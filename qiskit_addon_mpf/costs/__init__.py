# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cost functions for MPF coefficients.

.. currentmodule:: qiskit_addon_mpf.costs

This module provides a number of optimization problem generator functions, each implementing a
different cost function as the problem's target objective. All of the functions provided by this
module take a linear system of equations (:class:`.LSE`) encoding the parameters of the optimization
problem as their first argument.

.. autoclass:: LSE

Optimization problem constructors
---------------------------------

.. autofunction:: setup_exact_problem

.. autofunction:: setup_sum_of_squares_problem

.. autofunction:: setup_frobenius_problem
"""

from .exact import setup_exact_problem
from .frobenius import setup_frobenius_problem
from .lse import LSE
from .sum_of_squares import setup_sum_of_squares_problem

__all__ = [
    "LSE",
    "setup_exact_problem",
    "setup_sum_of_squares_problem",
    "setup_frobenius_problem",
]
