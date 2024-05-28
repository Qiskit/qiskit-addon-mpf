# This code is part of a Qiskit project.
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

"""Exact static MPF coefficients."""

from __future__ import annotations

import cvxpy as cp

from .lse import LSE


def setup_exact_model(lse: LSE) -> tuple[cp.Problem, cp.Variable]:
    r"""Construct a :external:class:`cvxpy.Problem` for finding exact static MPF coefficients.

    .. note::

       The coefficients found via this optimization problem will be identical to the analytical ones
       obtained from the :meth:`.LSE.solve` method. This additional interface exists to highlight
       the parallel to the :func:`.setup_approximate_model` interface. It also serves educational
       purposes for how-to approach optimization problems targeting MPF coefficients.

    The optimization problem constructed by this class is defined as follows:

    - the cost function minimizes the L1-norm (:external:class:`~cvxpy.atoms.norm1.norm1`) of the
      variables (:attr:`.LSE.x`)
    - the constraints correspond to each equation of the :class:`.LSE`:

      .. math::
         \sum_j A_{ij} x_j = b_i


    Here is an example:

    >>> from qiskit_addon_mpf.static import setup_lse, setup_exact_model
    >>> lse = setup_lse([1,2,3], order=2, symmetric=True)
    >>> problem, coeffs = setup_exact_model(lse)
    >>> print(problem)
    minimize norm1(x)
    subject to Sum([1. 1. 1.] @ x, None, False) == 1.0
               Sum([1. 0.25   0.11111111] @ x, None, False) == 0.0
               Sum([1. 0.0625 0.01234568] @ x, None, False) == 0.0

    You can then solve the problem and extract the expansion coefficients like so:

    >>> final_cost = problem.solve()
    >>> print(coeffs.value)  # doctest: +FLOAT_CMP
    [ 0.04166667 -1.06666667  2.025     ]

    Args:
        lse: the linear system of equations from which to build the model.

    Returns:
        The optimization problem and coefficients variable.

    References:
        [1]: A. Carrera Vazquez et al., Quantum 7, 1067 (2023).
             https://quantum-journal.org/papers/q-2023-07-25-1067/
    """
    coeffs = lse.x
    cost = cp.norm1(coeffs)
    constraints = [cp.sum(lse.A[idx] @ coeffs) == b for idx, b in enumerate(lse.b)]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, coeffs
