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

"""Sum-of-squares MPF coefficients."""

from __future__ import annotations

import cvxpy as cp

from .lse import LSE


def setup_sum_of_squares_problem(
    lse: LSE, *, max_l1_norm: float = 10.0
) -> tuple[cp.Problem, cp.Variable]:
    r"""Construct a :external:class:`cvxpy.Problem` for finding approximate MPF coefficients.

    The optimization problem constructed by this function is defined as follows:

    - the cost function minimizes the sum of squares
      (:external:func:`~cvxpy.atoms.sum_squares.sum_squares`) of the distances to an exact solution
      for all equations of the :class:`.LSE`:

      .. math::
         \sum_i \left( \sum_j A_{ij} x_j - b_i \right)^2

    - two constraints are set:

      1. the variables must sum to 1: :math:`\sum_i x_i == 1`
      2. the L1-norm (:external:class:`~cvxpy.atoms.norm1.norm1`) of the variables is bounded by
         ``max_l1_norm``

    Here is an example:

    >>> from qiskit_addon_mpf.costs import setup_sum_of_squares_problem
    >>> from qiskit_addon_mpf.static import setup_static_lse
    >>> lse = setup_static_lse([1,2,3], order=2, symmetric=True)
    >>> problem, coeffs = setup_sum_of_squares_problem(lse, max_l1_norm=3.0)
    >>> print(problem)  # doctest: +FLOAT_CMP
    minimize quad_over_lin(Vstack([1. 1.     1.]         @ x + -1.0,
                                  [1. 0.25   0.11111111] @ x + -0.0,
                                  [1. 0.0625 0.01234568] @ x + -0.0), 1.0)
    subject to Sum(x, None, False) == 1.0
               norm1(x) <= 3.0

    You can then solve the problem and access the expansion coefficients like so:

    >>> final_cost = problem.solve()
    >>> print(coeffs.value)  # doctest: +FLOAT_CMP
    [ 0.03513467 -1.          1.96486533]

    Args:
        lse: the linear system of equations from which to build the model.
        max_l1_norm: the upper limit to use for the constrain of the L1-norm of the variables.

    Returns:
        The optimization problem and coefficients variable.

    References:
        [1]: S. Zhuk et al., Phys. Rev. Research 6, 033309 (2024).
             https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033309
    """
    coeffs = lse.x
    # NOTE: the following list comprehension is required to support parameterized LSE objects
    cost = cp.sum_squares(cp.vstack([lse.A[i] @ coeffs - b for i, b in enumerate(lse.b)]))
    constraints = [cp.sum(coeffs) == 1, cp.norm1(coeffs) <= max_l1_norm]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, coeffs
