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

"""Tests for the ``qiskit_addon_mpf.static.sum_of_squares`` module."""

from typing import Any, ClassVar

import cvxpy as cp
import numpy as np
import pytest
from qiskit_addon_mpf.costs.sum_of_squares import setup_sum_of_squares_problem
from qiskit_addon_mpf.static import setup_static_lse


class TestSumOfSquaresCoeffs:
    """Tests for the ``qiskit_addon_mpf.static.sum_of_squares`` module."""

    CVXPY_SOLVER_SETTINGS: ClassVar[dict[str, Any]] = {
        "warm_start": False,
        "eps_abs": 1e-7,
        "eps_rel": 1e-7,
    }

    def test_setup_sum_of_squares_problem(self, subtests):
        """Tests the :meth:`.setup_sum_of_squares_problem` method."""
        trotter_steps = [1, 2, 4]
        lse = setup_static_lse(trotter_steps, order=2, symmetric=True)
        problem, coeffs = setup_sum_of_squares_problem(lse)

        with subtests.test(msg="final cost"):
            final_cost = problem.solve(**self.CVXPY_SOLVER_SETTINGS)
            pytest.approx(final_cost, 0)

        with subtests.test(msg="optimal coefficients"):
            expected = np.array([0.02222225, -0.44444444, 1.42222216])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-4)

    def test_setup_sum_of_squares_problem_max_l1_norm(self, subtests):
        """Tests the :meth:`.setup_sum_of_squares_problem` method with ``max_l1_norm``."""
        trotter_steps = [1, 2, 4]
        lse = setup_static_lse(trotter_steps, order=2, symmetric=True)
        problem, coeffs = setup_sum_of_squares_problem(lse, max_l1_norm=1.5)

        with subtests.test(msg="final cost"):
            final_cost = problem.solve(**self.CVXPY_SOLVER_SETTINGS)
            pytest.approx(final_cost, 0.00035765)

        with subtests.test(msg="optimal coefficients"):
            expected = np.array([-0.001143293, -0.2488567, 1.25])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-4)

    def test_setup_sum_of_squares_problem_params(self, subtests):
        """Tests the :meth:`.setup_sum_of_squares_problem` method with parameters."""
        ks = cp.Parameter(3)
        lse = setup_static_lse(ks, order=2, symmetric=True)
        problem, coeffs = setup_sum_of_squares_problem(lse)

        ks.value = [1, 2, 4]

        with subtests.test(msg="final cost"):
            final_cost = problem.solve(**self.CVXPY_SOLVER_SETTINGS)
            pytest.approx(final_cost, 0)

        with subtests.test(msg="optimal coefficients"):
            expected = np.array([0.02222225, -0.44444444, 1.42222216])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-4)
