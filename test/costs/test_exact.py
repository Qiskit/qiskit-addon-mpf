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

"""Tests for the ``qiskit_addon_mpf.static.exact`` module."""

import cvxpy as cp
import numpy as np
import pytest
from qiskit_addon_mpf.costs.exact import setup_exact_problem
from qiskit_addon_mpf.static import setup_static_lse


class TestExactCoeffs:
    """Tests for the ``qiskit_addon_mpf.static.exact`` module."""

    def test_setup_exact_problem(self, subtests):
        """Tests the :meth:`.setup_exact_problem` method."""
        trotter_steps = [1, 2, 4]
        lse = setup_static_lse(trotter_steps, order=2, symmetric=True)
        problem, coeffs = setup_exact_problem(lse)

        with subtests.test(msg="final cost"):
            final_cost = problem.solve()
            pytest.approx(final_cost, 1.888888888888)

        with subtests.test(msg="optimal coefficients"):
            expected = np.array([0.02222222, -0.44444444, 1.42222222])
            np.testing.assert_allclose(coeffs.value, expected)

    @pytest.mark.parametrize(
        ["trotter_steps", "expected_coeffs", "expected_cost"],
        [
            # well-conditioned
            ([1, 2], [-1.0, 2.0], 3.0),
            ([1, 3], [-0.5, 1.5], 2.0),
            ([2, 4], [-1.0, 2.0], 3.0),
            ([2, 5], [-2 / 3, 5 / 3], 7 / 3),
            ([1, 2, 6], [0.2, -1.0, 1.8], 3.0),
            ([1, 2, 7], [1 / 6, -4 / 5, 49 / 30], 2.6),
            # ill-conditioned
            ([6, 7], [-6, 7], 13.0),
            ([3, 4, 5, 6, 7], [27 / 8, -128 / 3, 625 / 4, -216, 2401 / 24], 518.3333332),
            (
                [1, 2, 3, 4, 5, 6, 7],
                [1 / 720, -8 / 15, 243 / 16, -1024 / 9, 15625 / 48, -3888 / 10, 117649 / 720],
                1007.22220288,
            ),
        ],
    )
    def test_exact_order_1_references(
        self, subtests, trotter_steps: list[int], expected_coeffs: list[float], expected_cost: float
    ):
        """This test ensures the correct behavior of the exact static MPF coefficient model.

        It does so, using the test-cases listed in Appendix E of [1].

        [1]: A. Carrera Vazquez et al., Quantum 7, 1067 (2023).
             https://quantum-journal.org/papers/q-2023-07-25-1067/
        """
        lse = setup_static_lse(trotter_steps, order=1)
        problem, coeffs = setup_exact_problem(lse)

        with subtests.test(msg="final cost"):
            final_cost = problem.solve()
            pytest.approx(final_cost, expected_cost)

        with subtests.test(msg="optimal coefficients"):
            np.testing.assert_allclose(coeffs.value, expected_coeffs)

    @pytest.mark.parametrize(
        ["order", "trotter_steps", "expected_coeffs", "expected_cost"],
        [
            (2, [1, 2], [-1.0 / 3.0, 4.0 / 3.0], 5.0 / 3.0),
            (2, [1, 2, 4], [1.0 / 45.0, -4.0 / 9.0, 64.00 / 45.0], 85.0 / 45.0),
            (2, [1, 2, 6], [1.0 / 105.0, -1.0 / 6.0, 81.0 / 70.0], 4.0 / 3.0),
            (4, [1, 2], [-1.0 / 15.0, 16.0 / 15.0], 17.0 / 15.0),
            (4, [1, 2, 3], [1 / 336.0, -32 / 105.0, 729 / 560.0], 1.60952381),
            (4, [1, 2, 4], [1 / 945.0, -16 / 189.0, 1024 / 945.0], 1.16931217),
        ],
    )
    def test_exact_higher_order_references(
        self,
        subtests,
        order: int,
        trotter_steps: list[int],
        expected_coeffs: list[float],
        expected_cost: float,
    ):
        """This test ensures the correct behavior for higher order formulas.

        It does so, using some of the test-cases listed in Appendix A of [1].

        [1]: G. H. Low et al, arXiv:1907.11679v2 (2019).
             https://arxiv.org/abs/1907.11679v2
        """
        lse = setup_static_lse(trotter_steps, order=order, symmetric=True)
        problem, coeffs = setup_exact_problem(lse)

        with subtests.test(msg="final cost"):
            final_cost = problem.solve()
            pytest.approx(final_cost, expected_cost)

        with subtests.test(msg="optimal coefficients"):
            np.testing.assert_allclose(coeffs.value, expected_coeffs)

    def test_setup_exact_problem_params(self, subtests):
        """Tests the :meth:`.setup_exact_problem` method with parameters."""
        ks = cp.Parameter(2)
        lse = setup_static_lse(ks, order=1)
        problem, coeffs = setup_exact_problem(lse)

        for trotter_steps, expected_coeffs, expected_cost in [
            ([1, 2], [-1.0, 2.0], 3.0),
            ([1, 3], [-0.5, 1.5], 2.0),
            ([2, 4], [-1.0, 2.0], 3.0),
            ([2, 5], [-2 / 3, 5 / 3], 7 / 3),
            ([6, 7], [-6, 7], 13.0),
        ]:
            ks.value = trotter_steps

            with subtests.test(msg="final cost"):
                final_cost = problem.solve()
                pytest.approx(final_cost, expected_cost)

            with subtests.test(msg="optimal coefficients"):
                np.testing.assert_allclose(coeffs.value, expected_coeffs)
