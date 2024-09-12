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

"""Tests for the ``qiskit_addon_mpf.static.approximate`` module."""

import unittest

import cvxpy as cp
import numpy as np
from qiskit_addon_mpf.static.approximate import setup_approximate_model
from qiskit_addon_mpf.static.lse import setup_lse


class TestApproximateCoeffs(unittest.TestCase):
    """Tests for the ``qiskit_addon_mpf.static.approximate`` module."""

    def test_setup_approximate_model(self):
        """Tests the :meth:`.setup_approximate_model` method."""
        trotter_steps = [1, 2, 4]
        lse = setup_lse(trotter_steps, order=2, symmetric=True)
        problem, coeffs = setup_approximate_model(lse)

        with self.subTest("final cost"):
            final_cost = problem.solve()
            self.assertAlmostEqual(final_cost, 0)

        with self.subTest("optimal coefficients"):
            expected = np.array([0.02222225, -0.44444444, 1.42222216])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-5)

    def test_setup_approximate_model_max_l1_norm(self):
        """Tests the :meth:`.setup_approximate_model` method with ``max_l1_norm``."""
        trotter_steps = [1, 2, 4]
        lse = setup_lse(trotter_steps, order=2, symmetric=True)
        problem, coeffs = setup_approximate_model(lse, max_l1_norm=1.5)

        with self.subTest("final cost"):
            final_cost = problem.solve()
            self.assertAlmostEqual(final_cost, 0.00035765)

        with self.subTest("optimal coefficients"):
            expected = np.array([-0.001143293, -0.2488567, 1.25])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-5)

    def test_setup_approximate_model_params(self):
        """Tests the :meth:`.setup_approximate_model` method with parameters."""
        ks = cp.Parameter(3)
        lse = setup_lse(ks, order=2, symmetric=True)
        problem, coeffs = setup_approximate_model(lse)

        ks.value = [1, 2, 4]

        with self.subTest("final cost"):
            final_cost = problem.solve()
            self.assertAlmostEqual(final_cost, 0)

        with self.subTest("optimal coefficients"):
            expected = np.array([0.02222225, -0.44444444, 1.42222216])
            np.testing.assert_allclose(coeffs.value, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
