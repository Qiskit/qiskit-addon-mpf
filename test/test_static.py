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

"""Tests for the ``qiskit_addon_mpf.static.lse`` module."""

import cvxpy as cp
import numpy as np
import pytest
from qiskit_addon_mpf.static import setup_static_lse


class TestLSE:
    """Tests for the ``qiskit_addon_mpf.static.lse`` module."""

    def test_setup_static_lse(self, subtests):
        """Tests the :meth:` setup_static_lse` method."""
        trotter_steps = [1, 2, 4]
        expected_b = np.array([1, 0, 0])

        with subtests.test(msg="order=1"):
            lse = setup_static_lse(trotter_steps, order=1)
            expected_A = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.25], [1.0, 0.25, 0.0625]])
            np.testing.assert_allclose(lse.A, expected_A)
            np.testing.assert_allclose(lse.b, expected_b)
            expected_c = np.array([0.33333333, -2.0, 2.66666667])
            np.testing.assert_allclose(lse.solve(), expected_c)

        with subtests.test(msg="order=2"):
            lse = setup_static_lse(trotter_steps, order=2, symmetric=True)
            expected_A = np.array([[1.0, 1.0, 1.0], [1.0, 0.25, 0.0625], [1.0, 0.0625, 0.00390625]])
            np.testing.assert_allclose(lse.A, expected_A)
            np.testing.assert_allclose(lse.b, expected_b)
            expected_c = np.array([0.022222222, -0.444444444, 1.422222222])
            np.testing.assert_allclose(lse.solve(), expected_c)

    def test_lse_with_params(self, subtests):
        """Tests the :meth:` setup_static_lse` method with parameters."""
        trotter_steps = cp.Parameter(3)

        lse = setup_static_lse(trotter_steps)

        with subtests.test(msg="assert ValueError"), pytest.raises(ValueError):
            lse.solve()

        trotter_steps.value = [1, 2, 4]

        expected_A = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.25], [1.0, 0.25, 0.0625]])
        for idx, row in enumerate(expected_A):
            np.testing.assert_allclose(lse.A[idx].value, row)

        expected_c = np.array([0.33333333, -2.0, 2.66666667])
        np.testing.assert_allclose(lse.solve(), expected_c)
