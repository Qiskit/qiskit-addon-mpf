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

"""Tests for the ``qiskit_addon_mpf.costs.LSE`` class."""

import numpy as np
import pytest
from qiskit_addon_mpf.costs import LSE
from qiskit_addon_mpf.static import setup_static_lse


class TestLSE:
    """Tests for the ``qiskit_addon_mpf.costs.LSE`` class."""

    def test_lse_solve(self):
        """Tests the :meth:`.LSE.solve` method."""
        trotter_steps = [1, 2, 4]
        lse = setup_static_lse(trotter_steps, order=2, symmetric=True)
        coeffs = lse.solve()
        expected = np.array([0.022222222, -0.444444444, 1.422222222])
        np.testing.assert_allclose(coeffs, expected)

    def test_lse_solve_invalid(self):
        """Tests the handling of an invalid model."""
        mat_a = np.asarray([[1.0, 0.9], [0.9, 1.0]])
        vec_b = np.asarray([0.9, 0.9])
        lse = LSE(mat_a, vec_b)
        with pytest.raises(ValueError):
            lse.solve()
