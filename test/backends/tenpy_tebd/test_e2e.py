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

from functools import partial

import numpy as np
import pytest
from qiskit_addon_mpf.backends import HAS_TENPY
from qiskit_addon_mpf.costs import setup_frobenius_problem
from qiskit_addon_mpf.dynamic import setup_dynamic_lse

if HAS_TENPY:
    from qiskit_addon_mpf.backends.tenpy_tebd import MPOState, MPS_neel_state, TEBDEvolver
    from tenpy.models import XXZChain2


@pytest.mark.skipif(not HAS_TENPY, reason="TeNPy is required for these unittests")
class TestEndToEnd:
    @pytest.mark.parametrize(
        ["time", "expected_A", "expected_b", "expected_coeffs"],
        [
            (
                0.5,
                [[1.0, 0.9997562], [0.9997562, 1.0]],
                [0.99944125, 0.99857914],
                [2.26805572, -1.26805543],
            ),
            (
                1.0,
                [[1.0, 0.99189288], [0.99189288, 1.0]],
                [0.98478594, 0.9676077],
                [1.55870386, -0.55870386],
            ),
            (
                1.5,
                [[1.0, 0.95352741], [0.95352741, 1.0]],
                [0.93918471, 0.71967399],
                [2.8617227, -1.8617227],
            ),
        ],
    )
    def test_end_to_end(self, time, expected_A, expected_b, expected_coeffs):
        np.random.seed(0)

        # constants
        L = 10
        W = 0.5
        epsilon = 0.5

        J = np.random.rand(L - 1) + W * np.ones(L - 1)
        # ZZ couplings
        Jz = 1.0
        # XX and YY couplings
        Jxx = epsilon

        # base coupling
        # external field
        hz = 0.000000001 * np.array([(-1) ** i for i in range(L)])

        # This is the full model that we want to simulate. It is used for the "exact" time evolution
        # (which is approximated via a fourth-order Suzuki-Trotter formula).
        exact_model = XXZChain2(
            {
                "L": L,
                "Jz": 4.0 * Jz * J,
                "Jxx": 4.0 * Jxx * J,
                "hz": 2.0 * hz,
                "bc_MPS": "finite",
                "sort_charge": False,
            }
        )

        options_common = {
            "trunc_params": {
                "chi_max": 10,
                "svd_min": 1e-5,
                "trunc_cut": None,
            },
            "preserve_norm": False,
        }
        options_exact = options_common.copy()
        options_exact["order"] = 4

        options_approx = options_common.copy()
        options_approx["order"] = 2

        initial_state = MPS_neel_state(exact_model.lat)

        model = setup_dynamic_lse(
            [4, 3],
            time,
            partial(MPOState.initialize_from_lattice, exact_model.lat),
            partial(
                TEBDEvolver,
                model=exact_model,
                dt=0.05,
                options=options_exact,
            ),
            partial(
                TEBDEvolver,
                model=exact_model,
                options=options_approx,
            ),
            initial_state,
        )
        np.testing.assert_allclose(model.b, expected_b, rtol=1e-4)
        np.testing.assert_allclose(model.A, expected_A, rtol=1e-4)

        prob, coeffs = setup_frobenius_problem(model)
        prob.solve()
        np.testing.assert_allclose(coeffs.value, expected_coeffs, rtol=1e-4)
