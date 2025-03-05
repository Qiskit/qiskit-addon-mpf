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
from qiskit_addon_mpf.backends import HAS_QUIMB
from qiskit_addon_mpf.costs import setup_frobenius_problem
from qiskit_addon_mpf.dynamic import setup_dynamic_lse

if HAS_QUIMB:
    from qiskit_addon_mpf.backends.quimb_tebd import MPOState, TEBDEvolver
    from quimb.tensor import MPO_identity, MPS_neel_state, SpinHam1D


@pytest.mark.skipif(not HAS_QUIMB, reason="Quimb is required for these unittests")
class TestEndToEnd:
    @pytest.mark.parametrize(
        ["time", "expected_A", "expected_b", "expected_coeffs"],
        [
            (
                0.5,
                [[1.0, 0.9997562], [0.9997562, 1.0]],
                [0.99952645, 0.99854528],
                [2.51221012, -1.51220984],
            ),
            (
                1.0,
                [[1.0, 0.99189288], [0.99189288, 1.0]],
                [0.9871936, 0.96466791],
                [1.88925272, -0.88925272],
            ),
            (
                1.5,
                [[1.0, 0.95352741], [0.95352741, 1.0]],
                [0.8887205, 0.79052836],
                [1.55645214, -0.55645214],
            ),
        ],
    )
    def test_end_to_end_builtin_suzuki(self, time, expected_A, expected_b, expected_coeffs):
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

        # Initialize the builder for a spin 1/2 chain
        builder = SpinHam1D(S=1 / 2)

        # Add XX and YY couplings for neighboring sites
        for i in range(L - 1):
            builder[i, i + 1] += 2.0 * Jxx * J[i], "-", "+"
            builder[i, i + 1] += 2.0 * Jxx * J[i], "+", "-"

        # Add ZZ couplings for neighboring sites
        for i in range(L - 1):
            builder[i, i + 1] += 4.0 * Jz * J[i], "Z", "Z"

        # Add the external Z-field (hz) to each site
        for i in range(L):
            builder[i] += -2.0 * hz[i], "Z"

        # Build the local Hamiltonian
        exact_model = builder.build_local_ham(L)

        split_opts = {
            "max_bond": 10,
            "cutoff": 1e-5,
            "cutoff_mode": "rel",
            "method": "svd",
            "renorm": False,
        }

        initial_state = MPS_neel_state(L)

        model = setup_dynamic_lse(
            [4, 3],
            time,
            lambda: MPOState(MPO_identity(L)),
            partial(
                TEBDEvolver,
                H=exact_model,
                dt=0.05,
                order=4,
                split_opts=split_opts,
            ),
            partial(
                TEBDEvolver,
                H=exact_model,
                order=2,
                split_opts=split_opts,
            ),
            initial_state,
        )
        np.testing.assert_allclose(model.b, expected_b, rtol=1.5e-3)
        np.testing.assert_allclose(model.A, expected_A, rtol=1e-3)

        prob, coeffs = setup_frobenius_problem(model)
        prob.solve()
        # NOTE: this particular test converges to fairly different overlaps in the CI on MacOS only.
        # Thus, the assertion threshold is so loose.
        np.testing.assert_allclose(coeffs.value, expected_coeffs, rtol=0.1)
