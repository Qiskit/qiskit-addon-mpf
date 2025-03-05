# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
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
    from quimb.tensor import MPO_identity, MPS_neel_state, ham_1d_heis


@pytest.mark.skipif(not HAS_QUIMB, reason="Quimb is required for this regression test")
class TestDynamic:
    def test_numerical_check_accuracy(self):
        """This is a regression test against the evolution time comparisons."""
        np.random.seed(0)

        # constants
        L = 4
        exact_model = ham_1d_heis(L)

        split_opts = {
            "max_bond": 32,
            "cutoff": 1e-8,
            "cutoff_mode": "rel",
            "method": "svd",
            "renorm": False,
        }

        initial_state = MPS_neel_state(L)

        model = setup_dynamic_lse(
            [2, 3, 4],
            1.000000001,
            lambda: MPOState(MPO_identity(L)),
            partial(
                TEBDEvolver,
                H=exact_model,
                dt=0.1,
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
        np.testing.assert_allclose(model.b, [0.999878, 0.999976, 0.999993], rtol=1e-3)
        np.testing.assert_allclose(
            model.A,
            [[1.0, 0.999962, 0.999931], [0.999962, 1.0, 0.999995], [0.999931, 0.999995, 1.0]],
            rtol=1e-3,
        )

        prob, coeffs = setup_frobenius_problem(model)
        prob.solve()
        # NOTE: this particular test converges to fairly different overlaps in the CI on MacOS only.
        # Thus, the assertion threshold is so loose.
        np.testing.assert_allclose(coeffs.value, [-0.494253, 0.643685, 0.850568], rtol=0.1)
