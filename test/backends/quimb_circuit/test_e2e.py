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
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_addon_mpf.backends import HAS_QUIMB
from qiskit_addon_mpf.costs import setup_frobenius_problem
from qiskit_addon_mpf.dynamic import setup_dynamic_lse
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit

if HAS_QUIMB:
    from qiskit_addon_mpf.backends.quimb_circuit import CircuitEvolver, CircuitState


@pytest.mark.skipif(not HAS_QUIMB, reason="Quimb is required for these unittests")
class TestEndToEnd:
    @pytest.mark.parametrize(
        ["time", "expected_A", "expected_b", "expected_coeffs"],
        [
            (
                0.5,
                [[1.0, 0.99961572], [0.99961572, 1.0]],
                [0.99939447, 0.99804667],
                [2.25365838, -1.25365812],
            ),
        ],
    )
    def test_end_to_end(self, time, expected_A, expected_b, expected_coeffs):
        np.random.seed(0)

        # constants
        L = 6
        W = 0.5
        epsilon = 0.5

        J = np.random.rand(L - 1) + W * np.ones(L - 1)
        # ZZ couplings
        Jz = 1.0
        # XX and YY couplings
        Jxx = epsilon
        hz = 0.000000001 * np.array([(-1) ** i for i in range(L)])

        hamil = SparsePauliOp.from_sparse_list(
            [("Z", [k], hz[k]) for k in range(L)]
            + [("ZZ", [k, k + 1], J[k] * Jz) for k in range(L - 1)]
            + [("YY", [k, k + 1], J[k] * Jxx) for k in range(L - 1)]
            + [("XX", [k, k + 1], J[k] * Jxx) for k in range(L - 1)],
            num_qubits=L,
        )

        dt = Parameter("dt")
        suz_4 = generate_time_evolution_circuit(hamil, synthesis=SuzukiTrotter(order=4), time=dt)
        suz_2 = generate_time_evolution_circuit(hamil, synthesis=SuzukiTrotter(order=2), time=dt)

        initial_state = QuantumCircuit(L)
        for k in range(1, L, 2):
            initial_state.x(k)

        model = setup_dynamic_lse(
            [4, 3],
            time,
            CircuitState,
            partial(CircuitEvolver, circuit=suz_4, dt=0.05),
            partial(CircuitEvolver, circuit=suz_2),
            initial_state,
        )
        np.testing.assert_allclose(model.b, expected_b, rtol=1e-4)
        np.testing.assert_allclose(model.A, expected_A, rtol=1e-4)

        prob, coeffs = setup_frobenius_problem(model)
        prob.solve()
        np.testing.assert_allclose(coeffs.value, expected_coeffs, rtol=1e-4)
