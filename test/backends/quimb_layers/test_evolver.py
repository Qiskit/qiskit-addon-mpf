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


import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit_addon_mpf.backends import HAS_QUIMB

if HAS_QUIMB:
    from qiskit_addon_mpf.backends.quimb_layers import LayerModel, LayerwiseEvolver
    from quimb.tensor import MPS_neel_state, SpinHam1D


def gen_ext_field_layer(n, hz):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.rz(-hz[q], q)
    return qc


def trotter_step(qc, q0, q1, Jxx, Jz):
    qc.rzz(Jz, q0, q1)
    qc.append(XXPlusYYGate(2.0 * Jxx), [q0, q1])


def gen_odd_coupling_layer(n, Jxx, Jz, J):
    qc = QuantumCircuit(n)
    for q in range(0, n, 2):
        trotter_step(qc, q, q + 1, J[q] * Jxx, J[q] * Jz)
    return qc


def gen_even_coupling_layer(n, Jxx, Jz, J):
    qc = QuantumCircuit(n)
    for q in range(1, n - 1, 2):
        q0 = q
        q1 = (q + 1) % n
        if q1 < q0:
            qc.barrier()
        trotter_step(qc, q0, q1, J[q0] * Jxx, J[q0] * Jz)
    return qc


@pytest.mark.skipif(not HAS_QUIMB, reason="Quimb is required for these unittests")
class TestLayerwiseEvolver:
    def test_compare_statevector(self):
        """Test the time-evolution logic by comparing against an exact statevector simulation.

        The reference value against which is being compared here can be obtained from:

        .. code-block:: python

            odd_coupling_layer = gen_odd_coupling_layer(L, dt * Jxx, dt * Jz, J)
            even_coupling_layer = gen_even_coupling_layer(L, dt * Jxx, dt * Jz, J)
            onsite_layer = gen_ext_field_layer(L, dt * hz)
            layers = [
                odd_coupling_layer,
                even_coupling_layer,
                onsite_layer,
                onsite_layer,
                even_coupling_layer,
                odd_coupling_layer,
            ]

            trotter_circ = QuantumCircuit(L)
            for layer in layers:
                trotter_circ = trotter_circ.compose(layer)
            trotter_circ = trotter_circ.repeat(N)

            init_circ = QuantumCircuit(L)
            init_circ.x(1)
            init_circ.x(3)

            full_circ = init_circ.copy()
            full_circ = full_circ.compose(trotter_circ)

            init_state_vec = Statevector(init_circ)
            full_state_vec = Statevector(full_circ)
            reference = full_state_vec.inner(init_state_vec)
        """

        np.random.seed(0)

        L = 4
        W = 0.5
        epsilon = 0.5
        J = np.random.rand(L - 1) + W * np.ones(L - 1)
        Jz = 1.0
        Jxx = epsilon
        hz = 0.000000001 * np.array([(-1) ** i for i in range(L)])

        N = 10
        dt = 0.05

        odd_coupling_layer = gen_odd_coupling_layer(L, Jxx, Jz, J)
        even_coupling_layer = gen_even_coupling_layer(L, Jxx, Jz, J)
        ext_field_layer = gen_ext_field_layer(L, hz)

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

        layers = [
            LayerModel.from_quantum_circuit(odd_coupling_layer, cyclic=False),
            LayerModel.from_quantum_circuit(even_coupling_layer, cyclic=False),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=True, cyclic=False),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=False, cyclic=False),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=False, cyclic=False),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=True, cyclic=False),
            LayerModel.from_quantum_circuit(even_coupling_layer, cyclic=False),
            LayerModel.from_quantum_circuit(odd_coupling_layer, cyclic=False),
        ]

        trunc_options = {
            "max_bond": 100,
            "cutoff": 1e-15,
            "cutoff_mode": "rel",
            "method": "svd",
            "renorm": False,
        }

        initial_state = MPS_neel_state(L)
        mps_state = initial_state.copy()
        mps_evo = LayerwiseEvolver(
            evolution_state=mps_state, layers=layers, dt=dt, split_opts=trunc_options
        )
        for _ in range(N):
            mps_evo.step()

        np.testing.assert_almost_equal(
            initial_state.overlap(mps_evo.pt), -0.2607402383827852 - 0.6343830867298741j
        )
