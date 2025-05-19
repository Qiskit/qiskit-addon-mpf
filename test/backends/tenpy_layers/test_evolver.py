# This code is a Qiskit project.
#
# (C) Copyright IBM 2024, 2025.
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
from qiskit_addon_mpf.backends import HAS_TENPY

if HAS_TENPY:
    from qiskit_addon_mpf.backends.tenpy_layers import LayerModel, LayerwiseEvolver
    from qiskit_addon_mpf.backends.tenpy_tebd import MPOState, MPS_neel_state


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


@pytest.mark.skipif(not HAS_TENPY, reason="Tenpy is required for these unittests")
class TestLayerwiseEvolver:
    def test_N_steps_guard(self):
        L = 6

        qc = QuantumCircuit(L)
        for i in range(0, L - 1, 2):
            qc.rzz(1.0, i, i + 1)

        model = LayerModel.from_quantum_circuit(qc)

        common_state = MPOState.initialize_from_lattice(model.lat)

        algo = LayerwiseEvolver(common_state, [model], {})

        with pytest.raises(RuntimeError):
            algo.evolve(2, 0.1)

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

        model_opts = {
            "bc_MPS": "finite",
            "conserve": "Sz",
            "sort_charge": False,
        }

        layers = [
            LayerModel.from_quantum_circuit(odd_coupling_layer, **model_opts),
            LayerModel.from_quantum_circuit(even_coupling_layer, **model_opts),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=True, **model_opts),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=False, **model_opts),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=False, **model_opts),
            LayerModel.from_quantum_circuit(ext_field_layer, keep_only_odd=True, **model_opts),
            LayerModel.from_quantum_circuit(even_coupling_layer, **model_opts),
            LayerModel.from_quantum_circuit(odd_coupling_layer, **model_opts),
        ]

        trunc_options = {
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1e-15,
                "trunc_cut": None,
            },
            "preserve_norm": False,
            "order": 2,
        }

        initial_state = MPS_neel_state(layers[0].lat)
        mps_state = initial_state.copy()
        mps_evo = LayerwiseEvolver(evolution_state=mps_state, layers=layers, options=trunc_options)
        for _ in range(N):
            mps_evo.run_evolution(1, dt)

        np.testing.assert_almost_equal(
            mps_state.overlap(initial_state), -0.2607402383827852 - 0.6343830867298741j
        )

    def test_compare_middle_out(self):
        """Regression test against https://github.com/Qiskit/qiskit-addon-mpf/issues/78.

        Essentially, this test ensures that the layers are applied in _reverse_ order when acting on
        an MPO identity as the common state.

        The reference value against which is being compared here can be obtained from:

        .. code-block:: python

            odd_coupling_layer_lhs = gen_odd_coupling_layer(L, dt_lhs * Jxx, dt_lhs * Jz, J)
            even_coupling_layer_lhs = gen_even_coupling_layer(L, dt_lhs * Jxx, dt_lhs * Jz, J)
            onsite_layer_lhs = gen_ext_field_layer(L, dt_lhs * hz)
            layers = [
                odd_coupling_layer_lhs,
                even_coupling_layer_lhs,
                onsite_layer_lhs,
            ]

            trotter_circ_lhs = QuantumCircuit(L)
            for layer in layers:
                trotter_circ_lhs.compose(layer, inplace=True)
            trotter_circ_lhs = trotter_circ_lhs.repeat(int(time / dt_lhs))

            odd_coupling_layer_rhs = gen_odd_coupling_layer(L, dt_rhs * Jxx, dt_rhs * Jz, J)
            even_coupling_layer_rhs = gen_even_coupling_layer(L, dt_rhs * Jxx, dt_rhs * Jz, J)
            onsite_layer_rhs = gen_ext_field_layer(L, dt_rhs * hz)
            layers = [
                odd_coupling_layer_rhs,
                even_coupling_layer_rhs,
                onsite_layer_rhs,
            ]

            trotter_circ_rhs = QuantumCircuit(L)
            for layer in layers:
                trotter_circ_rhs.compose(layer, inplace=True)
            trotter_circ_rhs = trotter_circ_rhs.repeat(int(time / dt_rhs))

            init_circ = QuantumCircuit(L)
            init_circ.x(1)
            init_circ.x(3)

            full_circ_lhs = init_circ.copy()
            full_circ_lhs.compose(trotter_circ_lhs, inplace=True)

            full_circ_rhs = init_circ.copy()
            full_circ_rhs.compose(trotter_circ_rhs, inplace=True)

            full_state_vec_lhs = Statevector(full_circ_lhs)
            full_state_vec_rhs = Statevector(full_circ_rhs)
            reference = full_state_vec_lhs.inner(full_state_vec_rhs)
        """

        np.random.seed(0)

        L = 4
        W = 0.5
        epsilon = 0.5
        J = np.random.rand(L - 1) + W * np.ones(L - 1)
        Jz = 1.0
        Jxx = epsilon
        hz = 0.000000001 * np.array([(-1) ** i for i in range(L)])

        dt_lhs = 0.01
        dt_rhs = 0.25
        time = 0.5

        odd_coupling_layer = gen_odd_coupling_layer(L, Jxx, Jz, J)
        even_coupling_layer = gen_even_coupling_layer(L, Jxx, Jz, J)
        ext_field_layer = gen_ext_field_layer(L, hz)

        model_opts = {
            "bc_MPS": "finite",
            "conserve": "Sz",
            "sort_charge": False,
        }

        layers = [
            LayerModel.from_quantum_circuit(odd_coupling_layer, **model_opts),
            LayerModel.from_quantum_circuit(even_coupling_layer, **model_opts),
            LayerModel.from_quantum_circuit(ext_field_layer, **model_opts),
        ]

        trunc_options = {
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1e-15,
                "trunc_cut": None,
            },
            "preserve_norm": False,
            "order": 2,
        }

        initial_state = MPS_neel_state(layers[0].lat)
        common_state = MPOState.initialize_from_lattice(layers[0].lat)
        mpo_evo_lhs = LayerwiseEvolver(
            evolution_state=common_state, layers=layers, options=trunc_options
        )
        mpo_evo_lhs.conjugate = True
        mpo_evo_rhs = LayerwiseEvolver(
            evolution_state=common_state, layers=layers, options=trunc_options
        )

        while np.round(mpo_evo_lhs.evolved_time, 8) < time:
            while np.round(mpo_evo_rhs.evolved_time, 8) < np.round(mpo_evo_lhs.evolved_time, 8):
                mpo_evo_rhs.run_evolution(1, dt_rhs)
            mpo_evo_lhs.run_evolution(1, dt_lhs)

        while np.round(mpo_evo_rhs.evolved_time, 8) < np.round(mpo_evo_lhs.evolved_time, 8):
            mpo_evo_rhs.run_evolution(1, dt_rhs)

        np.testing.assert_almost_equal(
            common_state.overlap(initial_state), 0.99600363 - 0.00497457j
        )
