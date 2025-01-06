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


import pytest
from qiskit.circuit import QuantumCircuit
from qiskit_addon_mpf.backends import HAS_TENPY

if HAS_TENPY:
    from qiskit_addon_mpf.backends.tenpy_layers import LayerModel, LayerwiseEvolver
    from qiskit_addon_mpf.backends.tenpy_tebd import MPOState


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
