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


import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit_addon_mpf.backends import HAS_QUIMB

if HAS_QUIMB:
    from qiskit_addon_mpf.backends.quimb_layers import LayerModel


@pytest.mark.skipif(not HAS_QUIMB, reason="Quimb is required for these unittests")
class TestLayerModel:
    def test_from_quantum_circuit(self):
        L = 6

        qc = QuantumCircuit(L)
        for i in range(0, L - 1, 2):
            qc.rzz(1.0, i, i + 1)
        for i in range(L):
            qc.rz(1.0, i)
        for i in range(1, L - 1, 2):
            qc.append(XXPlusYYGate(1.0), [i, i + 1])

        qc = qc.repeat(2).decompose()

        model = LayerModel.from_quantum_circuit(qc)
        expected_terms = {
            (0, 1): np.array(
                [
                    [7.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -3.0, 0.0],
                    [0.0, 0.0, 0.0, -5.0],
                ]
            ),
            (2, 3): np.array(
                [
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -3.0],
                ]
            ),
            (1, 2): np.array(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -4.0],
                ]
            ),
            (4, 5): np.array(
                [
                    [7.0, 0.0, 0.0, 0.0],
                    [0.0, -3.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, -5.0],
                ]
            ),
            (3, 4): np.array(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -4.0],
                ]
            ),
        }
        for sites in model.terms:
            np.testing.assert_allclose(model.terms[sites], expected_terms[sites])

    def test_handling_unsupportedown_gate(self):
        qc = QuantumCircuit(1)
        qc.rx(1.0, 0)
        with pytest.raises(NotImplementedError):
            LayerModel.from_quantum_circuit(qc)
