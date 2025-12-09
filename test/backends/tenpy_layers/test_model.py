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
from qiskit_addon_mpf.backends import HAS_TENPY

if HAS_TENPY:
    from qiskit_addon_mpf.backends.tenpy_layers import LayerModel


@pytest.mark.skipif(not HAS_TENPY, reason="Tenpy is required for these unittests")
class TestLayerModel:
    def test_from_quantum_circuit(self):
        L = 6

        qc = QuantumCircuit(L)
        for i in range(0, L - 1, 2):
            qc.rzz(1.0, i, i + 1)
        for i in range(L):
            qc.rz(2.0, i)
        for i in range(1, L - 1, 2):
            qc.append(XXPlusYYGate(1.0), [i, i + 1])

        qc = qc.repeat(2).decompose()

        model = LayerModel.from_quantum_circuit(qc)
        expected_H_bonds = [
            np.array(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, -2.0],
                ]
            ),
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -2.0],
                ]
            ),
            np.array(
                [
                    [3.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, -1.0],
                ]
            ),
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -2.0],
                ]
            ),
            np.array(
                [
                    [4.0, 0.0, 0.0, -2.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -2.0],
                ]
            ),
        ]
        assert model.H_bond[0] is None
        for expected, actual in zip(expected_H_bonds, model.H_bond[1:], strict=True):
            np.testing.assert_allclose(expected, actual.to_ndarray().reshape((4, 4)))

    def test_handling_unsupportedown_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        with pytest.raises(NotImplementedError):
            LayerModel.from_quantum_circuit(qc)
