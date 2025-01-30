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

"""A quimb-based model for describing a single layer of interactions."""

from __future__ import annotations

from typing import cast

import numpy as np
from qiskit.circuit import QuantumCircuit
from quimb.gen.operators import pauli
from quimb.tensor import LocalHam1D


class LayerModel(LocalHam1D):
    """A model for representing a layer of time-evolution interactions.

    Essentially, this class is a simple wrapper of :external:class:`quimb.tensor.LocalHam1D`. Its
    main purpose is to provide a simple interface for constructing a Quimb-compatible Hamiltonian
    from Qiskit objects.
    """

    def get_gate_expm(self, where: tuple[int, int], x: float) -> np.ndarray | None:
        """Get the local term at the sites ``where``, matrix exponentiated by ``x``.

        Args:
            where: the pair of site indices of the local term to get. This identifies the bond
                index.
            x: the value with which to matrix exponentiate the interaction term.

        Returns:
            The interaction in terms of an array or ``None`` if this layer has no interaction on
            this bond.
        """
        try:
            return cast(np.ndarray, self._expm_cached(self.get_gate(where), x))
        except KeyError:
            return None

    @classmethod
    def from_quantum_circuit(
        cls,
        circuit: QuantumCircuit,
        *,
        keep_only_odd: bool | None = None,
        **kwargs,
    ) -> LayerModel:
        """Construct a :class:`LayerModel` from a :external:class:`~qiskit.circuit.QuantumCircuit`.

        You can see an example of this function in action in the docs of :mod:`quimb_layers`.

        Args:
            circuit: the quantum circuit to parse.
            keep_only_odd: whether to keep only interactions on bonds with odd indices.
            kwargs: any additional keyword arguments to pass to the :class:`LayerModel` constructor.

        Returns:
            A new LayerModel instance.

        Raises:
            NotImplementedError: if an unsupported quantum gate is encountered.
        """
        H2: dict[tuple[int, int] | None, np.ndarray] = {}
        H1: dict[int, np.ndarray] = {}
        paulis_cache: dict[str, np.ndarray] = {}

        for instruction in circuit.data:
            op = instruction.operation
            sites = tuple(circuit.find_bit(qubit)[0] for qubit in instruction.qubits)

            # NOTE: the hard-coded scaling factors below account for the Pauli matrix conversion
            if op.name == "rzz":
                term = paulis_cache.get(op.name, None)
                if term is None:
                    paulis_cache[op.name] = pauli("Z") & pauli("Z")
                    term = paulis_cache[op.name]
                if sites in H2:
                    H2[sites] += 0.5 * op.params[0] * term
                else:
                    H2[sites] = 0.5 * op.params[0] * term
            elif op.name == "xx_plus_yy":
                term = paulis_cache.get(op.name, None)
                if term is None:
                    paulis_cache["rxx"] = pauli("X") & pauli("X")
                    paulis_cache["ryy"] = pauli("Y") & pauli("Y")
                    paulis_cache[op.name] = paulis_cache["rxx"] + paulis_cache["ryy"]
                    term = paulis_cache[op.name]
                if sites in H2:
                    H2[sites] += 0.25 * op.params[0] * term
                else:
                    H2[sites] = 0.25 * op.params[0] * term
            elif op.name == "rz":
                term = paulis_cache.get(op.name, None)
                if term is None:
                    paulis_cache[op.name] = pauli("Z")
                    term = paulis_cache[op.name]
                if sites[0] in H1:
                    H1[sites[0]] += 0.5 * op.params[0] * term
                else:
                    H1[sites[0]] = 0.5 * op.params[0] * term
            else:
                raise NotImplementedError(f"Cannot handle gate of type {op.name}")

        if len(H2) == 0:
            H2[None] = np.zeros((4, 4))

        ret = cls(
            circuit.num_qubits,
            H2,
            H1,
            **kwargs,
        )

        if keep_only_odd is not None:
            # NOTE: if `keep_only_odd` was specified, that means we explicitly overwrite those H_bond
            # values with `None` which we do not want to keep. In the case of (for example) coupling
            # layers, this should have no effect since those bonds were `None` to begin with. However,
            # in the case of onsite layers, this will remove half of the bonds ensuring that we split
            # the bond updates into even and odd parts (as required by the TEBD algorithm).
            for i in range(0 if keep_only_odd else 1, circuit.num_qubits, 2):
                _ = ret.terms.pop((i - 1, i), None)

        return ret
