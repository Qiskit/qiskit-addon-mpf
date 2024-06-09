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

    def __init__(self, L, H2, H1=None, cyclic=False, keep_only_odd=None) -> None:
        """Initialize a :class:`LayerModel` instance.

        Most of the arguments below are simply forwarded to
        :external:class:`quimb.tensor.LocalHam1D` so check out its documentation for more details.

        Args:
            L: the number of qubits.
            H2: the two-site interactions.
            H1: the optional on-site interactions.
            cyclic: whether to apply periodic boundary conditions.
            keep_only_odd: whether to keep only odd bond interactions. For more details see
                :attr:`keep_only_odd`.
        """
        super().__init__(L, H2, H1, cyclic)
        self.keep_only_odd = keep_only_odd
        """Whether to keep only interactions on bonds with odd indices."""

    def get_gate_expm(self, where: tuple[int, int], x: float) -> np.ndarray | None:
        """Get the local term at the sites ``where``, matrix exponentiated by ``x``.

        If ``where`` applies to an even bond index and :attr:`keep_only_odd` is ``True``, this
        method will return ``None``.

        Args:
            where: the pair of site indices of the local term to get. This identifies the bond
                index.
            x: the value with which to matrix exponentiate the interaction term.

        Returns:
            The interaction in terms of an array or ``None`` depending on :attr:`keep_only_odd` (see
            above).
        """
        if self.keep_only_odd is not None and where[0] % 2 - self.keep_only_odd:
            return None
        try:
            return cast(np.ndarray, self._expm_cached(self.get_gate(where), x))
        except KeyError:
            return None

    @classmethod
    def from_quantum_circuit(
        cls,
        circuit: QuantumCircuit,
        *,
        scaling_factor: float = 1.0,
        keep_only_odd: bool | None = None,
        **kwargs,
    ) -> LayerModel:
        """Construct a :class:`LayerModel` from a :external:class:`~qiskit.circuit.QuantumCircuit`.

        You can see an example of this function in action in the docs of :mod:`quimb_layers`.

        Args:
            circuit: the quantum circuit to parse.
            scaling_factor: a factor with which to scale the term strengths. This can be used to
                apply (for example) a time step scaling factor. It may also be used (e.g.) to split
                onsite terms into two layers (even and odd) with $0.5$ of the strength, each.
            keep_only_odd: the value to use for :attr:`keep_only_odd`.
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
                    H2[sites] += 0.5 * scaling_factor * op.params[0] * term
                else:
                    H2[sites] = 0.5 * scaling_factor * op.params[0] * term
            elif op.name == "xx_plus_yy":
                term = paulis_cache.get(op.name, None)
                if term is None:
                    paulis_cache["rxx"] = pauli("X") & pauli("X")
                    paulis_cache["ryy"] = pauli("Y") & pauli("Y")
                    paulis_cache[op.name] = paulis_cache["rxx"] + paulis_cache["ryy"]
                    term = paulis_cache[op.name]
                if sites in H2:
                    H2[sites] += 0.25 * scaling_factor * op.params[0] * term
                else:
                    H2[sites] = 0.25 * scaling_factor * op.params[0] * term
            elif op.name == "rz":
                term = paulis_cache.get(op.name, None)
                if term is None:
                    paulis_cache[op.name] = pauli("Z")
                    term = paulis_cache[op.name]
                if sites[0] in H1:
                    H1[sites[0]] += 2.0 * scaling_factor * op.params[0] * term
                else:
                    H1[sites[0]] = 2.0 * scaling_factor * op.params[0] * term
            else:
                raise NotImplementedError(f"Cannot handle gate of type {op.name}")

        if len(H2) == 0:
            H2[None] = np.zeros((4, 4))

        return cls(
            circuit.num_qubits,
            H2,
            H1,
            keep_only_odd=keep_only_odd,
            **kwargs,
        )
