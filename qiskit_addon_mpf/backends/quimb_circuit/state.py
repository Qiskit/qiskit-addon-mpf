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

"""A circuit-based MPO-like time-evolution state based on quimb."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_quimb import quimb_circuit
from quimb.tensor import Circuit, TensorNetwork

from .. import State


class CircuitState(State):
    """An MPO-like representation of a time-evolution state based on quantum circuits.

    This time-evolution state can be evolved on its left- and right-hand side as required by the
    :class:`.DynamicMPF` algorithm.
    """

    def __init__(self) -> None:
        """Initialize a :class:`CircuitState` instance."""
        self.lhs: Circuit | None = None
        """The left-hand side circuit in form of a tensor network."""
        self.rhs: Circuit | None = None
        """The right-hand side circuit in form of a tensor network."""

    def overlap(self, initial_state: Any) -> complex:
        """Compute the overlap of this state with the provided initial state.

        .. warning::
           This implementation only supports instances of
           :external:class:`qiskit.circuit.QuantumCircuit` for ``initial_state``.

        Args:
            initial_state: the initial state with which to compute the overlap.

        Raises:
            TypeError: if the provided initial state has an incompatible type.

        Returns:
            The overlap of this state with the provided one.
        """
        if not isinstance(initial_state, QuantumCircuit):
            raise TypeError(
                "CircuitState.overlap is only implemented for qiskit.QuantumCircuit! "
                "But not for states of type '%s'",
                type(initial_state),
            )

        if self.lhs is None or self.rhs is None:
            raise RuntimeError("You must evolve the state before an overlap can be computed!")

        lhs = quimb_circuit(initial_state)
        lhs.apply_gates(self.lhs.gates)

        rhs = quimb_circuit(initial_state)
        rhs.apply_gates(self.rhs.gates)

        # TODO: find a good way to inject arguments into .contract() below
        # For example, specifying backend="jax" would allow us to run this on a GPU (if available
        # and installed properly).
        ovlp = TensorNetwork((lhs.psi.H, rhs.psi)).contract()

        return float(np.abs(ovlp) ** 2)
