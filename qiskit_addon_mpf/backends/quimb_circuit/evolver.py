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

"""A time-evolution engine based on quantum circuits."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit_quimb import quimb_circuit

from .. import Evolver
from .state import CircuitState


class CircuitEvolver(Evolver):
    """A time-evolution engine based on quantum circuits.

    This algorithm performs time-evolution by means of successively applying a quantum circuit
    corresponding to a single Trotter step to its internal state. More specifically, it builds out a
    tensor network in the :class:`~.quimb_circuit.CircuitState`. As required by the
    :class:`.DynamicMPF` algorithm, it tracks a left- and right-hand side of the time-evolution for
    computing the overlap of two circuits. Depending on :attr:`conjugate`, an instance of this
    engine will apply the quantum gates of its template circuit to the corresponding side (see
    :mod:`.quimb_circuit` for more details).
    """

    def __init__(self, evolution_state: CircuitState, circuit: QuantumCircuit, dt: float) -> None:
        """Initialize a :class:`CircuitEvolver` instance.

        Args:
            evolution_state: a reference to the time-evolution state.
            circuit: the template circuit encoding the time-evolution of a single Trotter step. This
                circuit **must** be parametrized (see :external:class:`~qiskit.circuit.Parameter` in
                place of the Trotter methods time step. This parameter must be named ``dt``.
            dt: the time step that will be used and later bound to the
                :external:class:`~qiskit.circuit.Parameter` of the ``circuit`` object.
        """
        self.evolution_state = evolution_state
        """The time-evolution state (see also :attr:`.DynamicMPF.evolution_state`)."""
        self.evolved_time = 0
        self.dt = dt
        self.circuit = quimb_circuit(circuit.assign_parameters({"dt": dt}, inplace=False))
        """The parameterized :external:class:`~qiskit.circuit.QuantumCircuit` describing the Trotter
        step."""
        self._conjugate = False

    @property
    def conjugate(self) -> bool:
        """Returns whether this time-evolver instance acts on the right-hand side."""
        return self._conjugate

    @conjugate.setter
    def conjugate(self, conjugate: bool) -> None:
        self._conjugate = conjugate

    @property
    def evolved_time(self) -> float:
        """Returns the current evolution time."""
        return self._evolved_time

    @evolved_time.setter
    def evolved_time(self, evolved_time: float) -> None:
        self._evolved_time = evolved_time

    def step(self) -> None:
        """Perform a single time step of TEBD.

        This will apply the gates of the :attr:`circuit` to the :attr:`evolution_state`. If
        :attr:`conjugate` is ``True``, it applies to :attr:`.CircuitState.lhs`, otherwise to
        :attr:`.CircuitState.rhs`.
        """
        self.evolved_time += self.dt

        if self.conjugate:
            if self.evolution_state.lhs is None:
                self.evolution_state.lhs = self.circuit.copy()
            else:
                self.evolution_state.lhs.apply_gates(self.circuit.gates)
        else:
            if self.evolution_state.rhs is None:
                self.evolution_state.rhs = self.circuit.copy()
            else:
                self.evolution_state.rhs.apply_gates(self.circuit.gates)
