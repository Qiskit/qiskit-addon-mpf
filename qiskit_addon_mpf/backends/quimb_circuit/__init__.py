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

"""A circuit-based time-evolution backend using :external:mod:`quimb`.

.. currentmodule:: qiskit_addon_mpf.backends.quimb_circuit

.. warning::
   This backend is only available if the optional dependencies have been installed:

   .. code-block::

      pip install "qiskit-addon-mpf[quimb]"

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:
   :template: autosummary/class_without_inherited_members.rst

   CircuitEvolver
   CircuitState


Underlying method
-----------------

Quimb boasts direct support for the simulation of quantum circuits in the form of its tensor-network
based :external:class:`quimb.tensor.Circuit` representation. We can leverage this, to bypass any
explicit time-evolution algorithm and instead directly encode the time-evolution in a
:external:class:`~qiskit.circuit.QuantumCircuit` and use :external:mod:`quimb` to compute the
overlap between two such circuits. For more information, check out their guide on
:external:std:doc:`tensor-circuit`.


Code example
------------

This section shows a simple example to get you started with using this backend. The example shows
how to create the three factory functions required for the :func:`.setup_dynamic_lse`.

The :class:`.IdentityStateFactory` protocol is already fulfilled by the
:class:`~.quimb_circuit.CircuitState` constructor, rendering the ``identity_factory`` argument
trivial:

>>> from qiskit_addon_mpf.backends.quimb_circuit import CircuitState
>>> identity_factory = CircuitState

The setup of the :class:`~.quimb_circuit.CircuitEvolver` is slightly more involved. It requires a
**parameterized** :external:class:`~qiskit.circuit.QuantumCircuit` object as its input where the
:class:`~qiskit.circuit.Parameter` should take the place of the Trotter methods time step (``dt``).

To show how such a parameterized Trotter circuit template is constructed, we reuse the same
Hamiltonian and second-order Suzuki-Trotter formula as in :mod:`.quimb_layers`.

>>> from qiskit.quantum_info import SparsePauliOp
>>> hamil = SparsePauliOp.from_sparse_list(
...     [("ZZ", (i, i+1), 1.0) for i in range(0, 9, 2)] +
...     [("Z", (i,), 0.5) for i in range(10)] +
...     [("ZZ", (i, i+1), 1.0) for i in range(1, 9, 2)] +
...     [("X", (i,), 0.25) for i in range(10)],
...     num_qubits=10,
... )

But this time, we specify a :class:`~qiskit.circuit.Parameter` as the ``time`` argument when
constructing the actual circuits.

>>> from functools import partial
>>> from qiskit.circuit import Parameter
>>> from qiskit.synthesis import SuzukiTrotter
>>> from qiskit_addon_mpf.backends.quimb_circuit import CircuitEvolver
>>> from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
>>> dt = Parameter("dt")
>>> suzuki_2 = generate_time_evolution_circuit(hamil, time=dt, synthesis=SuzukiTrotter(order=2))
>>> approx_evolver_factory = partial(CircuitEvolver, circuit=suzuki_2)

.. caution::
   It is **necessary** that the name of the :class:`~qiskit.circuit.Parameter` is ``dt``!

We can choose a higher order Trotter formula for the ``exact_evolver_factory``. But note, that we
must once again use a parameterized circuit, even if we immediately bind its parameter when
constructing the ``partial`` function.

>>> suzuki_4 = generate_time_evolution_circuit(hamil, time=dt, synthesis=SuzukiTrotter(order=4))
>>> exact_evolver_factory = partial(CircuitEvolver, circuit=suzuki_4, dt=0.05)

These factory functions may now be used to run the :func:`.setup_dynamic_lse`. Refer to its
documentation for more details on that.
"""

# ruff: noqa: E402
from .. import HAS_QUIMB

HAS_QUIMB.require_now(__name__)

from .evolver import CircuitEvolver
from .state import CircuitState

__all__ = [
    "CircuitEvolver",
    "CircuitState",
]
