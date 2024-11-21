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

"""A :external:mod:`quimb`-based TEBD backend.

.. currentmodule:: qiskit_addon_mpf.backends.quimb_tebd

.. warning::
   This backend is only available if the optional dependencies have been installed:

   .. code-block::

      pip install "qiskit-addon-mpf[quimb]"

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:
   :template: autosummary/class_without_inherited_members.rst

   TEBDEvolver
   MPOState

Underlying method
-----------------

This module provides a time-evolution backend for computing dynamic MPF coefficients based on the
time-evolving block decimation (TEBD) algorithm [1] implemented in the :external:mod:`quimb` tensor
network library.

The classes provided by this module serve two purposes:

1. Connecting :external:mod:`quimb`'s implementation to the interface set out by
   :mod:`qiskit_addon_mpf.backends`.
2. Extending :external:mod:`quimb`'s TEBD implementation to handle an internal MPO (rather than
   MPS) state (see also :class:`.State` for more details).


In the simplest sense, this module provides a straight-forward extension of the TEBD algorithm to
evolve an internal MPO state.
As such, if you wish to use this backend for your dynamic MPF algorithm, you must encode the
Hamiltonian that you wish to time-evolve, in a :external:mod:`quimb`-native form. To be more
concrete, the :class:`~qiskit_addon_mpf.backends.quimb_tebd.TEBDEvolver` class (which is a subclass
of :external:class:`quimb.tensor.TEBD`) works with a Hamiltonian in the form of a
:external:class:`quimb.tensor.LocalHam1D`. Quimb provides a number of convenience methods for
constructing such Hamiltonians in its :external:mod:`quimb.tensor.tensor_builder` module.
If none of those fulfill your needs, you can consider using the
:class:`~qiskit_addon_mpf.backends.quimb_layers.LayerModel` class which implements some conversion
methods from Qiskit-native objects.

Code example
------------

This section shows a simple example to get you started with using this backend. The example shows
how to create the three factory functions required for the :func:`.setup_dynamic_lse`.

First, we create the ``identity_factory`` which has to match the :class:`.IdentityStateFactory`
protocol. We do so simply by using the :external:func:`quimb.tensor.MPO_identity` function and
wrapping the resulting :external:class:`quimb.tensor.MatrixProductOperator` with our custom
:class:`~.quimb_tebd.MPOState` interface.

>>> from qiskit_addon_mpf.backends.quimb_tebd import MPOState
>>> from quimb.tensor import MPO_identity
>>> num_qubits = 10
>>> identity_factory = lambda: MPOState(MPO_identity(num_qubits))

Next, before being able to define the :class:`.ExactEvolverFactory` and
:class:`.ApproxEvolverFactory` protocols, we must define the Hamiltonian which we would like to
time-evolve. Here, we simply choose one of :external:mod:`quimb`'s convenience methods.

>>> from quimb.tensor import ham_1d_heis
>>> hamil = ham_1d_heis(num_qubits, 0.8, 0.3, cyclic=False)

We can now construct the exact and approximate time-evolution instance factories. To do so, we can
simply use :func:`functools.partial` to bind the pre-defined values of the
:class:`~qiskit_addon_mpf.backends.quimb_tebd.TEBDEvolver` initializer, reducing it to the correct
interface as expected by the :class:`.ExactEvolverFactory` and :class:`.ApproxEvolverFactory`
protocols, respectively.

>>> from functools import partial
>>> from qiskit_addon_mpf.backends.quimb_tebd import TEBDEvolver
>>> exact_evolver_factory = partial(
...     TEBDEvolver,
...     H=hamil,
...     dt=0.05,
...     order=4,
... )

Notice, how we have fixed the ``dt`` value to a small time step and have used a higher-order
Suzuki-Trotter decomposition to mimic the exact time evolution above.

Below, we do not fix the ``dt`` value and use only a second-order Suzuki-Trotter formula for the
approximate time evolution. Additionally, we also specify some truncation settings.

>>> approx_evolver_factory = partial(
...     TEBDEvolver,
...     H=hamil,
...     order=2,
...     split_opts={"max_bond": 10, "cutoff": 1e-5},
... )

Of course, you are not limited to the examples shown here, and we encourage you to play around with
the other settings provided by the :external:class:`quimb.tensor.TEBD` implementation.

Limitations
-----------

Finally, we point out a few known limitations on what kind of Hamiltonians can be treated by this
backend:

* all interactions must be 1-dimensional
* the interactions must be acylic

Resources
---------

[1]: https://en.wikipedia.org/wiki/Time-evolving_block_decimation
"""

# ruff: noqa: E402
from .. import HAS_QUIMB

HAS_QUIMB.require_now(__name__)

from .evolver import TEBDEvolver
from .state import MPOState

__all__ = [
    "TEBDEvolver",
    "MPOState",
]
