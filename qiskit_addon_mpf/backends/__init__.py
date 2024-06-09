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

"""Optional backends for the :class:`.DynamicMPF` algorithm.

.. currentmodule:: qiskit_addon_mpf.backends

Availability
------------

Whether a certain backend can be used depends on the availability of the underlying tensor network
library. This can easily be asserted at runtime using the following indicators:

.. autoclass:: HAS_QUIMB

.. autoclass:: HAS_TENPY

Backends
--------

Depending on the availability (see above), the following backends are available:

.. autosummary::
   :toctree:

   quimb_tebd
   quimb_layers
   quimb_circuit
   tenpy_tebd
   tenpy_layers

Interface
---------

The interface implemented by any one of these optional backends is made up of the following classes:

.. autoclass:: Evolver

.. autoclass:: State
"""

from qiskit.utils import LazyImportTester as _LazyImportTester

from .interface import Evolver, State

HAS_QUIMB = _LazyImportTester("quimb", install="pip install qiskit-addon-mpf[quimb]")
"""Indicates whether the optional :external:mod:`quimb` dependency is installed."""

HAS_TENPY = _LazyImportTester("tenpy", install="pip install qiskit-addon-mpf[tenpy]")
"""Indicates whether the optional :external:mod:`tenpy` dependency is installed."""

__all__ = [
    "Evolver",
    "State",
]
