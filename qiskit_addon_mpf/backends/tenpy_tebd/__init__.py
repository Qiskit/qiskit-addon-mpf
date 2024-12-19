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

"""A :external:mod:`tenpy`-based TEBD backend.

.. currentmodule:: qiskit_addon_mpf.backends.tenpy_tebd

.. caution::
   The optional dependency `TeNPy <https://github.com/tenpy/tenpy>`_ was previously offered under a
   GPLv3 license.
   As of the release of `v1.0.4 <https://github.com/tenpy/tenpy/releases/tag/v1.0.4>`_ on October
   2nd, 2024, it has been offered under the Apache v2 license.
   The license of this package is only compatible with Apache-licensed versions of TeNPy.

.. warning::
   This backend is only available if the optional dependencies have been installed:

   .. code-block::

      pip install "qiskit-addon-mpf[tenpy]"

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:
   :template: autosummary/class_without_inherited_members.rst

   TEBDEvolver
   MPOState
   MPS_neel_state

Underlying method
-----------------

This module provides a time-evolution backend for computing dynamic MPF coefficients based on the
time-evolving block decimation (TEBD) algorithm [1] implemented in the :external:mod:`tenpy` tensor
network library.

The classes provided by this module serve two purposes:

1. Connecting :external:mod:`tenpy`'s implementation to the interface set out by
   :mod:`qiskit_addon_mpf.backends`.
2. Extending :external:mod:`tenpy`'s TEBD implementation to handle an internal MPO (rather than
   MPS) state (see also :class:`.State` for more details).


In the simplest sense, this module provides a straight-forward extension of the TEBD algorithm to
evolve an internal MPO state.
As such, if you wish to use this backend for your dynamic MPF algorithm, you must encode the
Hamiltonian that you wish to time-evolve, in a :external:mod:`tenpy`-native form. To be more
concrete, the :class:`~qiskit_addon_mpf.backends.tenpy_tebd.TEBDEvolver` class (which is a subclass
of :external:class:`tenpy.algorithms.tebd.TEBDEngine`) works with a Hamiltonian in the form of a
:external:class:`~tenpy.models.model.Model`. TeNPy provides a number of convenience methods for
constructing such Hamiltonians in its :external:mod:`tenpy.models` module.
If none of those fulfill your needs, you can consider using the
:class:`~qiskit_addon_mpf.backends.tenpy_layers.LayerModel` class which implements some conversion
methods from Qiskit-native objects.

Code example
------------

This section shows a simple example to get you started with using this backend. The example shows
how to create the three factory functions required for the :func:`.setup_dynamic_lse`.

First of all, we define the Hamiltonian which we would like to time-evolve. Here, we simply choose
one of :external:mod:`tenpy`'s convenience methods.

>>> from tenpy.models import XXZChain2
>>> hamil = XXZChain2(
...     {
...         "L": 10,
...         "Jz": 0.8,
...         "Jxx": 0.7,
...         "hz": 0.3,
...         "bc_MPS": "finite",
...         "sort_charge": False,
...     }
... )

Next, we can create the ``identity_factory`` which has to match the :class:`.IdentityStateFactory`
protocol. We do so by using the :func:`~.tenpy_tebd.MPOState.initialize_from_lattice` convenience
method which takes the lattice underlying the Hamiltonian which we just defined as its only input.

>>> from functools import partial
>>> from qiskit_addon_mpf.backends.tenpy_tebd import MPOState
>>> identity_factory = partial(MPOState.initialize_from_lattice, hamil.lat),

We can now construct the :class:`.ExactEvolverFactory` and :class:`.ApproxEvolverFactory`
time-evolution instance factories. To do so, we can simply bind the pre-defined values of the
:class:`~qiskit_addon_mpf.backends.tenpy_tebd.TEBDEvolver` initializer, reducing it to the correct
interface as expected by the respective function protocols.

>>> from qiskit_addon_mpf.backends.tenpy_tebd import TEBDEvolver
>>> exact_evolver_factory = partial(
...     TEBDEvolver,
...     model=hamil,
...     dt=0.05,
...     options={
...         "order": 4,
...         "preserve_norm": False,
...     },
... )

Notice, how we have fixed the ``dt`` value to a small time step and have used a higher-order
Suzuki-Trotter decomposition to mimic the exact time-evolution above.

Below, we do not fix the ``dt`` value and use only a second-order Suzuki-Trotter formula for the
approximate time-evolution. Additionally, we also specify some truncation settings.

>>> approx_evolver_factory = partial(
...     TEBDEvolver,
...     model=hamil,
...     options={
...         "order": 2,
...         "preserve_norm": False,
...         "trunc_params": {
...             "chi_max": 10,
...             "svd_min": 1e-5,
...             "trunc_cut": None,
...         },
...     },
... )

Of course, you are not limited to the examples shown here, and we encourage you to play around with
the other settings provided by TeNPy's :external:class:`~tenpy.algorithms.tebd.TEBDEngine`
implementation.

Limitations
-----------

Finally, we point out a few known limitations on what kind of Hamiltonians can be treated by this
backend:

* all interactions must be 1-dimensional
* the interactions must use finite boundary conditions

Resources
---------

[1]: https://en.wikipedia.org/wiki/Time-evolving_block_decimation
"""

# ruff: noqa: E402
from .. import HAS_TENPY

HAS_TENPY.require_now(__name__)

from .evolver import TEBDEvolver
from .state import MPOState, MPS_neel_state

__all__ = [
    "MPOState",
    "MPS_neel_state",
    "TEBDEvolver",
]
