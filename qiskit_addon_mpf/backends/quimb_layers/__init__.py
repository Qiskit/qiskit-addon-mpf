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

"""A layer-wise time-evolution backend using :external:mod:`quimb`.

.. currentmodule:: qiskit_addon_mpf.backends.quimb_layers

.. warning::
   This backend is only available if the optional dependencies have been installed:

   .. code-block::

      pip install "qiskit-addon-mpf[quimb]"

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:
   :template: autosummary/class_without_inherited_members.rst

   LayerwiseEvolver
   LayerModel


Underlying method
-----------------

This module provides a time-evolution backend similar to the TEBD-based one provided by the
:mod:`~qiskit_addon_mpf.backends.quimb_tebd` module. The main difference is that this module gives
the user full flexibility for defining their product formulas, thereby not limiting them to the
options built into the :external:mod:`quimb` library.

At its core, the algorithm provided by this module is still a TEBD [1] algorithm. However, rather
than enforcing the alternating updates to the even and odd bonds of the time-evolution state (see
also :meth:`.quimb_tebd.TEBDEvolver.sweep`) this implementation outsources the responsibility of
updating bonds in alternating fashion to the definition of multiple time-evolution **layers**.

This is best explained with an example. Let us assume, we have some generic Hamiltonian acting on a
1-dimensional chain of sites.

.. hint::
   Below we are very deliberate about the order of the Hamiltonian's Pauli terms because this
   directly impacts the structure of the time-evolution circuit later on.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit.quantum_info import SparsePauliOp
   >>> hamil = SparsePauliOp.from_sparse_list(
   ...     [("ZZ", (i, i+1), 1.0) for i in range(1, 9, 2)] +
   ...     [("Z", (i,), 0.5) for i in range(10)] +
   ...     [("ZZ", (i, i+1), 1.0) for i in range(0, 9, 2)],
   ...     num_qubits=10,
   ... )

Let us now inspect the time-evolution circuit of this Hamiltonian using a second-order
Suzuki-Trotter formula.

.. plot::
   :alt: Output from the previous code.
   :context: close-figs
   :include-source:

   >>> from qiskit.synthesis import SuzukiTrotter
   >>> from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
   >>> circuit = generate_time_evolution_circuit(hamil, time=1.0, synthesis=SuzukiTrotter(order=2))
   >>> circuit.draw("mpl")  # doctest: +ELLIPSIS
   <Figure size 956...x869... with 1 Axes>

In the circuit above, we can clearly identify its layer-wise structure. We can emphasize this
further, by splitting the circuit into multiple layers as shown below (we recombine the ``layers``
into a single circuit with barriers between them to ease the visualization).

.. plot::
   :alt: Output from the previous code.
   :context: close-figs
   :include-source:

   >>> from qiskit_addon_utils.slicing import combine_slices, slice_by_gate_types
   >>> layers = slice_by_gate_types(circuit)
   >>> combine_slices(layers, include_barriers=True).draw("mpl")  # doctest: +ELLIPSIS
   <Figure size 1374...x869... with 1 Axes>

.. hint::
   The asymmetry of the central layers is a result of the implementation of Qiskit's
   :external:class:`~qiskit.synthesis.SuzukiTrotter` formula. In its second-order form, it combines
   the two half-time evolutions of the final term in the Hamiltonian into a single one of twice the
   length. We could transpile this circuit to collapse all such subequent gates in the central two
   layers (just like the last one), but for the sake of simplicity of this example, we will not do
   that here.

It is not possible to instruct Quimb's TEBD algorithm to simulate the exact structure of the circuit
shown above. The reason for that is a limitation in its interface, as it only accepts the full
Hamiltonian to be provided which is then time-evolved using pre-defined Trotter formulas. However,
in doing so it does not treat the order of the Pauli terms in a Hamiltonian with any significance
(like we do here).

If one wants to compute the dynamic MPF coefficients of a time-evolution employing a product formula
structure other than the ones implemented in Quimb (like the example above), then one can use the
time-evolution algorithm provided by this module. Rather than taking a single monolithic Hamiltonian
whose time-evolution is to be modeled, the :class:`~.quimb_layers.LayerwiseEvolver` accepts a list
of :class:`~.quimb_layers.LayerModel` objects, each one describing an individual layer of the
product formula. This gives the user full flexibility in defining the Trotter decomposition down to
the most granular level.

However, caution must be applied to ensure that the property of TEBD to update even and odd bonds in
an alternating manner is still guaranteed. Luckily, for quantum circuits consisting of at most
two-qubit gates, this property is satisfied by construction.


Code example
------------

In this section, we build up on the example above and show how to take a custom Trotter formula and
use it to construct a :class:`~.quimb_layers.LayerwiseEvolver` which can be used to replace the
:class:`.quimb_tebd.TEBDEvolver` in the workflow described in :mod:`.quimb_tebd`.

.. hint::
   The overall workflow of using this module is the same as of the :mod:`.quimb_tebd` module, so be
   sure to read those instructions as well.

Simply put, we must convert each one of the circuit ``layers`` (see above) into a
:class:`~.quimb_layers.LayerModel` instance. For this purpose, we can use its
:meth:`~.quimb_layers.LayerModel.from_quantum_circuit` method.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit_addon_mpf.backends.quimb_layers import LayerModel
   >>> model0 = LayerModel.from_quantum_circuit(layers[0])
   >>> layer_models = [model0]

In the code above you can see how simple the conversion is for layers which contain only two-qubit
gates acting on mutually exclusive qubits (which layers of depth 1 guarantee).

However, we must be more careful with layers including single-qubit gates. The reason for that is
that the TEBD algorithm underlying the :class:`~.quimb_layers.LayerwiseEvolver` must update even and
odd bonds in an alternating manner. And because single-qubit gates are not applied on a site, but
instead are split in half and applied to the bonds on either side, a layer of single-qubit gates
acting on all qubits would break this assumption.

To circumvent this problem, we can take any layer consisting of only single-qubit gates, and apply
twice (once on the even and once on the odd bonds).

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> model1a = LayerModel.from_quantum_circuit(layers[1], keep_only_odd=True)
   >>> model1b = LayerModel.from_quantum_circuit(layers[1], keep_only_odd=False)
   >>> layer_models.extend([model1a, model1b])

Now that we know how to treat layers consisting of two-qubit and single-qubit gates, we can
transform the remaining layers.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> for layer in layers[2:]:
   ...     num_qubits = len(layer.data[0].qubits)
   ...     if num_qubits == 2:
   ...         layer_models.append(LayerModel.from_quantum_circuit(layer))
   ...     else:
   ...         layer_models.append(
   ...             LayerModel.from_quantum_circuit(layer, keep_only_odd=True)
   ...         )
   ...         layer_models.append(
   ...             LayerModel.from_quantum_circuit(layer, keep_only_odd=False)
   ...         )
   >>> assert len(layer_models) == 8

In the end, we have 8 :class:`~.quimb_layers.LayerModel`'s, one for each of the 4 two-qubit layers,
and two for each of the 2 single-qubit layers.

Finally, we can define our :class:`.ApproxEvolverFactory` protocol to be used within the
:func:`.setup_dynamic_lse` function.

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from functools import partial
   >>> from qiskit_addon_mpf.backends.quimb_layers import LayerwiseEvolver
   >>> approx_evolver_factory = partial(
   ...     LayerwiseEvolver,
   ...     layers=layer_models,
   ...     split_opts={"max_bond": 10, "cutoff": 1e-5},
   ... )

.. caution::
   It should be noted, that in this workflow we have not yet fixed the time step used by the Trotter
   formula. We have also only set up a single repetition of the Trotter formula as the rest will be
   done by the internal :class:`.DynamicMPF` algorithm, executed during :func:`.setup_dynamic_lse`.

Of course, you could also use this to specify a :class:`.ExactEvolverFactory`. But you can also
mix-and-match a :class:`.quimb_layers.LayerwiseEvolver` with a :class:`.quimb_tebd.TEBDEvolver`.


Resources
---------

[1]: https://en.wikipedia.org/wiki/Time-evolving_block_decimation
"""

# ruff: noqa: E402
from .. import HAS_QUIMB

HAS_QUIMB.require_now(__name__)

from .evolver import LayerwiseEvolver
from .model import LayerModel

__all__ = [
    "LayerModel",
    "LayerwiseEvolver",
]
