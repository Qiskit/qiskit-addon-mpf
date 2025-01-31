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

"""A layer-wise time-evolver using quimb."""

from __future__ import annotations

from quimb.tensor import MatrixProductState

from .. import quimb_tebd
from .model import LayerModel


class LayerwiseEvolver(quimb_tebd.TEBDEvolver):
    """A special case of the :class:`~.quimb_tebd.TEBDEvolver` based on layer-wise evolution models.

    As also explained in :mod:`.quimb_layers`, this implementation extracts the alternating even/odd
    bond updates implemented inside of the original :external:class:`quimb.tensor.TEBD` to become
    the end users responsibility. It does so, by replacing the single Hamiltonian provided to the
    :class:`~.quimb_tebd.TEBDEvolver` instance with a sequence of :class:`~.quimb_layers.LayerModel`
    instances. Every single instance of these encodes a single **layer** of interactions. These
    should enforce the alternating updates of even and odd bonds of the underlying tensor network.

    The motivation for this more complicated interface is that is provides a lot more flexbility and
    enables users to define custom Trotter product formulas rather than being limited to the ones
    implemented by ``quimb`` directly.
    """

    def __init__(
        self,
        evolution_state: quimb_tebd.MPOState | MatrixProductState,
        layers: list[LayerModel],
        *args,
        **kwargs,
    ) -> None:
        """Initialize a :class:`LayerwiseEvolver` instance.

        Args:
            evolution_state: forwarded to :class:`~.quimb_tebd.TEBDEvolver`. Please refer to its
                documentation for more details.
            layers: the list of models describing single layers of interactions. See above as well
                as the explanations provided in :mod:`.quimb_layers`.
            args: any further positional arguments will be forwarded to the
                :class:`~.quimb_tebd.TEBDEvolver` constructor.
            kwargs: any further keyword arguments will be forwarded to the
                :class:`~.quimb_tebd.TEBDEvolver` constructor.
        """
        super().__init__(evolution_state, layers[0], *args, **kwargs)
        self.layers = layers
        """The layers of interactions used to implement the time-evolution."""

    def step(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        # NOTE: Somehow, pylint does not properly pick up on the attributes defined in the external
        # base classes.
        """Perform a single time step of TEBD.

        This will iterate over the :attr:`layers` and apply their interaction to the internal state.
        """
        dt = self._dt

        # NOTE: support for MatrixProductState objects is only added for testing/debugging purposes!
        # This is not meant for consumption by end-users of the `qiskit_addon_mpf.dynamic` module
        # and its use is highly discouraged.
        is_mps = isinstance(self._pt, MatrixProductState)

        for layer in self.layers:
            self.H = layer
            for i in range(self.L):
                sites = (i, (i + 1) % self.L)
                gate = self._get_gate_from_ham(1.0, sites)
                if gate is None:
                    continue
                if is_mps:
                    self._pt.gate_split_(gate, sites, **self.split_opts)
                else:
                    self._pt.gate_split_(gate, sites, conj=self.conjugate, **self.split_opts)

        self.t += dt
        self._err += float("NaN")
