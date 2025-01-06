# This code is a Qiskit project.
#
# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A layer-wise time-evolver using TeNPy."""

from __future__ import annotations

from tenpy.linalg import TruncationError

from .. import tenpy_tebd
from .model import LayerModel


class LayerwiseEvolver(tenpy_tebd.TEBDEvolver):
    """A special case of the :class:`~.tenpy_tebd.TEBDEvolver` based on layer-wise evolution models.

    As also explained in :mod:`.tenpy_layers`, this implementation extracts the alternating even/odd
    bond updates implemented inside of the original
    :external:class:`~tenpy.algorithms.tebd.TEBDEngine` to become the end users responsibility. It
    does so, by replacing the single Hamiltonian provided to the :class:`~.tenpy_tebd.TEBDEvolver`
    instance with a sequence of :class:`~.tenpy_layers.LayerModel` instances. Every single instance
    of these encodes a single **layer** of interactions. These should enforce the alternating
    updates of even and odd bonds of the underlying tensor network.

    The motivation for this more complicated interface is that is provides a lot more flexbility and
    enables users to define custom Trotter product formulas rather than being limited to the ones
    implemented by TeNPy directly.
    """

    def __init__(
        self,
        evolution_state: tenpy_tebd.MPOState,
        layers: list[LayerModel],
        *args,
        **kwargs,
    ) -> None:
        """Initialize a :class:`LayerwiseEvolver` instance.

        Args:
            evolution_state: forwarded to :class:`~.tenpy_tebd.TEBDEvolver`. Please refer to its
                documentation for more details.
            layers: the list of models describing single layers of interactions. See above as well
                as the explanations provided in :mod:`.tenpy_layers`.
            args: any further positional arguments will be forwarded to the
                :class:`~.tenpy_tebd.TEBDEvolver` constructor.
            kwargs: any further keyword arguments will be forwarded to the
                :class:`~.tenpy_tebd.TEBDEvolver` constructor.
        """
        super().__init__(evolution_state, layers[0], *args, **kwargs)
        self.layers = layers
        """The layers of interactions used to implement the time-evolution."""

    def evolve(self, N_steps: int, dt: float) -> TruncationError:
        # pylint: disable=attribute-defined-outside-init
        # NOTE: Somehow, pylint does not properly pick up on the attributes defined in the external
        # base classes.
        """Perform a single time step of TEBD.

        Args:
            N_steps: should always be ``1`` for this time-evolver, otherwise an error will be raised
                (see below).
            dt: the time-step to use.

        Returns:
            The truncation error.

        Raises:
            RuntimeError: if ``N_steps`` is not equal to ``1``.
        """
        if N_steps != 1:
            raise RuntimeError(
                "The LayerwiseEvolver only supports a single evolution step at a time!"
            )

        if dt is not None:  # pragma: no branch
            assert dt == self._U_param["delta_t"]

        trunc_err = TruncationError()
        for U_idx_dt in range(len(self.layers)):
            Us = self._U[U_idx_dt]
            for i_bond in range(self.psi.L):
                if Us[i_bond] is None:
                    # NOTE: in the original TeNPy implementation this handles finite vs. infinite
                    # boundary conditions
                    # NOTE: here, we leverage the same principle to automatically skip even and odd
                    # bond indices based on the LayerModel.calc_H_bond output
                    continue
                self._update_index = (U_idx_dt, i_bond)
                update_err = self.update_bond(i_bond, Us[i_bond])
                trunc_err += update_err
        self._update_index = None  # type: ignore[assignment]
        self.evolved_time = self.evolved_time + N_steps * self._U_param["tau"]
        self.trunc_err: TruncationError = self.trunc_err + trunc_err  # not += : make a copy!
        # (this is done to avoid problems of users storing self.trunc_err after each `evolve`)
        return trunc_err

    @staticmethod
    def suzuki_trotter_time_steps(order: int) -> list[float]:  # pylint: disable=unused-argument
        """Returns an empty list.

        .. note::
           This method is undefined for this subclass but we cannot raise an error upon calling
           it because of the internal algorithm flow. Instead, the Trotter decomposition in this
           class is encoded directly into the :attr:`layers`.

        Args:
            order: is being ignored.

        Returns:
            An empty list.
        """
        return []

    @staticmethod
    def suzuki_trotter_decomposition(
        order: int,  # pylint: disable=unused-argument
        N_steps: int,  # pylint: disable=unused-argument
    ) -> list[tuple[int, int]]:
        """Returns an empty list.

        .. note::
           This method is undefined for this subclass but we cannot raise an error upon calling
           it because of the internal algorithm flow. Instead, the Trotter decomposition in this
           class is encoded directly into the :attr:`layers`.

        Args:
            order: is being ignored.
            N_steps: is being ignored.

        Returns:
            An empty list.
        """
        return []  # pragma: no cover

    def calc_U(
        self,
        order: int,
        delta_t: float,
        type_evo: str = "real",
        E_offset: list[float] | None = None,
    ) -> None:
        # pylint: disable=attribute-defined-outside-init
        # NOTE: Somehow, pylint does not properly pick up on the attributes defined in the external
        # base classes.
        """Calculates the local bond updates.

        This adapts :external:meth:`~tenpy.algorithms.tebd.TEBDEngine.calc_U` to work with the
        layer-wise implementation.

        Args:
            order: this is being ignored.
            delta_t: the time-step to use.
            type_evo: the type of time-evolution. Imaginary time-evolution is not supported at this
                time.
            E_offset: a constant energy offset to be applied.
        """
        super().calc_U(order, delta_t, type_evo=type_evo, E_offset=E_offset)

        # NOTE: since suzuki_trotter_time_steps returns an empty list, we did not yet compute
        # self._U in the super-call above and do so manually, here
        L = self.psi.L
        self._U = []
        for layer in self.layers:
            self.model = layer
            U_bond = [
                self._calc_U_bond(i_bond, delta_t, type_evo, E_offset)
                for i_bond in range(L)
                # NOTE: this iteration over L implies a 1D chain!
            ]
            self._U.append(U_bond)
