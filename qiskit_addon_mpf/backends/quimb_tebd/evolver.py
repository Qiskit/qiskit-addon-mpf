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

"""A quimb-based TEBD algorithm for evolving an internal MPO."""

from __future__ import annotations

from typing import Literal, cast

from quimb.tensor import TEBD

from .. import Evolver
from .state import MPOState


class TEBDEvolver(TEBD, Evolver):
    """A TEBD algorithm for evolving an internal MPO.

    As discussed in more detail in :mod:`~qiskit_addon_mpf.backends.quimb_tebd`, this extension of
    quimb's existing :external:class:`quimb.tensor.TEBD` implementation time-evolves an internal
    matrix product operator (MPO) rather than the conventional matrix product state (MPS).

    More concretely, the internal object is expected to be an :class:`~.quimb_tebd.MPOState`.

    .. warning::
       The API of this class is actually much larger than shown here, because it inherits additional
       methods from the :external:class:`quimb.tensor.TEBD` base class. However, we do not duplicate
       that API here.
    """

    def __init__(self, evolution_state: MPOState, *args, order: int = 2, **kwargs) -> None:
        """Initialize a :class:`TEBDEvolver` instance.

        Args:
            evolution_state: a reference to the time-evolution state. This overwrites the ``p0``
                argument of the underlying :external:class:`quimb.tensor.TEBD` class.

                .. warning::
                   In contrast to the default behavior, this state will **NOT** be canonicalized.
                   Instead, it is taken as is and is kept **by reference** (i.e. no copy is
                   created). This ensures that the same object can be shared between two instances
                   of this class, as required by the :class:`.DynamicMPF` algorithm.

            args: any further positional arguments will be forwarded to the
                :external:class:`quimb.tensor.TEBD` constructor.
            order: the order of the builtin Suzuki-Trotter formula to use during time evolution.
                This will be the value forwarded to the :external:meth:`quimb.tensor.TEBD.step`
                method.
            kwargs: any further keyword arguments will be forwarded to the
                :external:class:`quimb.tensor.TEBD` constructor.
        """
        super().__init__(evolution_state, *args, **kwargs)
        # WARNING: we must forcefully overwrite self._pt to ensure that p0 is kept by reference!
        self._pt = evolution_state
        self._order = order
        self._conjugate = False

    @property
    def evolved_time(self) -> float:
        """Returns the current evolution time."""
        return cast(float, self.t)

    @property
    def conjugate(self) -> bool:
        """Returns whether this time-evolver instance acts on the right-hand side."""
        return self._conjugate

    @conjugate.setter
    def conjugate(self, conjugate: bool) -> None:
        self._conjugate = conjugate

    def step(self) -> None:
        """Perform a single time step of TEBD.

        This essentially calls :external:meth:`quimb.tensor.TEBD.step` and forwards the value of
        the ``order`` attribute that was provided upon construction.
        """
        TEBD.step(self, order=self._order)

    def sweep(
        self,
        direction: Literal["left", "right"],
        dt_frac: float,
        dt: float | None = None,
        queue: bool = False,
    ) -> None:
        """Perform a single sweep of the TEBD algorithm [1].

        The TEBD algorithm updates the even and odd bonds of the underlying tensor network in
        alternating fashion. In the implementation of the :external:class:`quimb.tensor.TEBD` base
        class, this is realized in the form of alternating "sweeps" in left and right directions
        over the internal state.

        We are overwriting the behavior of this method in this subclass, in order to call the
        specialized :meth:`~.quimb_tebd.MPOState.gate_split` method.

        Args:
            direction: the direction of the sweep. This must be either of the literal strings,
                ``"left"`` or ``"right"``.
            dt_frac: what fraction of the internal time step (``dt``) to time-evolve for. This is
                how any builtin Suzuki-Trotter formula specifies its splitting.
            dt: an optional value to overwrite the internal time step.
            queue: setting this to ``True`` will raise a :class:`NotImplementedError`.

        Raises:
            NotImplementedError: if ``queue=True``.
            NotImplementedError: if :external:attr:`~quimb.tensor.TEBD.cyclic` is ``True``.
            NotImplementedError: if :external:attr:`~quimb.tensor.TEBD.imag` is ``True``.
            RuntimeError: if an invalid ``direction`` is provided.

        References:
            [1]: https://en.wikipedia.org/wiki/Time-evolving_block_decimation
        """
        if queue:
            raise NotImplementedError(  # pragma: no cover
                "The TEBDEvolver does not support queued operations!"
            )

        if self.cyclic:
            raise NotImplementedError(  # pragma: no cover
                "The TEBDEvolver does not support PBCs!"
            )

        if self.imag:
            raise NotImplementedError(  # pragma: no cover
                "The TEBDEvolver does not support imaginary time!"
            )

        # if custom dt set, scale the dt fraction
        if dt is not None:
            dt_frac *= dt / self._dt  # pragma: no cover

        final_site_ind = self.L - 1
        if direction == "right":
            for i in range(0, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                gate = self._get_gate_from_ham(dt_frac, sites)
                self._pt.gate_split_(gate, sites, conj=self.conjugate, **self.split_opts)

        elif direction == "left":
            for i in range(1, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                gate = self._get_gate_from_ham(dt_frac, sites)
                self._pt.gate_split_(gate, sites, conj=self.conjugate, **self.split_opts)

        else:
            # NOTE: it should not be possible to reach this but we do a sanity check to ensure that
            # no faulty direction was provided to this algorithm for an unknown reason
            raise RuntimeError(  # pragma: no cover
                f"Expected the direction to be 'left' or 'right', not {direction}!"
            )
