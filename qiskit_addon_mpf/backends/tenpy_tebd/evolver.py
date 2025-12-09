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

"""A tenpy-based TEBD algorithm for evolving an internal MPO."""

from __future__ import annotations

import logging
from typing import cast

from tenpy import TEBDEngine, svd_theta
from tenpy.linalg import np_conserved as npc
from tenpy.networks import MPS

from .. import Evolver

LOGGER = logging.getLogger(__name__)


class TEBDEvolver(TEBDEngine, Evolver):
    """A TEBD algorithm for evolving an internal MPO.

    As discussed in more detail in :mod:`~qiskit_addon_mpf.backends.tenpy_tebd`, this extension of
    TeNPy's existing :external:class:`~tenpy.algorithms.tebd.TEBDEngine` implementation time-evolves
    an internal matrix product operator (MPO) rather than the conventional matrix product state
    (MPS).

    More concretely, the internal object is expected to be an :class:`~.tenpy_tebd.MPOState`.

    .. warning::
       The API of this class is actually much larger than shown here, because it inherits additional
       methods from the :external:class:`~tenpy.algorithms.tebd.TEBDEngine` base class. However, we
       do not duplicate that API here.
    """

    def __init__(self, *args, dt: float = 0.1, **kwargs) -> None:
        """Initialize a :class:`TEBDEvolver` instance.

        Args:
            args: any positional arguments will be forwarded to the
                :external:class:`~tenpy.algorithms.tebd.TEBDEngine` constructor.
            dt: the time step to be used by this time-evolution instance.
            kwargs: any further keyword arguments will be forwarded to the
                :external:class:`~tenpy.algorithms.tebd.TEBDEngine` constructor.
        """
        super().__init__(*args, **kwargs)
        self.dt = dt
        """The time step to be used by this time-evolution instance."""
        self._conjugate = False

    @property
    def evolved_time(self) -> float:
        """Returns the current evolution time."""
        return self._evolved_time

    @evolved_time.setter
    def evolved_time(self, evolved_time: float) -> None:
        self._evolved_time = evolved_time

    @property
    def conjugate(self) -> bool:
        """Returns whether this time-evolver instance acts on the right-hand side."""
        return self._conjugate

    @conjugate.setter
    def conjugate(self, conjugate: bool) -> None:
        self._conjugate = conjugate

    def step(self) -> None:
        """Perform a single time step of TEBD.

        This essentially calls :external:meth:`~tenpy.algorithms.tebd.TEBDEngine.run_evolution` and
        forwards the value of :attr:`dt` that was provided upon construction.
        """
        self.run_evolution(1, self.dt)

    def update_bond(self, i: int, U_bond: npc.Array) -> float:
        """Update the specified bond.

        Overwrites the original (MPS-based) implementation to support an MPO as the shared state.

        Args:
            i: the bond index.
            U_bond: the bond to update.

        Returns:
            The truncation error.
        """
        i0, i1 = i - 1, i
        LOGGER.debug("Update sites (%d, %d)", i0, i1)

        # NOTE: support for MatrixProductState objects is only added for testing/debugging purposes!
        # This is not meant for consumption by end-users of the `qiskit_addon_mpf.dynamic` module
        # and its use is highly discouraged.
        is_mps = isinstance(self.psi, MPS)

        leg_lbl = "v" if is_mps else "w"
        left_leg = f"{leg_lbl}L"
        right_leg = f"{leg_lbl}R"
        p0s = ("p0",) if is_mps else ("p0", "p0*")
        p1s = ("p1",) if is_mps else ("p1", "p1*")
        ps = ("p",) if is_mps else ("p", "p*")

        # Construct the theta matrix
        C0 = self.psi.get_B(i0) if is_mps else self.psi.get_W(i0)
        C1 = self.psi.get_B(i1) if is_mps else self.psi.get_W(i1)

        C = npc.tensordot(C0, C1, axes=([right_leg], [left_leg]))
        new_labels = [left_leg, *p0s, *p1s, right_leg]
        C.iset_leg_labels(new_labels)

        # apply U to C
        if self.conjugate:
            C = npc.tensordot(U_bond.conj(), C, axes=(["p0", "p1"], ["p0*", "p1*"]))  # apply U
        else:
            C = npc.tensordot(U_bond, C, axes=(["p0*", "p1*"], ["p0", "p1"]))  # apply U

        C.itranspose([left_leg, *p0s, *p1s, right_leg])

        theta = C.scale_axis(self.psi.get_SL(i0) if is_mps else self.psi.Ss[i0], left_leg)
        # now theta is the same as if we had done
        #   theta = self.psi.get_theta(i0, n=2)
        #   theta = npc.tensordot(U_bond, theta, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U
        # but also have C which is the same except the missing "S" on the left
        # so we don't have to apply inverses of S (see below)

        theta = theta.combine_legs([(left_leg, *p0s), (*p1s, right_leg)], qconj=[+1, -1])

        # Perform the SVD and truncate the wavefunction
        _U, S, V, trunc_err, renormalize = svd_theta(
            theta, self.trunc_params, [None, None], inner_labels=[right_leg, left_leg]
        )

        # Split tensor and update matrices
        B_R = V.split_legs(1).ireplace_labels(p1s, ps)

        # In general, we want to do the following:
        #     U = U.iscale_axis(S, 'vR')
        #     B_L = U.split_legs(0).iscale_axis(self.psi.get_SL(i0)**-1, 'vL')
        #     B_L = B_L.ireplace_label('p0', 'p')
        # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**-1 U S``
        #
        # However, the inverse of SL is problematic, as it might contain very small singular
        # values.  Instead, we use ``C == SL**-1 theta == SL**-1 U S V``,
        # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
        # here, C is the same as theta, but without the `S` on the very left
        # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)

        def conj(labels: tuple[str, ...]):
            """Conjugates a tuple of leg labels."""
            return tuple(lbl[:-1] if lbl[-1] == "*" else lbl + "*" for lbl in labels)

        B_L = npc.tensordot(
            C.combine_legs((*p1s, right_leg), pipes=theta.legs[1]),
            V.conj(),
            axes=[f"({'.'.join(p1s)}.{right_leg})", f"({'.'.join(conj(p1s))}.{right_leg}*)"],
        )
        B_L.ireplace_labels([f"{left_leg}*", *p0s], [right_leg, *ps])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1

        if is_mps:
            self.psi.norm *= renormalize
            self.psi.set_SR(i0, S)
            self.psi.set_B(i0, B_L, form="B")
            self.psi.set_B(i1, B_R, form="B")
        else:
            self.psi.Ss[i1] = S
            self.psi.set_W(i0, B_L)
            self.psi.set_W(i1, B_R)

        self._trunc_err_bonds[i0] = self._trunc_err_bonds[i0] + trunc_err

        return cast(float, trunc_err)
