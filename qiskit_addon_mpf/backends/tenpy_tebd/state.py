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

"""A TeNPy-based MPO state."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import Lattice
from tenpy.networks import MPO, MPS

from .. import State


class MPOState(MPO, State):
    """A mediator class to make TeNPy's MPO match the :class:`.State` interface.

    This class simply ensures that a :external:class:`tenpy.networks.mpo.MPO` object can work as a
    :class:`.State` instance.
    """

    @classmethod
    def initialize_from_lattice(cls, lat: Lattice, *, conserve: bool = True) -> MPOState:
        """Construct an identity :class:`MPOState` instance matching the provided lattice shape.

        Given a lattice, this method constructs a new MPO identity matching the shape of the
        lattice.

        Args:
            lat: the lattice describing the MPO sites.
            conserve: whether to conserve ``Sz``. This is a simplified version of the more elaborate
                ``conserve`` property of :class:`~tenpy.networks.site.SpinHalfSite`. The boolean
                value simply indicates ``Sz`` (``True``) or ``None`` conservation (``False``)

        Returns:
            An identity MPO.
        """
        # creates a 4d array filled with zeros - shape 1x2x2x1
        B = np.zeros([1, 2, 2, 1], dtype=float)
        # sets element values of B array to 1
        # creates a tensor that represents identity for MPO
        B[0, 0, 0, 0] = 1
        B[0, 1, 1, 0] = 1

        labels = ["wL", "p", "p*", "wR"]

        if conserve:
            # creates a list of tensor leg charge objects encoding charges + conjugations for tensor
            # legs (i.e. dimensions)
            leg_charge = [
                # e.g. charge information for tensor leg / dimension [1] and label ["2*Sz"]
                # creates a LegCharge object from the flattened list of charges
                # one for each of four legs or dimensions on B
                npc.LegCharge.from_qflat(npc.ChargeInfo([1], ["2*Sz"]), [1], qconj=1),
                npc.LegCharge.from_qflat(npc.ChargeInfo([1], ["2*Sz"]), [1, -1], qconj=1),
                npc.LegCharge.from_qflat(npc.ChargeInfo([1], ["2*Sz"]), [1, -1], qconj=-1),
                npc.LegCharge.from_qflat(npc.ChargeInfo([1], ["2*Sz"]), [1], qconj=-1),
            ]

            B_array = npc.Array.from_ndarray(B, legcharges=leg_charge, labels=labels)
        else:
            B_array = npc.Array.from_ndarray_trivial(B, labels=labels)

        num_sites = lat.N_sites
        # initialize the MPO psi with the wavepacket and an identity operator
        psi = cls.from_wavepacket(lat.mps_sites(), [1.0] * num_sites, "Id")

        # set the wavefunction at each site in psi to B_array
        for k in range(num_sites):
            psi.set_W(k, B_array)

        # srt the bond strengths of psi to a list of lists with all elements as 1.0
        psi.Ss = [[1.0]] * num_sites
        # psi is now an MPO representing the identity operator
        # psi consists of an identical B_array at each site
        # psi is a product of local identity operators since the bond dimensions are all 1
        return cast(MPOState, psi)

    def overlap(self, initial_state: Any) -> complex:
        """Compute the overlap of this state with the provided initial state.

        .. warning::
           This implementation only supports instances of
           :external:class:`tenpy.networks.mps.MPS` for ``initial_state``.

        Args:
            initial_state: the initial state with which to compute the overlap.

        Raises:
            TypeError: if the provided initial state has an incompatible type.

        Returns:
            The overlap of this state with the provided one.
        """
        if not isinstance(initial_state, MPS):
            raise TypeError(
                "MPOState.overlap is only implemented for tenpy.networks.mps.MPS states! "
                "But not for states of type '%s'",
                type(initial_state),
            )

        for k in range(self.L):
            self.set_W(k, np.sqrt(2.0) * self.get_W(k))

        overlap = self.expectation_value(initial_state)

        for k in range(self.L):
            self.set_W(k, (1.0 / np.sqrt(2.0)) * self.get_W(k))

        return cast(complex, overlap)


def MPS_neel_state(lat: Lattice) -> MPS:
    """Constructs the Néel state as an MPS.

    Args:
        lat: the lattice describing the MPS sites.

    Returns:
        A Néel state as an MPS.
    """
    num_sites = lat.N_sites
    product_state = ["up", "down"] * (num_sites // 2) + (num_sites % 2) * ["up"]
    initial_state = MPS.from_product_state(lat.mps_sites(), product_state, bc=lat.bc_MPS)
    return initial_state
