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

"""The interface for :class:`.DynamicMPF` backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evolver(ABC):
    """The interface for the time-evolution algorithms used within :class:`.DynamicMPF`.

    This time-evolution interface is used by the :attr:`.DynamicMPF.lhs` and :attr:`.DynamicMPF.rhs`
    and should time-evolve a :class:`.State` object under its hood. The exact mechanism of the
    algorithm is described in more detail in :class:`.DynamicMPF`, :class:`.State`, and
    :func:`.setup_dynamic_lse`.
    """

    @abstractmethod
    def step(self) -> None:
        """Perform a single time step of this time-evolution algorithm.

        This should act on the internally referenced :class:`.State` object (for which no name is
        prescribed by this interface). Whether this time-evolution algorithm instance should evolve
        the :class:`.State` from the left- or right-hand side, depends on the value of
        :attr:`.conjugate`.
        """

    @property
    @abstractmethod
    def evolved_time(self) -> float:
        """Returns the current evolution time."""

    @property
    @abstractmethod
    def conjugate(self) -> bool:
        """Returns whether this time-evolver instance acts on the right-hand side."""

    @conjugate.setter
    @abstractmethod
    def conjugate(self, conjugate: bool) -> None: ...  # pragma: no cover


class State(ABC):
    """The interface for the :attr:`.DynamicMPF.evolution_state`.

    This time-evolution state is shared between the LHS and RHS :class:`.Evolver` instances of the
    :class:`.DynamicMPF` instance. In most cases where a concrete backend implementing this
    interface is based on tensor networks, this state will be a matrix product operator (MPO).
    This is because most time-evolution algorithms would normally evolve a matrix product state
    (MPS) as shown pictorially below, where time evolution blocks (``U#``) are successively applied
    to a 1-dimensional MPS (``S#``). Here, the tensor network grows towards the right as time goes
    on.

    .. code-block::

        MPS Evolution

        S0┄┄┲━━━━┱┄┄┄┄┄┄┄┄┲━━━━┱┄
        │   ┃ U1 ┃        ┃ U5 ┃
        S1┄┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄
        │          ┃ U3 ┃
        S2┄┄┲━━━━┱┄┺━━━━┹┄┲━━━━┱┄ ...
        │   ┃ U2 ┃        ┃ U6 ┃
        S3┄┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄
        │          ┃ U4 ┃
        S4┄┄┄┄┄┄┄┄┄┺━━━━┹┄┄┄┄┄┄┄┄

    However, in our case, we want two time-evolution engines to share a single state. In order to
    achieve that, we can have one of them evolve the state from the right (just as before, ``U#``),
    but have the second one evolve the state from the left (``V#``). This requires the state to also
    have bonds going of in that direction, rendering it a 2-dimensional MPO (``M#``) rather than the
    1-dimensional MPS from before.

    .. code-block::

        MPO Evolution

            ┄┲━━━━┱┄┄┄┄┄┄┄┄┲━━━━┱┄┄M0┄┄┲━━━━┱┄┄┄┄┄┄┄┄┲━━━━┱┄
             ┃ V5 ┃        ┃ V1 ┃  │   ┃ U1 ┃        ┃ U5 ┃
            ┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄┄M1┄┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄
                    ┃ V3 ┃         │          ┃ U3 ┃
        ... ┄┲━━━━┱┄┺━━━━┹┄┲━━━━┱┄┄M2┄┄┲━━━━┱┄┺━━━━┹┄┲━━━━┱┄ ...
             ┃ V6 ┃        ┃ V2 ┃  │   ┃ U2 ┃        ┃ U6 ┃
            ┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄┄M3┄┄┺━━━━┹┄┲━━━━┱┄┺━━━━┹┄
                    ┃ V4 ┃         │          ┃ U4 ┃
            ┄┄┄┄┄┄┄┄┺━━━━┹┄┄┄┄┄┄┄┄┄M4┄┄┄┄┄┄┄┄┄┺━━━━┹┄┄┄┄┄┄┄┄
    """

    @abstractmethod
    def overlap(self, initial_state: Any) -> complex:
        """Compute the overlap of this state with the provided initial state.

        .. warning::
           A concrete implementation of this method should raise a :class:`TypeError` if the
           provided ``initial_state`` object is not supported by the implementing backend.

        Args:
            initial_state: the initial state with which to compute the overlap.

        Raises:
            TypeError: if the provided initial state has an incompatible type.

        Returns:
            The overlap of this state with the provided one.
        """
