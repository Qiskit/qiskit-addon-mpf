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

"""A quimb-based MPO state."""

from __future__ import annotations

from functools import partialmethod
from itertools import zip_longest
from typing import Any, cast

import numpy as np
from autoray import do
from quimb.tensor import (
    MatrixProductOperator,
    MatrixProductState,
    Tensor,
    expec_TN_1D,
)
from quimb.tensor.tensor_core import (
    group_inds,
    rand_uuid,
    tensor_contract,
    tensor_split,
)

from .. import State


class MPOState(MatrixProductOperator, State):
    """An MPO enforcing the Vidal gauge.

    This specialization of quimb's existing :external:class:`quimb.tensor.MatrixProductOperator`
    enforces the Vidal gauge throughout its existence. This ensures a stable behavior of the
    :class:`.DynamicMPF` algorithm when using the
    :class:`~qiskit_addon_mpf.backends.quimb_tebd.TEBDEvolver`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a :class:`MPOState` instance.

        .. hint::
           All arguments (positional and keyword) are simply forwarded to the
           :external:class:`quimb.tensor.MatrixProductOperator` constructor. Additionally, the
           :attr:`vidal_singular_values` attribute gets initialized to a list of empty lists of
           length equal to the number of sites in this MPO.

        Args:
            args: all positional arguments will be forwarded to the
                :external:class:`quimb.tensor.MatrixProductState` constructor.
            kwargs: all keyword arguments will be forwarded to the
                :external:class:`quimb.tensor.MatrixProductState` constructor.
        """
        super().__init__(*args, **kwargs)

        self.vidal_singular_values: list[list[float]] = [[]] * self._L
        """A nested list of singular values. The outer list is of equal length as this MPO itself
        (:external:attr:`quimb.tensor.TensorNetwork1D.L`). Every item is another list of all the
        singular values for determining the Vidal gauge at that site.
        """

    # TODO: extend the capabilities of this function to work for gates not acting on 2 qubits.
    def gate_split(
        self,
        gate: np.ndarray,
        where: tuple[int, int],
        inplace: bool = False,
        conj: bool = False,
        **split_opts,
    ) -> MPOState:
        """Apply a two-site gate and contract it back into the MPO.

        The basic principle of this method is the same as that of
        :external:meth:`quimb.tensor.MatrixProductState.gate_split`. However, the implementation
        ensures that the Vidal gauge is conserved.

        Args:
            gate: the gate to be applied to the MPO. Its shape should be either ``(d**2, d**2)`` for
                a physical dimension of ``d``, or a reshaped version thereof.
            where: the indices of the sites where the gate should be applied.
            inplace: whether to perform the gate application in-place or return a new
                :class:`MPOState` with the gate applied to it.
            conj: whether the gate should be applied to the lower (``conj=False``, the default,
                :external:meth:`~quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator.lower_ind`)
                or upper (``conj=True``,
                :external:meth:`~quimb.tensor.tensor_arbgeom.TensorNetworkGenOperator.upper_ind`)
                indices of the underlying MPO.

                .. note::
                   This is essentially how the LHS and RHS of the :class:`.DynamicMPF` are
                   differentiated, by passing their :attr:`.Evolver.conjugate` property to this
                   argument.
            split_opts: additional keyword arguments that will be forwarded to the
                :external:func:`quimb.tensor.tensor_split` function. These can be used to affect the
                truncation of the tensor before it gets contracted back into the MPO.

        Returns:
            The :class:`MPOState` with the ``gate`` applied and contracted.
        """
        t_n = self if inplace else self.copy()

        # By default, the indices of the MPO to which we apply the gate are the "upper indices"
        # (commonly labeled "k" in quimb).
        inds = tuple(map(self.lower_ind, where))
        if conj:
            # However, when we are dealing with the conjugated side, we apply the gate to the "lower
            # indices" (commonly labeled "b" in quimb).
            inds = tuple(map(self.upper_ind, where))
            # Obviously we must also conjugate the actual gate being applied.
            gate = gate.conj()

        # Now that we know the indices, we can extract the left and right tensors on which this gate
        # acts.
        t_l, t_r = t_n._inds_get(*inds)
        # And we can extract the indices of the left and right legs. The middle item of the tuple
        # contains the indices of any legs shared between the tensors. In this case, we know that
        # there is exactly one such index but we do not need it later and, thus, ignore it here.
        left_inds, (_,), right_inds = group_inds(t_l, t_r)

        # Next, we must prepare the gate by converting it to a Tensor.
        if gate.ndim != 2 * len(inds):  # pragma: no branch
            # The gate was supplied as a matrix, so we must factorize it.
            dims = [t_n.ind_size(ix) for ix in inds]
            gate = do("reshape", gate, dims * 2)

        # We must generate new indices to join the old physical sites to the new gate.
        bnds = [rand_uuid() for _ in range(len(inds))]
        # And we keep track of mapping inds to bnds.
        reindex_map = dict(zip(inds, bnds))

        # Now we actually create the tensor of our gate.
        gate_tensor = Tensor(gate, inds=(*inds, *bnds), left_inds=bnds)
        # And contract it with the left and right tensors of our MPO.
        C = tensor_contract(t_l.reindex_(reindex_map), t_r.reindex_(reindex_map), gate_tensor)
        # At this point we have built and contracted a network which looks as follows:
        #
        #        |   |
        #      --tL--tR--
        #        |   |
        #        G_gate
        #        |   |

        # Now we create a copy of C because we want to scale the left tensor with the singular
        # values ensuring our Vidal gauge. We call this new tensor `theta`:
        #
        #        |   |
        #   --S--tL--tR--
        #        |   |
        #        G_gate
        #        |   |
        theta = C.copy()
        theta.modify(
            data=np.asarray(
                [
                    (s or 1.0) * t
                    for s, t in zip_longest(self.vidal_singular_values[where[0]], theta.data)
                ]
            )
        )

        # Next, we split the contracted tensor network using SVD. Since we want to obtain the
        # singular values, we must set absorb=None.
        # NOTE: we can ignore U because we will rely solely in V.
        _, S, V = tensor_split(
            theta,
            left_inds=left_inds,
            right_inds=right_inds,
            get="tensors",
            **split_opts,
            absorb=None,  # NOTE: this takes precedence over what might be specified in split_opts
        )
        # We now store the renormalized singular values on the right index.
        renorm = np.linalg.norm(S.data)
        t_n.vidal_singular_values[where[1]] = S.data / renorm

        # And since we have renormalized the singular values, the same must be done for U, which we
        # compute by contracting C with the conjugated V tensor.
        U = tensor_contract(C, V.conj())
        U.modify(data=U.data / renorm)

        # And finally, before being able to update our left and right tensors, we must reverse the
        # reindexing which was applied earlier.
        rev_reindex_map = {v: k for k, v in reindex_map.items()}
        t_l.reindex_(rev_reindex_map).modify(data=U.transpose_like_(t_l).data)
        t_r.reindex_(rev_reindex_map).modify(data=V.transpose_like_(t_r).data)

        return t_n

    gate_split_ = partialmethod(gate_split, inplace=True)
    """The ``inplace=True`` variant of :meth:`gate_split`."""

    def overlap(self, initial_state: Any) -> complex:
        """Compute the overlap of this state with the provided initial state.

        .. warning::
           This implementation only supports instances of
           :external:class:`quimb.tensor.MatrixProductState` for ``initial_state``.

        Args:
            initial_state: the initial state with which to compute the overlap.

        Raises:
            TypeError: if the provided initial state has an incompatible type.

        Returns:
            The overlap of this state with the provided one.
        """
        if not isinstance(initial_state, MatrixProductState):
            raise TypeError(
                "MPOState.overlap is only implemented for quimb.tensor.MatrixProductState states! "
                "But not for states of type '%s'",
                type(initial_state),
            )

        for k in range(self._L):
            C = self[k]
            C.modify(data=np.sqrt(2.0) * C.data)

        ret = expec_TN_1D(initial_state, self, initial_state)

        for k in range(self._L):
            C = self[k]
            C.modify(data=(1.0 / np.sqrt(2.0)) * C.data)

        return cast(complex, ret)
