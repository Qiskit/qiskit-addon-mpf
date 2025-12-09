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

"""Dynamic MPF coefficients.

.. currentmodule:: qiskit_addon_mpf.dynamic

This module provides the generator function for the linear system of equations (:class:`.LSE`) for
computing dynamic (that is, time-dependent) MPF coefficients.

.. autofunction:: setup_dynamic_lse

Factory Protocols
-----------------

The following protocols define the function signatures for the various object factory arguments.

.. autoclass:: IdentityStateFactory

.. autoclass:: ExactEvolverFactory

.. autoclass:: ApproxEvolverFactory

Core algorithm
--------------

.. autoclass:: DynamicMPF
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Protocol, cast

import numpy as np

from .backends.interface import Evolver, State
from .costs import LSE

LOGGER = logging.getLogger(__name__)


class DynamicMPF:
    """The dynamic MPF algorithm.

    Instantiated with a LHS and RHS :class:`.Evolver` this algorithm will
    :meth:`~.DynamicMPF.evolve` a shared :class:`.State` up to a target evolution time.
    Afterwards, the :meth:`.DynamicMPF.overlap` of the time-evolved :class:`.State` with some
    initial state can be computed. See :func:`.setup_dynamic_lse` for a more detailed explanation on
    how this is used to compute the elements :math:`M_{ij}` and :math:`L_i` making up the
    :class:`.LSE` of the dynamic MPF coefficients.

    References:
        [1]: S. Zhuk et al., Phys. Rev. Research 6, 033309 (2024).
             https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033309
        [2]: N. Robertson et al., arXiv:2407.17405 (2024).
             https://arxiv.org/abs/2407.17405
    """

    TIME_DECIMALS: int = 8
    """The number of decimal places used for rounding the evolution times.

    During the time evolution of the :attr:`evolution_state`, we often compare the evolved times of
    the LHS and RHS engines against each other as well as the target evolution time. These checks
    compare floating point numbers and this setting specifies the number of decimal places to which
    we round.
    """

    def __init__(self, evolution_state: State, lhs: Evolver, rhs: Evolver) -> None:
        """Construct a :class:`.DynamicMPF` instance.

        Args:
            evolution_state: the state to be shared by the LHS and RHS time-evolution engines.
            lhs: the LHS time-evolution engine.
            rhs: the RHS time-evolution engine.
        """
        self.evolution_state = evolution_state
        """The state shared between the LHS and RHS time-evolution engines."""
        self.lhs = lhs
        """The LHS time-evolution engine."""
        self.rhs = rhs
        """The RHS time-evolution engine."""

    def evolve(self, time: float) -> None:
        """Evolve the dynamic MPF algorithm up to the provided time.

        This actually runs the dynamic MPF algorithm by time-evolving
        :attr:`.DynamicMPF.evolution_state` up to the specified time using the LHS and RHS
        :class:`.Evolver` instances.

        Args:
            time: the total target evolution time.

        Raises:
            RuntimeError: if the LHS and RHS evolved times are not equal at the end.
        """
        round_ = partial(np.round, decimals=self.TIME_DECIMALS)
        time = cast(float, round_(time))
        while round_(self.lhs.evolved_time) < time:
            while round_(self.rhs.evolved_time) < round_(self.lhs.evolved_time):
                self.rhs.step()
                LOGGER.info("Stepped RHS to %s", self.rhs.evolved_time)
            self.lhs.step()
            LOGGER.info("Stepped LHS to %s", self.lhs.evolved_time)
        # we must ensure that the RHS can catch up with the LHS
        while round_(self.rhs.evolved_time) < time:
            self.rhs.step()
            LOGGER.info("Stepped RHS to %s", self.rhs.evolved_time)

        LOGGER.info("Final times are %s and %s", self.lhs.evolved_time, self.rhs.evolved_time)

        if not np.isclose(
            self.lhs.evolved_time, self.rhs.evolved_time, rtol=10**-self.TIME_DECIMALS
        ):
            raise RuntimeError(
                "Check the numerical accuracy of your target evolution time! The evolved times of "
                "the LHS and RHS are not equal: ",
                self.lhs.evolved_time,
                self.rhs.evolved_time,
            )

    def overlap(self, initial_state: Any) -> complex:
        """Compute the overlap of :attr:`.DynamicMPF.evolution_state` with the provided state.

        .. warning::
           The type of the provided ``initial_state`` will depend on the chosen backend used for the
           :class:`.State` and :class:`.Evolver` instances provided to this :class:`.DynamicMPF`
           instance. In other words, a backend may only support a specific type of ``initial_state``
           objects for this overlap computation. See also the explanations of the ``initial_state``
           argument to the :func:`.setup_dynamic_lse` for more details.

        Args:
            initial_state: the initial state with which to compute the overlap.

        Raises:
            TypeError: if the provided initial state has an incompatible type.

        Returns:
            The overlap of :attr:`.DynamicMPF.evolution_state` with the provided one.
        """
        return self.evolution_state.overlap(initial_state)


class IdentityStateFactory(Protocol):
    r"""The factory function protocol for constructing an identity :class:`.State` instance.

    As explained in more detail in :func:`.setup_dynamic_lse`, this factory function is called to
    initialize the :attr:`.DynamicMPF.evolution_state` with an identity or empty state. This
    function should not take any arguments and return a :class:`.State` instance.
    """

    def __call__(self) -> State:  # noqa: D102
        ...  # pragma: no cover


class ExactEvolverFactory(Protocol):
    r"""The factory function protocol for constructing an exact :class:`.Evolver` instance.

    As explained in more detail in :func:`.setup_dynamic_lse`, this factory function is called to
    initialize the :attr:`.DynamicMPF.lhs` instances of :class:`.Evolver` which produce the exact
    time-evolution state, :math:`\rho(t)`, when computing the elements :math:`L_i`.
    """

    def __call__(self, evolution_state: State, /) -> Evolver:  # noqa: D102
        ...  # pragma: no cover


class ApproxEvolverFactory(Protocol):
    r"""The factory function protocol for constructing an approximate :class:`.Evolver` instance.

    As explained in more detail in :func:`.setup_dynamic_lse`, this factory function is called to
    initialize either the :attr:`.DynamicMPF.rhs` instances of :class:`.Evolver` when computing the
    elements :math:`L_i` or both sides (:attr:`.DynamicMPF.lhs` and :attr:`.DynamicMPF.rhs`) when
    computing elements :math:`M_{ij}`. Since these approximate time evolution states depend on the
    Trotter step (:math:`\rho_{k_i}(t)`), this function requires the time step of the time evolution
    to be provided as a keyword argument called ``dt``.
    """

    def __call__(self, evolution_state: State, /, *, dt: float = 1.0) -> Evolver:  # noqa: D102
        ...  # pragma: no cover


def setup_dynamic_lse(
    trotter_steps: list[int],
    time: float,
    identity_factory: IdentityStateFactory,
    exact_evolver_factory: ExactEvolverFactory,
    approx_evolver_factory: ApproxEvolverFactory,
    initial_state: Any,
) -> LSE:
    r"""Return the linear system of equations for computing dynamic MPF coefficients.

    This function uses the :class:`.DynamicMPF` algorithm to compute the components of the Gram
    matrix (:attr:`.LSE.A`, :math:`M` in [1] and [2]) and the overlap vector (:attr:`.LSE.b`,
    :math:`L` in [1] and [2]) for the provided time-evolution parameters.

    The elements of the Gram matrix, :math:`M_{ij}`, and overlap vector, :math:`L_i`, are defined as

    .. math::
        M_{ij} &= \text{Tr}(\rho_{k_i}(t)\rho_{k_j}(t)) \, , \\
        L_i &= \text{Tr}(\rho(t)\rho_{k_i}(t)) \, ,

    where :math:`\rho(t)` is the exact time-evolution state at time :math:`t` and
    :math:`\rho_{k_i}(t)` is the time-evolution state approximated using :math:`k_i` Trotter steps.

    Computing the dynamic (that is, time-dependent) MPF coefficients from :math:`M` and :math:`L`
    amounts to finding a solution to the :class:`.LSE` (similarly to how the :mod:`.static` MPF
    coefficients are computed) while enforcing the constraint that all coefficients must sum to 1
    (:math:`\sum_i x_i = 1`), which is not enforced as part of this LSE (unlike in the static case).
    Optimization problems which include this additional constraint are documented in the
    :mod:`.costs` module. The one suggested by [1] and [2] is the
    :meth:`~qiskit_addon_mpf.costs.setup_frobenius_problem`.

    Evaluating every element :math:`M_{ij}` and :math:`L_i` requires computing the overlap between
    two time-evolution states. The :class:`.DynamicMPF` algorithm does so by means of tensor network
    calculations, provided by one of the optional dependencies. The available backends are listed
    and explained in more detail in the :mod:`.backends` module.

    Below, we provide an example using the :mod:`~.qiskit_addon_mpf.backends.quimb_tebd` backend.
    We briefly explain each element.

    First, we initialize a simple Heisenberg Hamiltonian which we would like to time-evolve. Since
    we are using a time-evolver based on :external:mod:`quimb`, we also initialize the Hamiltonian
    using that library.

    >>> from quimb.tensor import ham_1d_heis
    >>> num_qubits = 10
    >>> hamil = ham_1d_heis(num_qubits, 0.8, 0.3, cyclic=False)

    Next, we define the number of Trotter steps to make up our MPF, the target evolution time as
    well as the initial state (:math:`\psi_{in}` in [1] and :math:`\psi_0` in [2], resp.) with
    respect to which we compute the overlap between the time-evolution states. Here, we simply use
    the Néel state which we also construct using :external:mod:`quimb`:

    >>> trotter_steps = [3, 4]
    >>> time = 0.9

    >>> from quimb.tensor import MPS_neel_state
    >>> initial_state = MPS_neel_state(num_qubits)

    Since we must run the full :class:`.DynamicMPF` algorithm for computing every element of
    :math:`M_{ij}` and :math:`L_i`, we must provide factory methods for initializing the input
    arguments of the :class:`.DynamicMPF` instances. To this end, we must provide three functions.
    To construct these, we will use the :func:`functools.partial` function.

    >>> from functools import partial

    First, we need a function to initialize an empty time-evolution state (see also
    :attr:`.DynamicMPF.evolution_state` for more details). This constructor function may not take
    any positional or keyword arguments and must return a :class:`.State` object.

    >>> from qiskit_addon_mpf.backends.quimb_tebd import MPOState
    >>> from quimb.tensor import MPO_identity
    >>> identity_factory = lambda: MPOState(MPO_identity(num_qubits))

    The second and third function must construct the left- and right-hand side time-evolution
    engines (see also :attr:`.DynamicMPF.lhs` and :attr:`.DynamicMPF.rhs` for more details). These
    functions should follow the :class:`.ExactEvolverFactory` and :class:`.ApproxEvolverFactory`
    protocols, respectively.

    The :class:`.ExactEvolverFactory` function should take a :class:`.State` object as its only
    positional argument and should return a :class:`.Evolver` object, which will be used for
    computing the LHS of the :math:`L_i` elements (i.e. it should produce the exact time-evolution
    state, :math:`\rho(t)`).

    Here, we approximate the exact time-evolved state with a fourth-order Suzuki-Trotter formula
    using a small time step of 0.05. We also specify some :external:mod:`quimb`-specific truncation
    options to bound the maximum bond dimension of the underlying tensor network as well as the
    minimum singular values of the split tensor network bonds.

    >>> from qiskit_addon_mpf.backends.quimb_tebd import TEBDEvolver
    >>> exact_evolver_factory = partial(
    ...     TEBDEvolver,
    ...     H=hamil,
    ...     dt=0.05,
    ...     order=4,
    ...     split_opts={"max_bond": 10, "cutoff": 1e-5},
    ... )

    The :class:`.ApproxEvolverFactory` function should also take a :class:`.State` object as its
    only positional argument and additionally a keyword argument called ``dt`` to specify the time
    step of the time-evolution. It should also return a :class:`.Evolver` object which produces the
    approximate time-evolution states, :math:`\rho_{k_i}(t)`, where :math:`k_i` is determined by the
    chosen time step, ``dt``. As such, these instances will be used for computing the RHS of the
    :math:`L_i` as well as both sides of the :math:`M_{ij}` elements.

    Here, we use a second-order Suzuki-Trotter formula with the same truncation settings as before.

    >>> approx_evolver_factory = partial(
    ...     TEBDEvolver,
    ...     H=hamil,
    ...     order=2,
    ...     split_opts={"max_bond": 10, "cutoff": 1e-5},
    ... )

    Finally, we can initialize and run the :func:`.setup_dynamic_lse` function to obtain the
    :class:`.LSE` described at the top.

    >>> from qiskit_addon_mpf.dynamic import setup_dynamic_lse
    >>> lse = setup_dynamic_lse(
    ...     trotter_steps,
    ...     time,
    ...     identity_factory,
    ...     exact_evolver_factory,
    ...     approx_evolver_factory,
    ...     initial_state,
    ... )
    >>> print(lse.A)  # doctest: +FLOAT_CMP
    [[1.         0.99998513]
     [0.99998513 1.        ]]
    >>> print(lse.b)  # doctest: +FLOAT_CMP
    [1.00001585 0.99998955]

    Args:
        trotter_steps: the sequence of trotter steps to be used.
        time: the total target evolution time.
        identity_factory: a function to generate an empty :class:`.State` object.
        exact_evolver_factory: a function to initialize the :class:`.Evolver` instance which
            produces the exact time-evolution state, :math:`\rho(t)`.
        approx_evolver_factory: a function to initialize the :class:`.Evolver` instance which
            produces the approximate time-evolution state, :math:`\rho_{k_i}(t)`, for different
            values of :math:`k_i` depending on the provided time step, ``dt``.
        initial_state: the initial state (:math:`\psi_{in}` or :math:`\psi_0`) with respect to which
            to compute the elements :math:`M_{ij}` of :attr:`.LSE.A` and :math:`L_i` of
            :attr:`.LSE.b`. The type of this object must match the tensor network backend chosen for
            the previous arguments.

    Returns:
        The :class:`.LSE` to find the dynamic MPF coefficients as described above.

    References:
        [1]: S. Zhuk et al., Phys. Rev. Research 6, 033309 (2024).
             https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033309
        [2]: N. Robertson et al., arXiv:2407.17405v2 (2024).
             https://arxiv.org/abs/2407.17405v2
    """
    Bs = np.zeros(len(trotter_steps), dtype=np.complex128)
    for idx, k in enumerate(trotter_steps):
        dt = time / k

        common_state = identity_factory()
        lhs = exact_evolver_factory(common_state)
        lhs.conjugate = True
        rhs = approx_evolver_factory(common_state, dt=dt)

        algo = DynamicMPF(common_state, lhs, rhs)
        algo.evolve(time)

        overlap = algo.overlap(initial_state)

        Bs[idx] = overlap

    Bs = np.power(np.abs(Bs), 2)

    As = np.eye(len(trotter_steps), dtype=np.complex128)
    for idx1, idx2 in zip(*np.triu_indices_from(As, k=1), strict=True):
        dta = time / trotter_steps[idx1]
        dtb = time / trotter_steps[idx2]

        common_state = identity_factory()
        lhs = approx_evolver_factory(common_state, dt=dta)
        lhs.conjugate = True
        rhs = approx_evolver_factory(common_state, dt=dtb)

        algo = DynamicMPF(common_state, lhs, rhs)
        algo.evolve(time)

        overlap = algo.overlap(initial_state)

        As[idx1][idx2] = overlap
        As[idx2][idx1] = overlap

    As = np.power(np.abs(As), 2)

    return LSE(As, Bs)
