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

"""Frobenius norm MPF coefficients."""

from __future__ import annotations

import cvxpy as cp

from .lse import LSE


def setup_frobenius_problem(
    lse: LSE, *, max_l1_norm: float = 10.0, assume_PSD: bool = False
) -> tuple[cp.Problem, cp.Variable]:
    r"""Construct a :external:class:`cvxpy.Problem` for finding approximate MPF coefficients.

    The optimization problem constructed by this function is defined as follows:

    - the cost function minimizes the following quadratic expression:

      .. math::
         1 + x^T A x - 2 x^T b

      As shown in [1] and [2], this expression arises from the Frobenius norm of the error between
      an exact time evolution state and a dynamic MPF. As such, taking the :class:`.LSE` constructed
      by :func:`.setup_dynamic_lse` and plugging it into this function will yield Eq. (20) of [1]
      (which is identical to Eq. (2) of [2]), which we repeat below

      .. math::
         1 + \sum_{i,j} A_{ij}(t) x_i(t) x_j(t) - 2 \sum_i b_i(t) x_i(t) \, ,

      where $A$ and $b$ of our :class:`.LSE` correspond to the Gram matrix ($M$ in [1] and [2]) and
      the overlap vector ($L$ in [1] and [2]), respectively. Additionally, we use $x(t)$ to denote
      the MPF variables (or coefficients) rather than $c$ in [1] and [2].

    - two constraints are set:

      1. the variables must sum to 1: :math:`\sum_i x_i == 1`
      2. the L1-norm (:external:class:`~cvxpy.atoms.norm1.norm1`) of the variables is bounded by
         ``max_l1_norm``

    Below is an example which uses the ``lse`` object constructed in the example for
    :func:`.setup_dynamic_lse`.

    .. testsetup::
        >>> from functools import partial
        >>> from qiskit_addon_mpf.backends.quimb_tebd import MPOState, TEBDEvolver
        >>> from qiskit_addon_mpf.dynamic import setup_dynamic_lse
        >>> from quimb.tensor import ham_1d_heis, MPO_identity, MPS_neel_state
        >>> trotter_steps = [3, 4]
        >>> time = 0.9
        >>> num_qubits = 10
        >>> initial_state = MPS_neel_state(num_qubits)
        >>> hamil = ham_1d_heis(num_qubits, 0.8, 0.3, cyclic=False)
        >>> identity_factory = lambda: MPOState(MPO_identity(num_qubits))
        >>> exact_evolver_factory = partial(
        ...     TEBDEvolver,
        ...     H=hamil,
        ...     dt=0.05,
        ...     order=4,
        ...     split_opts={"max_bond": 10, "cutoff": 1e-5},
        ... )
        >>> approx_evolver_factory = partial(
        ...     TEBDEvolver,
        ...     H=hamil,
        ...     order=2,
        ...     split_opts={"max_bond": 10, "cutoff": 1e-5},
        ... )
        >>> lse = setup_dynamic_lse(
        ...     trotter_steps,
        ...     time,
        ...     identity_factory,
        ...     exact_evolver_factory,
        ...     approx_evolver_factory,
        ...     initial_state,
        ... )

    .. doctest::
        >>> from qiskit_addon_mpf.costs import setup_frobenius_problem
        >>> problem, coeffs = setup_frobenius_problem(lse, max_l1_norm=3.0, assume_PSD=True)
        >>> print(problem)  # doctest: +FLOAT_CMP
        minimize 1.0 + QuadForm(x, psd_wrap([[1.00 1.00]
                                            [1.00 1.00]])) + -([2.00003171 1.99997911] @ x)
        subject to Sum(x, None, False) == 1.0
                   norm1(x) <= 3.0

    You can then solve the problem and access the expansion coefficients like so:

    .. doctest::
        :pyversion: < 3.10
        >>> final_cost = problem.solve()
        >>> print(coeffs.value)  # doctest: +FLOAT_CMP
        [0.50596416 0.49403584]

    Args:
        lse: the linear system of equations from which to build the model.
        max_l1_norm: the upper limit to use for the constrain of the L1-norm of the variables.
        assume_PSD: whether to assume the provided :attr:`lse.A` matrix is positive semi-definite.
            This is a keyword argument that gets forwarded to :func:`cvxpy.quad_form` and permits
            bypassing any sanity checks to allow handling cases in which close-to-zero eigenvalues
            are numerically instable.

    Returns:
        The optimization problem and coefficients variable.

    References:
        [1]: S. Zhuk et al., Phys. Rev. Research 6, 033309 (2024).
             https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033309
        [2]: N. Robertson et al., arXiv:2407.17405v2 (2024).
             https://arxiv.org/abs/2407.17405v2
    """
    coeffs = lse.x
    cost = 1.0 + cp.quad_form(coeffs, lse.A, assume_PSD=assume_PSD) - 2.0 * lse.b.T @ coeffs
    constraints = [cp.sum(coeffs) == 1, cp.norm1(coeffs) <= max_l1_norm]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, coeffs
