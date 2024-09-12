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

"""Linear system of equations for static MPF coefficients."""

from __future__ import annotations

from typing import NamedTuple, cast

import cvxpy as cp
import numpy as np


class LSE(NamedTuple):
    """A :class:`.namedtuple` representing a linear system of equations.

    .. math::
             A x = b
    """

    A: np.ndarray
    """The left hand side of the LSE."""

    b: np.ndarray
    """The right hand side of the LSE."""

    @property
    def x(self) -> cp.Variable:
        """Returns the $x$ :external:class:`~cvxpy.expressions.variable.Variable`."""
        return cp.Variable(shape=len(self.b), name="x")

    def solve(self) -> np.ndarray:
        """Return the solution to this LSE: :math:`x=A^{-1}b`.

        Returns:
            The solution to this LSE.

        Raises:
            ValueError: if this LSE is parameterized with unassigned values.
        """
        if self.A.ndim == 1:
            # self.A is a vector of cp.Expression objects
            mat_a = np.array([row.value for row in self.A])
            if any(row is None for row in mat_a):
                raise ValueError(
                    "This LSE contains unassigned parameter values! Assign a value to them first "
                    "before trying to solve this LSE again."
                )
        else:
            mat_a = self.A
        return cast(np.ndarray, np.linalg.solve(mat_a, self.b))


def setup_lse(
    trotter_steps: list[int] | cp.Parameter,
    *,
    order: int = 1,
    symmetric: bool = False,
) -> LSE:
    r"""Return the linear system of equations for computing the static MPF coefficients.

    This function constructs the following linear system of equations:

    .. math::
         A x = b,

    with

    .. math::
        A_{0,j} &= 1 \\
        A_{i>0,j} &= k_{j}^{-(\chi + s(i-1))} \\
        b_0 &= 1 \\
        b_{i>0} &= 0

    where $\\chi$ is the ``order``, $s$ is $2$ if ``symmetric`` is ``True`` and $1$ oterhwise,
    $k_{j}$ are the ``trotter_steps``, and $x$ are the variables to solve for.
    The indices $i$ and $j$ start at $0$.

    Here is an example:

    >>> from qiskit_addon_mpf.static import setup_lse
    >>> lse = setup_lse([1,2,3], order=2, symmetric=True)
    >>> print(lse.A)
    [[1.         1.         1.        ]
     [1.         0.25       0.11111111]
     [1.         0.0625     0.01234568]]
    >>> print(lse.b)
    [1. 0. 0.]

    Args:
        trotter_steps: the sequence of trotter steps from which to build $A$. Rather than a list of
            integers, this may also be a
            :external:class:`~cvxpy.expressions.constants.parameter.Parameter` instance of the
            desired size. In this case, the constructed :class:`.LSE` is parameterized whose values
            must be assigned before it can be solved.
        order: the order of the individual product formulas making up the MPF.
        symmetric: whether the individual product formulas making up the MPF are symmetric. For
            example, the Lie-Trotter formula is `not` symmetric, while Suzuki-Trotter `is`.

    Returns:
        An :class:`.LSE`.
    """
    symmetric_factor = 2 if symmetric else 1

    trotter_steps_arr: np.ndarray | cp.Parameter
    if isinstance(trotter_steps, cp.Parameter):
        assert trotter_steps.ndim == 1
        num_trotter_steps = trotter_steps.size
        trotter_steps_arr = trotter_steps
    else:
        num_trotter_steps = len(trotter_steps)
        trotter_steps_arr = np.array(trotter_steps)

    mat_a = np.array(
        [
            1.0 / trotter_steps_arr ** (0 if k == 0 else (order + symmetric_factor * (k - 1)))
            for k in range(num_trotter_steps)
        ]
    )
    vec_b = np.zeros(num_trotter_steps)
    vec_b[0] = 1
    return LSE(mat_a, vec_b)
