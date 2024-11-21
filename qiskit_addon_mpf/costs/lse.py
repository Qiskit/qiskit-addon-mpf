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

"""Linear system of equations supporting cvxpy variable and parameter objects."""

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
        r"""Return the solution to this LSE: :math:`x=A^{-1}b`.

        Returns:
            The solution to this LSE.

        Raises:
            ValueError: if this LSE is parameterized with unassigned values.
            ValueError: if this LSE does not include a row ensuring that :math:`\sum_i x_i == 1`
                which is a requirement for valid MPF coefficients.
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

        vec_b = self.b
        ones = [all(row == 1) for row in mat_a]
        if not any(ones) or not np.isclose(vec_b[np.where(ones)], 1.0):
            raise ValueError(
                "This LSE does not enforce the sum of all coefficients to be equal to 1 which is "
                "required for valid MPF coefficients. To find valid coefficients for this LSE use "
                "one of the non-exact cost functions provided in this module and find its optimal "
                "solution."
            )

        return cast(np.ndarray, np.linalg.solve(mat_a, vec_b))
