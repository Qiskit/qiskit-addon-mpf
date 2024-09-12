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

"""Static MPFs.

.. currentmodule:: qiskit_addon_mpf.static

Linear system of equations utilities
------------------------------------

.. autosummary::
   :toctree:  ../stubs/
   :nosignatures:

   LSE
   setup_lse

Exact static MPF coefficients
-----------------------------

.. autosummary::
   :toctree:  ../stubs/
   :nosignatures:

   setup_exact_model

Approximate static MPF coefficients
-----------------------------------

.. autosummary::
   :toctree:  ../stubs/
   :nosignatures:

   setup_approximate_model
"""

from .approximate import setup_approximate_model
from .exact import setup_exact_model
from .lse import LSE, setup_lse

__all__ = [
    "LSE",
    "setup_lse",
    "setup_exact_model",
    "setup_approximate_model",
]
