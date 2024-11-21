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

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit_addon_mpf.backends import HAS_TENPY

if HAS_TENPY:
    from qiskit_addon_mpf.backends.tenpy_tebd import MPOState
    from tenpy.models.lattice import Chain
    from tenpy.networks.site import SpinHalfSite


@pytest.mark.skipif(not HAS_TENPY, reason="Tenpy is required for these unittests")
class TestMPOState:
    def test_unsupported_state(self):
        """Test the handling of a non-supported state object."""
        site = SpinHalfSite()
        lattice = Chain(5, site)
        state = MPOState.initialize_from_lattice(lattice)
        circ = QuantumCircuit(5)
        with pytest.raises(TypeError):
            state.overlap(circ)
