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

"""A TeNPy-based model for describing a single layer of interactions."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from qiskit.circuit import QuantumCircuit
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.networks import Site, SpinHalfSite
from tenpy.tools.params import Config


class LayerModel(CouplingMPOModel, NearestNeighborModel):
    """A model for representing a layer of time-evolution interactions.

    Essentially, this class is a simple wrapper of
    :external:class:`tenpy.models.model.CouplingMPOModel` and
    :external:class:`tenpy.models.model.NearestNeighborModel`. Its main purpose is to provide a
    simple interface for constructing a TeNPy-compatible Hamiltonian from Qiskit objects.
    """

    def init_sites(self, model_params: Config) -> Site:
        """Initializes the sites of this Hamiltonian.

        See :external:meth:`~tenpy.models.model.CouplingMPOModel.init_sites` for more details.

        Args:
            model_params: the model parameters.

        Returns:
            The site to be used internally.
        """
        # WARN: we use our own default to NOT sort charges (contrary to TeNPy default: `True`)
        sort_charge = model_params.get("sort_charge", False, bool)
        conserve = model_params.get("conserve", "Sz", str)
        return SpinHalfSite(conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params: Config) -> None:
        """Initializes the terms of this Hamiltonian.

        See :external:meth:`~tenpy.models.model.CouplingMPOModel.init_terms` for more details.

        Args:
            model_params: the model parameters.
        """
        coupling_terms = model_params.get("coupling_terms", {})
        onsite_terms = model_params.get("onsite_terms", {})

        for category, terms in coupling_terms.items():
            for term in terms:
                self.add_coupling_term(*term, category=category)

        for category, terms in onsite_terms.items():
            for term in terms:
                self.add_onsite_term(*term, category=category)

    def calc_H_bond(self, tol_zero: float = 1e-15) -> list:
        """Calculate the interaction Hamiltonian based on the coupling and onsite terms.

        Essentially, this class overwrites
        :external:meth:`~tenpy.models.model.CouplingModel.calc_H_bond` and takes care of removing
        even or odd bond interaction Hamiltonians depending on the value of ``keep_only_odd`` (see
        :mod:`.tenpy_layers` for more details).

        Args:
            tol_zero: the threshold for values considered to be zero.

        Returns:
            The list of interaction Hamiltonians for all bonds.
        """
        H_bond = super().calc_H_bond(tol_zero=tol_zero)

        keep_only_odd = self.options.get("keep_only_odd", None, bool)
        if keep_only_odd is None:
            # return H_bond unchanged
            return cast(list, H_bond)

        # NOTE: if `keep_only_odd` was specified, that means we explicitly overwrite those H_bond
        # values with `None` which we do not want to keep. In the case of (for example) coupling
        # layers, this should have no effect since those bonds were `None` to begin with. However,
        # in the case of onsite layers, this will remove half of the bonds ensuring that we split
        # the bond updates into even and odd parts (as required by the TEBD algorithm).
        for i in range(0 if keep_only_odd else 1, self.lat.N_sites, 2):
            H_bond[i] = None

        return cast(list, H_bond)

    @classmethod
    def from_quantum_circuit(
        cls,
        circuit: QuantumCircuit,
        **kwargs,
    ) -> LayerModel:
        """Construct a :class:`LayerModel` from a :external:class:`~qiskit.circuit.QuantumCircuit`.

        You can see an example of this function in action in the docs of :mod:`tenpy_layers`.

        .. note::
           By default, TeNPy tries to enforce spin-conservation and, thus, some operations may not
           be available. If you encounter an error stating that some operator (e.g. ``Sx``) is not
           available, try specifying ``conserve="None"``. If that still does not work, converting
           your specific ``QuantumCircuit`` is currently not possible using this implementation.

        Args:
            circuit: the quantum circuit to parse.
            kwargs: any additional keyword arguments to pass to the :class:`LayerModel` constructor.

        Returns:
            A new LayerModel instance.

        Raises:
            NotImplementedError: if an unsupported quantum gate is encountered.
        """
        coupling_terms = defaultdict(list)
        onsite_terms = defaultdict(list)

        for instruction in circuit.data:
            op = instruction.operation
            sites = [circuit.find_bit(qubit)[0] for qubit in instruction.qubits]

            # NOTE: the hard-coded scaling factors below account for the Pauli matrix conversion
            if op.name in {"rxx", "ryy", "rzz"}:
                s_p = f"S{op.name[-1]}"
                coupling_terms[f"{s_p}_i {s_p}_j"].append(
                    (
                        2.0 * op.params[0],
                        *sites,
                        s_p,
                        s_p,
                        "Id",
                    )
                )
            elif op.name == "xx_plus_yy":
                coupling_terms["Sp_i Sm_j"].append((0.5 * op.params[0], *sites, "Sp", "Sm", "Id"))
                coupling_terms["Sp_i Sm_j"].append((0.5 * op.params[0], *sites, "Sm", "Sp", "Id"))
            elif op.name in {"rx", "ry", "rz"}:
                s_p = f"S{op.name[-1]}"
                onsite_terms[s_p].append((op.params[0], *sites, s_p))
            else:
                raise NotImplementedError(f"Cannot handle gate of type {op.name}")

        return cls(
            {
                "L": circuit.num_qubits,
                "coupling_terms": coupling_terms,
                "onsite_terms": onsite_terms,
                **kwargs,
            }
        )
