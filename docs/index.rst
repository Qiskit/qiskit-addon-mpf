##########################################
Qiskit addon: multi-product formulas (MPF)
##########################################

`Qiskit addons <https://docs.quantum.ibm.com/guides/addons>`_ are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for multi-product formulas (MPFs).
These can be used to reduce the Trotter error of Hamiltonian dynamics.

This package currently contains the following submodules:

- ``qiskit_addon_mpf.static`` for working with static MPFs [`1-2 <#references>`_].

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/qiskit-addon-mpf/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'qiskit-addon-mpf'


For more installation information refer to the `installation instructions <install.rst>`_ in the documentation.

Optional dependencies
+++++++++++++++++++++

The ``qiskit-addon-mpf`` package has a number of optional dependencies which enable certain features.
The dynamic MPF feature (see [`2-3 <#references>`_]) is one such example.
You can install the related optional dependencies like so:

.. code-block:: bash

    pip install 'qiskit-addon-mpf[dynamic]'

.. caution::
   The optional dependency `TeNPy <https://github.com/tenpy/tenpy>`_ was previously offered under a
   GPLv3 license.
   As of the release of `v1.0.4 <https://github.com/tenpy/tenpy/releases/tag/v1.0.4>`_ on October
   2nd, 2024, it has been offered under the Apache v2 license.
   The license of this package is only compatible with Apache-licensed versions of TeNPy.


Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-addon-mpf/release-notes.html>`_.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-mpf>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-mpf/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-mpf/issues/new/choose>`_ for tracking requests and bugs.

References
----------

1. A. Carrera Vazquez, D. J. Egger, D. Ochsner, and S. Wörner, `Well-conditioned multi-product formulas for hardware-friendly Hamiltonian simulation <https://quantum-journal.org/papers/q-2023-07-25-1067/>`_, Quantum 7, 1067 (2023).
2. S. Zhuk, N. Robertson, and S. Bravyi, `Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033309>`_, Phys. Rev. Research 6, 033309 (2024).
3. N. Robertson, et al. `Tensor Network enhanced Dynamic Multiproduct Formulas <https://arxiv.org/abs/2407.17405v2>`_, arXiv:2407.17405v2 [quant-ph].

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-mpf/blob/main/LICENSE.txt>`_


.. toctree::
  :hidden:

   Documentation Home <self>
   Installation Instructions <install>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   Explanations <explanations/index>
   API Reference <apidocs/index>
   GitHub <https://github.com/Qiskit/qiskit-addon-mpf>
   Release Notes <release-notes>
