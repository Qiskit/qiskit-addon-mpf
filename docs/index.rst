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
2. S. Zhuk, N. Robertson, and S. Bravyi, `Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation <https://arxiv.org/abs/2306.12569v2>`_, arXiv:2306.12569 [quant-ph].

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
   API Reference <apidocs/qiskit_addon_mpf>
   GitHub <https://github.com/Qiskit/qiskit-addon-mpf>
   Release Notes <release-notes>
