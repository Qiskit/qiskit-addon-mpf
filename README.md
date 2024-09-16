<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-mpf.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-mpf/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-mpf?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-mpf/)
  <!--[![DOI](https://zenodo.org/badge/TODO.svg)](https://zenodo.org/badge/latestdoi/TODO)-->
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-mpf?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-mpf.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-mpf/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-mpf/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-mpf/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-addon-mpf/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-addon-mpf?branch=main)

# Qiskit addon: multi-product formulas (MPF)

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://docs.quantum.ibm.com/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for multi-product formulas (MPFs).
These can be used to reduce the Trotter error of Hamiltonian dynamics.

This package currently contains the following submodules:

- `qiskit_addon_mpf.static` for working with static MPFs [1-2](#references)

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-mpf/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-mpf'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-mpf/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-mpf).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-mpf/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-mpf/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### References

1. A. Carrera Vazquez, D. J. Egger, D. Ochsner, and S. Wörner, [Well-conditioned multi-product formulas for hardware-friendly Hamiltonian simulation](https://quantum-journal.org/papers/q-2023-07-25-1067/), Quantum 7, 1067 (2023).
2. S. Zhuk, N. Robertson, and S. Bravyi, [Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation](https://arxiv.org/abs/2306.12569v2), arXiv:2306.12569 [quant-ph].

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
