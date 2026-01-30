<p align="center">
  <img src="assets/logo-dark.jpg" alt="RotaStellar" width="400">
</p>

<p align="center">
  <strong>Python SDK for Space Computing Infrastructure</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/rotastellar/"><img src="https://img.shields.io/pypi/v/rotastellar?color=blue&label=rotastellar" alt="PyPI"></a>
  <a href="https://pypi.org/project/rotastellar-compute/"><img src="https://img.shields.io/pypi/v/rotastellar-compute?color=blue&label=compute" alt="PyPI"></a>
  <a href="https://pypi.org/project/rotastellar-intel/"><img src="https://img.shields.io/pypi/v/rotastellar-intel?color=blue&label=intel" alt="PyPI"></a>
  <a href="https://pypi.org/project/rotastellar-distributed/"><img src="https://img.shields.io/pypi/v/rotastellar-distributed?color=blue&label=distributed" alt="PyPI"></a>
</p>

<p align="center">
  <a href="https://github.com/rotastellar/rotastellar-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9+-blue" alt="Python"></a>
  <a href="https://docs.rotastellar.com"><img src="https://img.shields.io/badge/docs-rotastellar.com-blue" alt="Documentation"></a>
</p>

---

Plan, simulate, and operate orbital data centers and space intelligence systems.

## Packages

| Package | Description |
|---------|-------------|
| [rotastellar](./packages/rotastellar) | Core types — Position, Orbit, Satellite, TimeRange |
| [rotastellar-compute](./packages/rotastellar-compute) | Feasibility, thermal, power, and latency analysis |
| [rotastellar-intel](./packages/rotastellar-intel) | Satellite tracking, TLE parsing, conjunction analysis |
| [rotastellar-distributed](./packages/rotastellar-distributed) | Federated learning, model partitioning, mesh routing |

## Installation

```bash
# Core SDK
pip install rotastellar

# All packages
pip install rotastellar rotastellar-compute rotastellar-intel rotastellar-distributed
```

## Quick Start

```python
from rotastellar import Position, Orbit, Satellite
from rotastellar_compute import FeasibilityCalculator, WorkloadProfile, WorkloadType

# Define a position
ksc = Position(latitude=28.5729, longitude=-80.6490, altitude_km=0.0)

# Analyze workload feasibility
calc = FeasibilityCalculator(altitude_km=550.0)
profile = WorkloadProfile(
    workload_type=WorkloadType.INFERENCE,
    compute_power_kw=10.0,
    memory_gb=32.0
)
result = calc.analyze(profile)
print(f"Feasible: {result.feasible}, Rating: {result.rating}")
```

## Links

- **Website:** https://rotastellar.com
- **Documentation:** https://docs.rotastellar.com/sdks/python
- **Node.js SDK:** https://github.com/rotastellar/rotastellar-node
- **Rust SDK:** https://github.com/rotastellar/rotastellar-rust

## Author

Created by [Subhadip Mitra](mailto:subhadipmitra@rotastellar.com) at [RotaStellar](https://rotastellar.com).

## License

MIT License — Copyright (c) 2026 RotaStellar
