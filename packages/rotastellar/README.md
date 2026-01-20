# rotastellar

**Python SDK for RotaStellar - Space Computing Infrastructure**

Plan, simulate, and operate orbital data centers and space intelligence systems.

🚀 **Launching Q1 2026**

## Installation

```bash
pip install rotastellar
```

## Overview

RotaStellar provides software tools for:

- **Orbital Compute Suite** — Plan and simulate space-based data centers
- **Orbital Intelligence Platform** — Track, analyze, and monitor orbital activity

## Coming Soon

```python
from rotastellar import OrbitalIntel

client = OrbitalIntel(api_key="...")

# Track any satellite
iss = client.satellite("ISS")
pos = iss.position()
print(f"ISS: {pos.lat}, {pos.lon}")

# Get conjunction alerts
alerts = client.conjunctions(
    satellite="starlink-1234",
    threshold_km=1.0
)
```

## Related Packages

- [rotastellar-compute](https://pypi.org/project/rotastellar-compute/) — Orbital compute planning tools
- [rotastellar-intel](https://pypi.org/project/rotastellar-intel/) — Orbital intelligence tools

## Links

- **Website:** https://rotastellar.com
- **Documentation:** https://rotastellar.com/docs
- **GitHub:** https://github.com/rotastellar/rotastellar-python

## Part of Rota, Inc.

- [rotalabs.ai](https://rotalabs.ai) — Trust Intelligence Research
- [rotascale.com](https://rotascale.com) — Enterprise AI & Data

## License

MIT License — Copyright (c) 2026 Rota, Inc.
