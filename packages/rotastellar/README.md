# rotastellar

**Python SDK for RotaStellar — Space Computing Infrastructure**

Plan, simulate, and operate orbital data centers and space intelligence systems.

## Installation

```bash
pip install rotastellar
```

## Quick Start

```python
from rotastellar import Position, Orbit, Satellite, TimeRange

# Create a geographic position (e.g., Kennedy Space Center)
ksc = Position(latitude=28.5729, longitude=-80.6490, altitude_km=0.0)
print(f"KSC: {ksc.latitude}°N, {ksc.longitude}°W")

# Define an ISS-like orbit
orbit = Orbit(
    semi_major_axis_km=6778.0,
    eccentricity=0.0001,
    inclination_deg=51.6,
    raan_deg=100.0,
    arg_periapsis_deg=90.0,
    true_anomaly_deg=0.0
)
print(f"Orbital period: {orbit.period_minutes:.1f} minutes")
print(f"Apogee: {orbit.apogee_km:.1f} km, Perigee: {orbit.perigee_km:.1f} km")

# Create a satellite
sat = Satellite(
    id="ISS",
    name="International Space Station",
    norad_id=25544,
    position=Position(latitude=45.0, longitude=-122.0, altitude_km=408.0),
    orbit=orbit
)
```

## Features

- **Position** — Geographic coordinates with altitude
- **Orbit** — Keplerian orbital elements with derived properties
- **Satellite** — Satellite metadata and state
- **TimeRange** — Time window specifications
- **Validation** — Automatic parameter validation

## Related Packages

| Package | Description |
|---------|-------------|
| [rotastellar-intel](https://pypi.org/project/rotastellar-intel/) | Satellite tracking, TLE parsing, conjunction analysis |
| [rotastellar-compute](https://pypi.org/project/rotastellar-compute/) | Thermal, power, latency, and feasibility analysis |
| [rotastellar-distributed](https://pypi.org/project/rotastellar-distributed/) | Federated learning, model partitioning, mesh routing |

## Links

- **Website:** https://rotastellar.com
- **Documentation:** https://docs.rotastellar.com
- **GitHub:** https://github.com/rotastellar/rotastellar-python

## Author

Created by [Subhadip Mitra](mailto:subhadipmitra@rotastellar.com) at [RotaStellar](https://rotastellar.com).

## License

MIT License — Copyright (c) 2026 RotaStellar
