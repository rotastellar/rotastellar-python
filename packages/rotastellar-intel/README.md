# rotastellar-intel

**Orbital Intelligence & Space Situational Awareness**

Track satellites, parse TLEs, analyze conjunctions, and detect orbital patterns.

## Installation

```bash
pip install rotastellar-intel
```

## Quick Start

### TLE Parsing

```python
from rotastellar_intel import TLE

# Parse a Two-Line Element set
tle_lines = [
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9025",
    "2 25544  51.6400 208.9163 0006703  40.5765  35.4667 15.49560927421258"
]

tle = TLE.parse(tle_lines)
print(f"Satellite: {tle.name}")
print(f"NORAD ID: {tle.norad_id}")
print(f"Inclination: {tle.inclination_deg:.2f}°")
print(f"Period: {tle.orbital_period_minutes:.2f} minutes")

# Get position at epoch
position = tle.propagate()
print(f"Position: {position.latitude:.2f}°, {position.longitude:.2f}°")
```

### Satellite Tracking

```python
from rotastellar_intel import Tracker, GroundStation
from rotastellar import Position

# Create a tracker
tracker = Tracker()
tracker.add_tle("ISS", tle)

# Get current position
pos = tracker.get_position("ISS")

# Calculate passes over a ground station
gs = GroundStation(
    name="KSC",
    position=Position(28.5729, -80.6490, 0.0),
    min_elevation_deg=10.0
)
passes = tracker.predict_passes("ISS", gs, hours=24)
for p in passes:
    print(f"AOS: {p.aos}, Max El: {p.max_elevation_deg:.1f}°")
```

### Conjunction Analysis

```python
from rotastellar_intel import ConjunctionAnalyzer, RiskLevel

analyzer = ConjunctionAnalyzer()

# Analyze collision probability
conjunction = analyzer.analyze(
    primary_id="ISS",
    secondary_id="DEBRIS-12345",
    miss_distance_km=0.5,
    relative_velocity_km_s=10.0
)

print(f"Risk Level: {conjunction.risk_level}")
print(f"Collision Probability: {conjunction.collision_probability:.2e}")

if conjunction.risk_level == RiskLevel.CRITICAL:
    print("⚠️  Maneuver recommended!")
```

### Pattern Detection

```python
from rotastellar_intel import PatternDetector, PatternType

detector = PatternDetector()

# Detect maneuvers from TLE history
patterns = detector.detect(satellite_id="STARLINK-1234", days=30)

for pattern in patterns:
    if pattern.pattern_type == PatternType.ORBIT_RAISE:
        print(f"Orbit raise detected: +{pattern.delta_altitude_km:.1f} km")
    elif pattern.pattern_type == PatternType.MANEUVER:
        print(f"Maneuver: Δv = {pattern.delta_v_m_s:.2f} m/s")
```

## Features

- **TLE Parsing** — Full Two-Line Element support with SGP4 propagation
- **Satellite Tracking** — Real-time position and pass prediction
- **Conjunction Analysis** — Collision probability using NASA CARA methodology
- **Pattern Detection** — Maneuver detection, anomaly identification

## Links

- **Website:** https://rotastellar.com/products/orbital-intelligence
- **Documentation:** https://docs.rotastellar.com/sdks/python/intel
- **Main SDK:** https://pypi.org/project/rotastellar/

## Author

Created by [Subhadip Mitra](mailto:subhadipmitra@rotastellar.com) at [RotaStellar](https://rotastellar.com).

## License

MIT License — Copyright (c) 2026 RotaStellar
