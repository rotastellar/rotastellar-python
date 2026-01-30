# rotastellar-compute

**Orbital Compute Planning & Simulation**

Feasibility analysis, thermal simulation, power budgeting, and latency modeling for space-based computing.

## Installation

```bash
pip install rotastellar-compute
```

## Quick Start

### Feasibility Analysis

```python
from rotastellar_compute import FeasibilityCalculator, WorkloadProfile, WorkloadType

# Create a calculator for 550km altitude
calc = FeasibilityCalculator(altitude_km=550.0)

# Define your workload
profile = WorkloadProfile(
    workload_type=WorkloadType.INFERENCE,
    compute_power_kw=10.0,
    memory_gb=32.0,
    latency_requirement_ms=100.0
)

# Analyze feasibility
result = calc.analyze(profile)
print(f"Feasible: {result.feasible}")
print(f"Rating: {result.rating}")  # EXCELLENT, GOOD, MARGINAL, or NOT_FEASIBLE
print(f"Thermal margin: {result.thermal_margin_percent:.1f}%")
print(f"Power margin: {result.power_margin_percent:.1f}%")
```

### Thermal Simulation

```python
from rotastellar_compute import ThermalSimulator, ThermalConfig, ThermalEnvironment

# Create simulator
sim = ThermalSimulator()

# Configure for 500W heat dissipation
config = ThermalConfig.for_power(500.0)

# LEO environment at 550km
env = ThermalEnvironment.leo(altitude_km=550.0)

# Run simulation
result = sim.simulate(config, env)
print(f"Equilibrium temperature: {result.equilibrium_temp_c:.1f}°C")
print(f"Max temperature: {result.max_temp_c:.1f}°C")
print(f"Radiator area required: {result.radiator_area_m2:.2f} m²")
```

### Power Analysis

```python
from rotastellar_compute import PowerAnalyzer, PowerProfile, SolarConfig, BatteryConfig

# Analyzer for 550km orbit
analyzer = PowerAnalyzer(altitude_km=550.0)

# Power requirements
profile = PowerProfile(
    average_power_w=500.0,
    peak_power_w=800.0
)

# Optional: customize solar and battery
solar = SolarConfig(efficiency=0.30, degradation_per_year=0.02)
battery = BatteryConfig(depth_of_discharge=0.40, efficiency=0.95)

# Analyze
budget = analyzer.analyze(profile, solar_config=solar, battery_config=battery)
print(f"Solar panel area: {budget.solar_panel_area_m2:.2f} m²")
print(f"Battery capacity: {budget.battery_capacity_wh:.0f} Wh")
print(f"Eclipse duration: {budget.eclipse_duration_min:.1f} minutes")
```

### Latency Modeling

```python
from rotastellar_compute import LatencySimulator

# Simulator for 550km altitude
sim = LatencySimulator(altitude_km=550.0)

# Simulate with 100ms processing time
result = sim.simulate(processing_time_ms=100.0)
print(f"Propagation delay: {result.propagation_delay_ms:.1f} ms")
print(f"Processing time: {result.processing_time_ms:.1f} ms")
print(f"Total latency: {result.total_latency_ms:.1f} ms")

# Compare different altitudes
altitudes = [400.0, 550.0, 800.0, 1200.0]
comparison = sim.compare_altitudes(altitudes)
for alt_result in comparison:
    print(f"{alt_result.altitude_km}km: {alt_result.typical_latency_ms:.1f}ms")
```

## Features

- **Feasibility Analysis** — Evaluate workload suitability for orbital compute
- **Thermal Simulation** — Model heat rejection using Stefan-Boltzmann law
- **Power Analysis** — Solar panel and battery sizing for orbital systems
- **Latency Modeling** — End-to-end latency for space-ground communication

## Links

- **Website:** https://rotastellar.com/products/compute
- **Documentation:** https://docs.rotastellar.com/sdks/python/compute
- **Main SDK:** https://pypi.org/project/rotastellar/

## Author

Created by [Subhadip Mitra](mailto:subhadipmitra@rotastellar.com) at [RotaStellar](https://rotastellar.com).

## License

MIT License — Copyright (c) 2026 RotaStellar
