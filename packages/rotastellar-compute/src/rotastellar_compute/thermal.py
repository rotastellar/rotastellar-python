"""
RotaStellar Compute - Thermal Simulation

Model heat rejection in vacuum for space-based compute systems.

subhadipmitra@: Thermal management is THE critical constraint for space computing.
In vacuum, you can only reject heat via radiation (Stefan-Boltzmann law).

Key insight: P = εσAT⁴ means radiator area scales with the FOURTH ROOT of power.
Doubling your compute requires only ~19% more radiator area, which is why
orbital compute can be power-dense despite the harsh thermal environment.

The simulation accounts for:
- Solar input (~1361 W/m² at 1 AU)
- Earth albedo (reflected sunlight, ~30% of solar)
- Earth IR (thermal emission, ~237 W/m²)
- Eclipse periods (thermal cycling)

Real systems also need active thermal control (pumped loops, heat pipes) but
this simplified model gives good first-order estimates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import math

from rotastellar import RotaStellarClient

# TODO(subhadipmitra): Add transient thermal analysis for eclipse crossings
# TODO: Model deployable radiators for high-power systems
# NOTE: Using single-node lumped-parameter model (adequate for sizing)


class OrbitType(Enum):
    """Types of orbits affecting thermal environment."""

    LEO = "leo"  # Low Earth Orbit (200-2000 km)
    MEO = "meo"  # Medium Earth Orbit (2000-35786 km)
    GEO = "geo"  # Geostationary Orbit (~35786 km)
    SSO = "sso"  # Sun-Synchronous Orbit
    POLAR = "polar"  # Polar orbit


@dataclass
class ThermalEnvironment:
    """Orbital thermal environment parameters.

    Attributes:
        orbit_type: Type of orbit
        altitude_km: Orbital altitude
        beta_angle_deg: Sun beta angle (affects solar exposure)
        eclipse_fraction: Fraction of orbit in eclipse (0-1)
        albedo_factor: Earth albedo factor (typically 0.3)
        earth_ir_w_m2: Earth infrared radiation (W/m²)
    """

    orbit_type: OrbitType = OrbitType.LEO
    altitude_km: float = 550.0
    beta_angle_deg: float = 0.0
    eclipse_fraction: float = 0.35
    albedo_factor: float = 0.3
    earth_ir_w_m2: float = 237.0


@dataclass
class ThermalConfig:
    """Thermal system configuration.

    Attributes:
        power_watts: Internal heat generation (W)
        radiator_area_m2: Radiator surface area (m²)
        radiator_emissivity: Radiator emissivity (0-1)
        solar_absorptivity: Solar absorptivity (0-1)
        mass_kg: System mass for thermal inertia
        specific_heat_j_kg_k: Specific heat capacity
    """

    power_watts: float = 500.0
    radiator_area_m2: float = 2.0
    radiator_emissivity: float = 0.85
    solar_absorptivity: float = 0.2
    mass_kg: float = 100.0
    specific_heat_j_kg_k: float = 900.0  # Typical for electronics


@dataclass
class ThermalResult:
    """Result of thermal simulation.

    Attributes:
        equilibrium_temp_c: Steady-state equilibrium temperature (°C)
        max_temp_c: Maximum temperature during orbit (°C)
        min_temp_c: Minimum temperature during orbit (°C)
        temp_swing_c: Temperature swing (max - min)
        power_dissipated_w: Heat dissipated through radiators
        thermal_margin_c: Margin below max operating temp
        warnings: List of thermal warnings
        time_series: Temperature time series (if requested)
    """

    equilibrium_temp_c: float
    max_temp_c: float
    min_temp_c: float
    temp_swing_c: float
    power_dissipated_w: float
    thermal_margin_c: float
    warnings: List[str] = field(default_factory=list)
    time_series: Optional[List[Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "equilibrium_temp_c": self.equilibrium_temp_c,
            "max_temp_c": self.max_temp_c,
            "min_temp_c": self.min_temp_c,
            "temp_swing_c": self.temp_swing_c,
            "power_dissipated_w": self.power_dissipated_w,
            "thermal_margin_c": self.thermal_margin_c,
            "warnings": self.warnings,
        }
        if self.time_series:
            result["time_series"] = self.time_series
        return result

    @property
    def is_safe(self) -> bool:
        """Check if temperatures are within safe operating range."""
        return -40 <= self.min_temp_c and self.max_temp_c <= 85


# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
SOLAR_CONSTANT = 1361.0  # W/m² at 1 AU


class ThermalSimulator:
    """Simulate thermal conditions for orbital compute systems.

    Example:
        >>> from rotastellar_compute import ThermalSimulator, ThermalConfig, ThermalEnvironment
        >>> simulator = ThermalSimulator(api_key="rs_live_xxx")
        >>>
        >>> config = ThermalConfig(power_watts=500, radiator_area_m2=2.0)
        >>> result = simulator.simulate(config)
        >>> print(f"Equilibrium: {result.equilibrium_temp_c:.1f}°C")
    """

    # Operating temperature limits (°C)
    MAX_OPERATING_TEMP = 85.0
    MIN_OPERATING_TEMP = -40.0
    OPTIMAL_TEMP = 25.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        **kwargs,
    ):
        """Initialize the thermal simulator.

        Args:
            api_key: RotaStellar API key
            client: Existing RotaStellarClient
            **kwargs: Additional client arguments
        """
        if client is not None:
            self._client = client
        else:
            self._client = RotaStellarClient(api_key=api_key, **kwargs)

    @property
    def client(self) -> RotaStellarClient:
        """Get the underlying API client."""
        return self._client

    def simulate(
        self,
        config: ThermalConfig,
        environment: Optional[ThermalEnvironment] = None,
        duration_hours: float = 24.0,
        time_step_minutes: float = 1.0,
        include_time_series: bool = False,
    ) -> ThermalResult:
        """Simulate thermal behavior.

        Args:
            config: Thermal system configuration
            environment: Orbital environment (default: LEO at 550 km)
            duration_hours: Simulation duration
            time_step_minutes: Time step for simulation
            include_time_series: Whether to include full time series

        Returns:
            Thermal simulation result
        """
        env = environment or ThermalEnvironment()

        # Calculate heat inputs
        solar_input = self._calc_solar_input(config, env)
        earth_ir_input = self._calc_earth_ir_input(config, env)
        albedo_input = self._calc_albedo_input(config, env)
        internal_heat = config.power_watts

        # Calculate equilibrium temperature
        total_input_sunlit = internal_heat + solar_input + earth_ir_input + albedo_input
        total_input_eclipse = internal_heat + earth_ir_input

        eq_temp_sunlit = self._calc_equilibrium_temp(
            total_input_sunlit, config.radiator_area_m2, config.radiator_emissivity
        )
        eq_temp_eclipse = self._calc_equilibrium_temp(
            total_input_eclipse, config.radiator_area_m2, config.radiator_emissivity
        )

        # Calculate time-dependent temperatures
        orbital_period_min = self._calc_orbital_period(env.altitude_km)
        time_series = []

        if include_time_series:
            time_series = self._simulate_time_series(
                config, env, duration_hours, time_step_minutes, orbital_period_min
            )
            temps = [t["temperature_c"] for t in time_series]
            max_temp = max(temps)
            min_temp = min(temps)
        else:
            # Estimate max/min from equilibrium
            max_temp = eq_temp_sunlit
            min_temp = eq_temp_eclipse

        # Weight equilibrium by sunlit/eclipse fraction
        equilibrium = (
            eq_temp_sunlit * (1 - env.eclipse_fraction)
            + eq_temp_eclipse * env.eclipse_fraction
        )

        # Calculate power dissipated
        eq_temp_k = equilibrium + 273.15
        power_dissipated = (
            config.radiator_emissivity
            * STEFAN_BOLTZMANN
            * config.radiator_area_m2
            * eq_temp_k**4
        )

        # Generate warnings
        warnings = []
        if max_temp > self.MAX_OPERATING_TEMP:
            warnings.append(f"Max temperature ({max_temp:.1f}°C) exceeds safe limit")
        if min_temp < self.MIN_OPERATING_TEMP:
            warnings.append(f"Min temperature ({min_temp:.1f}°C) below safe limit")
        if max_temp - min_temp > 60:
            warnings.append("High thermal cycling may reduce component lifetime")

        thermal_margin = self.MAX_OPERATING_TEMP - max_temp

        return ThermalResult(
            equilibrium_temp_c=round(equilibrium, 2),
            max_temp_c=round(max_temp, 2),
            min_temp_c=round(min_temp, 2),
            temp_swing_c=round(max_temp - min_temp, 2),
            power_dissipated_w=round(power_dissipated, 2),
            thermal_margin_c=round(thermal_margin, 2),
            warnings=warnings,
            time_series=time_series if include_time_series else None,
        )

    def _calc_solar_input(self, config: ThermalConfig, env: ThermalEnvironment) -> float:
        """Calculate solar heat input during sunlit portion."""
        # Effective solar exposure area (assume 1/4 of radiator area)
        solar_area = config.radiator_area_m2 * 0.25
        return SOLAR_CONSTANT * config.solar_absorptivity * solar_area

    def _calc_earth_ir_input(self, config: ThermalConfig, env: ThermalEnvironment) -> float:
        """Calculate Earth infrared heat input."""
        # View factor to Earth decreases with altitude
        view_factor = self._earth_view_factor(env.altitude_km)
        ir_area = config.radiator_area_m2 * 0.5  # Assume half facing Earth
        return env.earth_ir_w_m2 * view_factor * ir_area * config.radiator_emissivity

    def _calc_albedo_input(self, config: ThermalConfig, env: ThermalEnvironment) -> float:
        """Calculate Earth albedo heat input."""
        view_factor = self._earth_view_factor(env.altitude_km)
        albedo_area = config.radiator_area_m2 * 0.25
        # Only in sunlit portion
        return SOLAR_CONSTANT * env.albedo_factor * view_factor * albedo_area * config.solar_absorptivity

    def _earth_view_factor(self, altitude_km: float) -> float:
        """Calculate view factor to Earth from altitude."""
        earth_radius = 6371.0  # km
        r = earth_radius + altitude_km
        return (earth_radius / r) ** 2

    def _calc_equilibrium_temp(
        self, heat_input_w: float, area_m2: float, emissivity: float
    ) -> float:
        """Calculate equilibrium temperature in Celsius."""
        if area_m2 <= 0 or emissivity <= 0:
            return float('inf')

        # Stefan-Boltzmann law: P = εσAT⁴
        # T = (P / (εσA))^(1/4)
        temp_k = (heat_input_w / (emissivity * STEFAN_BOLTZMANN * area_m2)) ** 0.25
        return temp_k - 273.15

    def _calc_orbital_period(self, altitude_km: float) -> float:
        """Calculate orbital period in minutes."""
        earth_radius = 6371.0
        earth_mu = 398600.4418  # km³/s²

        a = earth_radius + altitude_km
        period_s = 2 * math.pi * math.sqrt(a**3 / earth_mu)
        return period_s / 60

    def _simulate_time_series(
        self,
        config: ThermalConfig,
        env: ThermalEnvironment,
        duration_hours: float,
        time_step_minutes: float,
        orbital_period_min: float,
    ) -> List[Dict[str, float]]:
        """Simulate temperature time series."""
        results = []
        steps = int(duration_hours * 60 / time_step_minutes)

        # Initial temperature
        temp_c = self.OPTIMAL_TEMP
        temp_k = temp_c + 273.15

        # Calculate heat inputs
        internal = config.power_watts
        earth_ir = self._calc_earth_ir_input(config, env)

        for i in range(steps):
            time_min = i * time_step_minutes
            orbit_phase = (time_min % orbital_period_min) / orbital_period_min

            # Determine if in eclipse
            in_eclipse = orbit_phase > (1 - env.eclipse_fraction)

            if in_eclipse:
                heat_in = internal + earth_ir
            else:
                solar = self._calc_solar_input(config, env)
                albedo = self._calc_albedo_input(config, env)
                heat_in = internal + solar + earth_ir + albedo

            # Heat radiated
            heat_out = (
                config.radiator_emissivity
                * STEFAN_BOLTZMANN
                * config.radiator_area_m2
                * temp_k**4
            )

            # Temperature change
            net_heat = heat_in - heat_out
            thermal_mass = config.mass_kg * config.specific_heat_j_kg_k
            dt_k = (net_heat * time_step_minutes * 60) / thermal_mass

            temp_k += dt_k
            temp_c = temp_k - 273.15

            results.append({
                "time_minutes": time_min,
                "temperature_c": round(temp_c, 2),
                "in_eclipse": in_eclipse,
            })

        return results

    def design_radiator(
        self,
        power_watts: float,
        target_temp_c: float = 50.0,
        environment: Optional[ThermalEnvironment] = None,
    ) -> Dict[str, Any]:
        """Design radiator for given power and target temperature.

        Args:
            power_watts: Heat to dissipate
            target_temp_c: Target operating temperature
            environment: Orbital environment

        Returns:
            Radiator design parameters
        """
        env = environment or ThermalEnvironment()
        target_k = target_temp_c + 273.15

        # Account for environmental heat loads
        # Estimate additional heat input (solar + Earth IR)
        # This is simplified - full calculation would iterate
        environmental_fraction = 0.2

        total_heat = power_watts * (1 + environmental_fraction)

        # Required radiator area
        emissivity = 0.85  # Typical high-emissivity coating
        required_area = total_heat / (emissivity * STEFAN_BOLTZMANN * target_k**4)

        # Add margin
        required_area *= 1.25

        return {
            "required_area_m2": round(required_area, 3),
            "power_watts": power_watts,
            "target_temp_c": target_temp_c,
            "recommended_emissivity": emissivity,
            "design_margin": 0.25,
        }
