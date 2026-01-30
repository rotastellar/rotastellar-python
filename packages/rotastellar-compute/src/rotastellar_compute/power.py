"""
RotaStellar Compute - Power Analysis

Solar panel and battery sizing for orbital compute systems.

subhadipmitra@: Power is the second critical constraint after thermal (they're related).
The solar constant is ~1361 W/m² at 1 AU, but effective collection depends on:
- Solar cell efficiency (20-30% for modern cells)
- Incidence angle (cos loss)
- Temperature (cells degrade ~0.5%/°C above 25°C)
- Radiation degradation (few % per year in LEO)
- Eclipse fraction (~35% for ISS orbit)

Battery sizing is driven by eclipse duration. LEO eclipses are short (~35 min)
but frequent, so cycle life matters. Li-ion needs ~15-20% oversizing for capacity
fade over mission life.

Rule of thumb: 1 kW average power needs ~5 m² of solar panel and ~500 Wh battery.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import math

from rotastellar import RotaStellarClient

# TODO(subhadipmitra): Add radiation degradation model for long missions
# TODO: Model seasonal variation in eclipse fraction
# NOTE: Using beginning-of-life (BOL) values; EOL is typically 80-90% of BOL


class SolarCellType(Enum):
    """Types of solar cells."""

    SILICON = "silicon"  # Traditional silicon cells
    TRIPLE_JUNCTION = "triple_junction"  # GaInP/GaAs/Ge
    PEROVSKITE = "perovskite"  # Emerging technology


class BatteryChemistry(Enum):
    """Battery chemistry types."""

    LITHIUM_ION = "lithium_ion"
    LITHIUM_POLYMER = "lithium_polymer"
    NICKEL_HYDROGEN = "nickel_hydrogen"


@dataclass
class PowerProfile:
    """Power consumption profile.

    Attributes:
        average_power_w: Average power consumption (W)
        peak_power_w: Peak power consumption (W)
        idle_power_w: Idle/standby power (W)
        duty_cycle: Fraction of time at peak power (0-1)
    """

    average_power_w: float
    peak_power_w: Optional[float] = None
    idle_power_w: Optional[float] = None
    duty_cycle: float = 0.5

    def __post_init__(self):
        if self.peak_power_w is None:
            self.peak_power_w = self.average_power_w * 1.5
        if self.idle_power_w is None:
            self.idle_power_w = self.average_power_w * 0.2


@dataclass
class SolarConfig:
    """Solar panel configuration.

    Attributes:
        cell_type: Type of solar cells
        efficiency: Cell efficiency (0-1)
        degradation_per_year: Efficiency loss per year
        panel_area_m2: Total panel area
        tracking: Whether panels track the sun
    """

    cell_type: SolarCellType = SolarCellType.TRIPLE_JUNCTION
    efficiency: float = 0.30  # 30% for triple junction
    degradation_per_year: float = 0.02
    panel_area_m2: Optional[float] = None
    tracking: bool = False


@dataclass
class BatteryConfig:
    """Battery configuration.

    Attributes:
        chemistry: Battery chemistry
        capacity_wh: Total capacity in Wh
        depth_of_discharge: Maximum DoD (0-1)
        cycle_efficiency: Round-trip efficiency
        specific_energy_wh_kg: Energy density
    """

    chemistry: BatteryChemistry = BatteryChemistry.LITHIUM_ION
    capacity_wh: Optional[float] = None
    depth_of_discharge: float = 0.8
    cycle_efficiency: float = 0.95
    specific_energy_wh_kg: float = 200.0


@dataclass
class PowerBudget:
    """Complete power budget analysis.

    Attributes:
        power_required_w: Required continuous power
        solar_power_generated_w: Solar power during sunlight
        battery_capacity_wh: Required battery capacity
        solar_panel_area_m2: Required solar panel area
        battery_mass_kg: Estimated battery mass
        solar_panel_mass_kg: Estimated solar panel mass
        eclipse_duration_min: Maximum eclipse duration
        positive_margin: Whether power budget is positive
        margin_percent: Power margin percentage
        warnings: List of warnings
    """

    power_required_w: float
    solar_power_generated_w: float
    battery_capacity_wh: float
    solar_panel_area_m2: float
    battery_mass_kg: float
    solar_panel_mass_kg: float
    eclipse_duration_min: float
    positive_margin: bool
    margin_percent: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "power_required_w": self.power_required_w,
            "solar_power_generated_w": self.solar_power_generated_w,
            "battery_capacity_wh": self.battery_capacity_wh,
            "solar_panel_area_m2": self.solar_panel_area_m2,
            "battery_mass_kg": self.battery_mass_kg,
            "solar_panel_mass_kg": self.solar_panel_mass_kg,
            "eclipse_duration_min": self.eclipse_duration_min,
            "positive_margin": self.positive_margin,
            "margin_percent": self.margin_percent,
            "warnings": self.warnings,
        }


# Physical constants
SOLAR_CONSTANT = 1361.0  # W/m² at 1 AU


class PowerAnalyzer:
    """Analyze power requirements for orbital compute systems.

    Example:
        >>> from rotastellar_compute import PowerAnalyzer, PowerProfile
        >>> analyzer = PowerAnalyzer(api_key="rs_live_xxx")
        >>>
        >>> profile = PowerProfile(average_power_w=500, peak_power_w=800)
        >>> budget = analyzer.analyze(profile, mission_duration_years=5)
        >>> print(f"Solar panel area: {budget.solar_panel_area_m2:.2f} m²")
    """

    # Default assumptions
    SOLAR_PANEL_SPECIFIC_POWER = 100.0  # W/kg for modern panels
    DESIGN_MARGIN = 0.2  # 20% margin

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        orbit_altitude_km: float = 550.0,
        **kwargs,
    ):
        """Initialize the power analyzer.

        Args:
            api_key: RotaStellar API key
            client: Existing RotaStellarClient
            orbit_altitude_km: Default orbit altitude
            **kwargs: Additional client arguments
        """
        if client is not None:
            self._client = client
        else:
            self._client = RotaStellarClient(api_key=api_key, **kwargs)

        self.orbit_altitude_km = orbit_altitude_km

    @property
    def client(self) -> RotaStellarClient:
        """Get the underlying API client."""
        return self._client

    def analyze(
        self,
        profile: PowerProfile,
        solar_config: Optional[SolarConfig] = None,
        battery_config: Optional[BatteryConfig] = None,
        orbit_altitude_km: Optional[float] = None,
        mission_duration_years: float = 5.0,
    ) -> PowerBudget:
        """Analyze power budget for a mission.

        Args:
            profile: Power consumption profile
            solar_config: Solar panel configuration
            battery_config: Battery configuration
            orbit_altitude_km: Orbit altitude
            mission_duration_years: Mission duration for degradation

        Returns:
            Complete power budget analysis
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km
        solar = solar_config or SolarConfig()
        battery = battery_config or BatteryConfig()

        # Calculate orbital parameters
        orbital_period_min = self._orbital_period(altitude)
        eclipse_fraction = self._eclipse_fraction(altitude)
        eclipse_duration = orbital_period_min * eclipse_fraction
        sunlight_duration = orbital_period_min * (1 - eclipse_fraction)

        # Power required with margin
        power_required = profile.average_power_w * (1 + self.DESIGN_MARGIN)

        # Account for degradation at end of life
        eol_efficiency = solar.efficiency * (1 - solar.degradation_per_year * mission_duration_years)

        # Required solar panel area
        if solar.panel_area_m2:
            panel_area = solar.panel_area_m2
        else:
            # Size panels to power load + charge batteries during sunlight
            # Energy needed per orbit
            orbit_energy_wh = power_required * orbital_period_min / 60

            # Solar generation only during sunlight
            # Account for cosine losses (average ~0.7 for body-mounted)
            cosine_factor = 0.9 if solar.tracking else 0.7

            solar_power_needed = orbit_energy_wh / (sunlight_duration / 60)
            panel_area = solar_power_needed / (SOLAR_CONSTANT * eol_efficiency * cosine_factor)

        # Calculate actual solar power generated
        cosine_factor = 0.9 if solar.tracking else 0.7
        solar_power = SOLAR_CONSTANT * panel_area * eol_efficiency * cosine_factor

        # Battery sizing
        # Energy needed during eclipse
        eclipse_energy_wh = power_required * eclipse_duration / 60

        # Account for DoD and efficiency
        battery_capacity = eclipse_energy_wh / (battery.depth_of_discharge * battery.cycle_efficiency)

        # Add margin
        battery_capacity *= (1 + self.DESIGN_MARGIN)

        if battery.capacity_wh:
            battery_capacity = max(battery_capacity, battery.capacity_wh)

        # Mass estimates
        battery_mass = battery_capacity / battery.specific_energy_wh_kg
        solar_mass = solar_power / self.SOLAR_PANEL_SPECIFIC_POWER

        # Check margin
        available_power = solar_power * (sunlight_duration / orbital_period_min)
        margin_percent = ((available_power - power_required) / power_required) * 100
        positive_margin = margin_percent > 0

        # Generate warnings
        warnings = []
        if not positive_margin:
            warnings.append("Negative power margin - increase solar panel area")
        if battery_capacity > 1000:
            warnings.append("Large battery capacity may impact mass budget")
        if eol_efficiency < 0.2:
            warnings.append("Significant solar cell degradation expected over mission life")
        if eclipse_duration > 40:
            warnings.append("Long eclipse duration - ensure adequate battery capacity")

        return PowerBudget(
            power_required_w=round(power_required, 1),
            solar_power_generated_w=round(solar_power, 1),
            battery_capacity_wh=round(battery_capacity, 1),
            solar_panel_area_m2=round(panel_area, 3),
            battery_mass_kg=round(battery_mass, 2),
            solar_panel_mass_kg=round(solar_mass, 2),
            eclipse_duration_min=round(eclipse_duration, 1),
            positive_margin=positive_margin,
            margin_percent=round(margin_percent, 1),
            warnings=warnings,
        )

    def size_solar_panels(
        self,
        power_required_w: float,
        orbit_altitude_km: Optional[float] = None,
        cell_type: SolarCellType = SolarCellType.TRIPLE_JUNCTION,
        mission_years: float = 5.0,
    ) -> Dict[str, Any]:
        """Size solar panels for power requirement.

        Args:
            power_required_w: Required power
            orbit_altitude_km: Orbit altitude
            cell_type: Solar cell type
            mission_years: Mission duration

        Returns:
            Solar panel sizing
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km

        # Get efficiency for cell type
        efficiencies = {
            SolarCellType.SILICON: 0.20,
            SolarCellType.TRIPLE_JUNCTION: 0.30,
            SolarCellType.PEROVSKITE: 0.25,
        }
        efficiency = efficiencies[cell_type]
        degradation = 0.02  # 2% per year

        # End of life efficiency
        eol_efficiency = efficiency * (1 - degradation * mission_years)

        # Account for eclipse
        eclipse_fraction = self._eclipse_fraction(altitude)
        sunlight_fraction = 1 - eclipse_fraction

        # Required solar power (including margin)
        required_solar = power_required_w / sunlight_fraction * (1 + self.DESIGN_MARGIN)

        # Panel area
        cosine_factor = 0.7  # Body-mounted average
        panel_area = required_solar / (SOLAR_CONSTANT * eol_efficiency * cosine_factor)

        return {
            "panel_area_m2": round(panel_area, 3),
            "cell_type": cell_type.value,
            "bol_efficiency": efficiency,
            "eol_efficiency": round(eol_efficiency, 3),
            "solar_power_w": round(required_solar, 1),
            "mass_estimate_kg": round(required_solar / self.SOLAR_PANEL_SPECIFIC_POWER, 2),
        }

    def size_battery(
        self,
        power_required_w: float,
        orbit_altitude_km: Optional[float] = None,
        chemistry: BatteryChemistry = BatteryChemistry.LITHIUM_ION,
    ) -> Dict[str, Any]:
        """Size battery for eclipse power.

        Args:
            power_required_w: Required power
            orbit_altitude_km: Orbit altitude
            chemistry: Battery chemistry

        Returns:
            Battery sizing
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km

        # Battery characteristics
        characteristics = {
            BatteryChemistry.LITHIUM_ION: {
                "specific_energy": 200,
                "dod": 0.8,
                "efficiency": 0.95,
                "cycle_life": 5000,
            },
            BatteryChemistry.LITHIUM_POLYMER: {
                "specific_energy": 180,
                "dod": 0.7,
                "efficiency": 0.93,
                "cycle_life": 3000,
            },
            BatteryChemistry.NICKEL_HYDROGEN: {
                "specific_energy": 60,
                "dod": 0.8,
                "efficiency": 0.85,
                "cycle_life": 50000,
            },
        }
        chars = characteristics[chemistry]

        # Eclipse duration
        orbital_period = self._orbital_period(altitude)
        eclipse_fraction = self._eclipse_fraction(altitude)
        eclipse_min = orbital_period * eclipse_fraction

        # Energy needed
        eclipse_energy = power_required_w * eclipse_min / 60

        # Required capacity
        capacity = eclipse_energy / (chars["dod"] * chars["efficiency"])
        capacity *= (1 + self.DESIGN_MARGIN)

        # Mass
        mass = capacity / chars["specific_energy"]

        # Cycles per year
        orbits_per_day = 24 * 60 / orbital_period
        cycles_per_year = orbits_per_day * 365

        return {
            "capacity_wh": round(capacity, 1),
            "chemistry": chemistry.value,
            "mass_kg": round(mass, 2),
            "eclipse_duration_min": round(eclipse_min, 1),
            "depth_of_discharge": chars["dod"],
            "cycles_per_year": round(cycles_per_year),
            "expected_life_years": round(chars["cycle_life"] / cycles_per_year, 1),
        }

    def _orbital_period(self, altitude_km: float) -> float:
        """Calculate orbital period in minutes."""
        earth_radius = 6371.0
        earth_mu = 398600.4418

        a = earth_radius + altitude_km
        period_s = 2 * math.pi * math.sqrt(a**3 / earth_mu)
        return period_s / 60

    def _eclipse_fraction(self, altitude_km: float) -> float:
        """Estimate eclipse fraction for circular orbit.

        Simplified model assuming circular orbit and average geometry.
        """
        earth_radius = 6371.0
        r = earth_radius + altitude_km

        # Angular radius of Earth as seen from satellite
        sin_rho = earth_radius / r

        # Eclipse half-angle (simplified)
        eclipse_half_angle = math.asin(sin_rho)

        # Fraction of orbit in eclipse
        return eclipse_half_angle / math.pi
