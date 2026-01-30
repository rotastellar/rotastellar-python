"""
RotaStellar Compute - Latency Simulation

End-to-end latency modeling for orbital compute systems.

subhadipmitra@: Latency for orbital compute has several components:
1. Propagation delay: ~2ms for 550km LEO (speed of light)
2. Ground station queuing: variable, depends on traffic
3. ISL hops: ~0.01ms per 1000km (negligible vs ground backhaul!)
4. Processing: workload-dependent

The counterintuitive insight: for multi-region workloads, LEO ISL mesh can be
FASTER than terrestrial fiber because light in vacuum > light in glass (0.67c).
SF to NYC: ~21ms via fiber, ~15ms via 550km LEO ISL constellation.

This model helps users understand their latency budget and identify bottlenecks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import math

from rotastellar import RotaStellarClient, Position

# TODO(subhadipmitra): Add jitter modeling for ground station handoffs
# TODO: Integrate with real-time constellation state for accurate ISL routing
# NOTE: Propagation delay assumes speed of light in vacuum (ISL) or 0.67c (fiber)


class LinkType(Enum):
    """Types of communication links."""

    GROUND_TO_SATELLITE = "ground_to_satellite"
    SATELLITE_TO_GROUND = "satellite_to_ground"
    INTER_SATELLITE = "inter_satellite"
    OPTICAL = "optical"
    RF = "rf"


@dataclass
class LatencyComponent:
    """A component of end-to-end latency.

    Attributes:
        name: Component name
        latency_ms: Latency contribution in milliseconds
        description: Description of the component
    """

    name: str
    latency_ms: float
    description: str = ""


@dataclass
class LatencyResult:
    """Result of latency simulation.

    Attributes:
        total_latency_ms: Total end-to-end latency
        propagation_latency_ms: Speed-of-light propagation delay
        processing_latency_ms: Processing/compute latency
        queuing_latency_ms: Queuing and scheduling delays
        components: Breakdown of latency components
        path: Communication path description
        achievable: Whether latency requirement is achievable
    """

    total_latency_ms: float
    propagation_latency_ms: float
    processing_latency_ms: float
    queuing_latency_ms: float
    components: List[LatencyComponent] = field(default_factory=list)
    path: str = ""
    achievable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_latency_ms": self.total_latency_ms,
            "propagation_latency_ms": self.propagation_latency_ms,
            "processing_latency_ms": self.processing_latency_ms,
            "queuing_latency_ms": self.queuing_latency_ms,
            "components": [
                {"name": c.name, "latency_ms": c.latency_ms, "description": c.description}
                for c in self.components
            ],
            "path": self.path,
            "achievable": self.achievable,
        }


# Physical constants
SPEED_OF_LIGHT_KM_S = 299792.458
EARTH_RADIUS_KM = 6371.0


class LatencySimulator:
    """Simulate network latency through orbital infrastructure.

    Example:
        >>> from rotastellar_compute import LatencySimulator
        >>> from rotastellar import Position
        >>> simulator = LatencySimulator(api_key="rs_live_xxx")
        >>>
        >>> source = Position(37.7749, -122.4194, 0)  # San Francisco
        >>> dest = Position(51.5074, -0.1278, 0)  # London
        >>> result = simulator.simulate(source, dest)
        >>> print(f"Total latency: {result.total_latency_ms:.1f} ms")
    """

    # Default latency assumptions (ms)
    DEFAULT_PROCESSING_LATENCY = 2.0
    DEFAULT_GROUND_STATION_LATENCY = 1.0
    DEFAULT_QUEUING_LATENCY = 0.5

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        orbit_altitude_km: float = 550.0,
        **kwargs,
    ):
        """Initialize the latency simulator.

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

    def simulate(
        self,
        source: Position,
        destination: Position,
        orbit_altitude_km: Optional[float] = None,
        relay_count: int = 0,
        include_compute: bool = False,
        compute_time_ms: float = 0.0,
    ) -> LatencyResult:
        """Simulate end-to-end latency.

        Args:
            source: Source position on Earth
            destination: Destination position on Earth
            orbit_altitude_km: Satellite orbit altitude
            relay_count: Number of inter-satellite relays
            include_compute: Whether to include compute time
            compute_time_ms: Additional compute processing time

        Returns:
            Latency simulation result
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km
        components = []

        # Ground station to satellite (uplink)
        uplink_distance = self._slant_range(source, altitude)
        uplink_latency = self._propagation_delay(uplink_distance)
        components.append(LatencyComponent(
            name="uplink",
            latency_ms=uplink_latency,
            description=f"Ground to satellite ({uplink_distance:.0f} km)"
        ))

        # Ground station processing
        components.append(LatencyComponent(
            name="ground_station_tx",
            latency_ms=self.DEFAULT_GROUND_STATION_LATENCY,
            description="Ground station transmit processing"
        ))

        # Inter-satellite links (if any)
        isl_latency = 0.0
        if relay_count > 0:
            # Calculate ISL distance based on relay count
            ground_distance = self._ground_distance(source, destination)
            isl_distance = self._estimate_isl_distance(ground_distance, relay_count, altitude)
            isl_latency = self._propagation_delay(isl_distance)
            components.append(LatencyComponent(
                name="inter_satellite",
                latency_ms=isl_latency,
                description=f"{relay_count} ISL hop(s), {isl_distance:.0f} km total"
            ))

            # ISL processing at each relay
            relay_processing = relay_count * 0.5  # 0.5ms per hop
            components.append(LatencyComponent(
                name="relay_processing",
                latency_ms=relay_processing,
                description=f"Processing at {relay_count} relay satellite(s)"
            ))

        # Satellite processing
        components.append(LatencyComponent(
            name="satellite_processing",
            latency_ms=self.DEFAULT_PROCESSING_LATENCY,
            description="Satellite onboard processing"
        ))

        # Compute time (if requested)
        if include_compute and compute_time_ms > 0:
            components.append(LatencyComponent(
                name="compute",
                latency_ms=compute_time_ms,
                description="Orbital compute processing"
            ))

        # Satellite to ground (downlink)
        downlink_distance = self._slant_range(destination, altitude)
        downlink_latency = self._propagation_delay(downlink_distance)
        components.append(LatencyComponent(
            name="downlink",
            latency_ms=downlink_latency,
            description=f"Satellite to ground ({downlink_distance:.0f} km)"
        ))

        # Ground station receive processing
        components.append(LatencyComponent(
            name="ground_station_rx",
            latency_ms=self.DEFAULT_GROUND_STATION_LATENCY,
            description="Ground station receive processing"
        ))

        # Queuing delay
        components.append(LatencyComponent(
            name="queuing",
            latency_ms=self.DEFAULT_QUEUING_LATENCY,
            description="Network queuing delay"
        ))

        # Calculate totals
        total_latency = sum(c.latency_ms for c in components)
        propagation = uplink_latency + isl_latency + downlink_latency
        processing = sum(
            c.latency_ms for c in components
            if c.name in ["satellite_processing", "ground_station_tx", "ground_station_rx", "relay_processing"]
        )
        queuing = self.DEFAULT_QUEUING_LATENCY

        # Build path description
        if relay_count > 0:
            path = f"Source -> Ground Station -> Satellite -> {relay_count}x ISL -> Satellite -> Ground Station -> Destination"
        else:
            path = "Source -> Ground Station -> Satellite -> Ground Station -> Destination"

        return LatencyResult(
            total_latency_ms=round(total_latency, 2),
            propagation_latency_ms=round(propagation, 2),
            processing_latency_ms=round(processing, 2),
            queuing_latency_ms=round(queuing, 2),
            components=components,
            path=path,
            achievable=True,
        )

    def compare_terrestrial(
        self,
        source: Position,
        destination: Position,
        orbit_altitude_km: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compare orbital vs terrestrial latency.

        Args:
            source: Source position
            destination: Destination position
            orbit_altitude_km: Satellite altitude

        Returns:
            Comparison results
        """
        altitude = orbit_altitude_km or self.orbit_altitude_km

        # Orbital latency (direct)
        orbital_direct = self.simulate(source, destination, altitude, relay_count=0)

        # Orbital with 1 relay
        orbital_relay = self.simulate(source, destination, altitude, relay_count=1)

        # Estimate terrestrial (fiber) latency
        ground_distance = self._ground_distance(source, destination)

        # Fiber is ~0.67c effective speed, plus routing overhead
        fiber_latency = (ground_distance / (SPEED_OF_LIGHT_KM_S * 0.67)) * 1000
        # Add typical internet routing delays
        fiber_latency += 10 + (ground_distance / 1000) * 2  # Base + per-1000km

        return {
            "ground_distance_km": round(ground_distance, 1),
            "orbital_direct_ms": orbital_direct.total_latency_ms,
            "orbital_relay_ms": orbital_relay.total_latency_ms,
            "terrestrial_fiber_ms": round(fiber_latency, 2),
            "orbital_advantage_ms": round(fiber_latency - orbital_direct.total_latency_ms, 2),
            "orbital_is_faster": orbital_direct.total_latency_ms < fiber_latency,
        }

    def minimum_latency(
        self,
        altitude_km: float,
        elevation_angle_deg: float = 25.0,
    ) -> float:
        """Calculate minimum achievable latency for given orbit.

        Args:
            altitude_km: Orbit altitude
            elevation_angle_deg: Minimum elevation angle

        Returns:
            Minimum round-trip latency in milliseconds
        """
        # Slant range at minimum elevation
        slant = self._slant_range_from_elevation(altitude_km, elevation_angle_deg)

        # Round trip propagation
        propagation = 2 * self._propagation_delay(slant)

        # Minimum processing
        min_processing = 2 * self.DEFAULT_GROUND_STATION_LATENCY + self.DEFAULT_PROCESSING_LATENCY

        return round(propagation + min_processing, 2)

    def _slant_range(self, ground_pos: Position, altitude_km: float) -> float:
        """Calculate slant range from ground to satellite overhead.

        For simplicity, assumes satellite directly overhead.
        Real calculation would consider satellite position.
        """
        # Simplified: assume moderate elevation angle (~45°)
        return altitude_km / math.sin(math.radians(45))

    def _slant_range_from_elevation(self, altitude_km: float, elevation_deg: float) -> float:
        """Calculate slant range from elevation angle."""
        elevation_rad = math.radians(elevation_deg)
        r_earth = EARTH_RADIUS_KM
        r_sat = r_earth + altitude_km

        # Law of sines
        sin_gamma = r_earth * math.cos(elevation_rad) / r_sat
        gamma = math.asin(sin_gamma)
        alpha = math.pi / 2 - elevation_rad - gamma

        slant = r_earth * math.sin(alpha) / math.sin(gamma)
        return slant

    def _propagation_delay(self, distance_km: float) -> float:
        """Calculate propagation delay in milliseconds."""
        return (distance_km / SPEED_OF_LIGHT_KM_S) * 1000

    def _ground_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate great circle distance between two positions."""
        lat1 = math.radians(pos1.latitude)
        lat2 = math.radians(pos2.latitude)
        dlon = math.radians(pos2.longitude - pos1.longitude)

        # Haversine formula
        a = (
            math.sin((lat2 - lat1) / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_KM * c

    def _estimate_isl_distance(
        self, ground_distance: float, relay_count: int, altitude_km: float
    ) -> float:
        """Estimate total inter-satellite link distance."""
        # Simplified: ISL distance is roughly ground distance at orbit altitude
        # Actual calculation would depend on constellation geometry
        r_sat = EARTH_RADIUS_KM + altitude_km

        # Arc length at satellite altitude
        arc_angle = ground_distance / EARTH_RADIUS_KM
        isl_arc = r_sat * arc_angle

        # Add some overhead for non-ideal routing
        return isl_arc * 1.1
