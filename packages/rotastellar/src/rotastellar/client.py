"""
RotaStellar SDK - Main Client

The main entry point for the RotaStellar SDK.

subhadipmitra@: This is the primary interface users interact with. Design goals:
- Simple: one import, one class, sensible defaults
- Safe: automatic retry, rate limiting, API key validation
- Flexible: override any default via constructor or env vars

The client is intentionally stateless (no caching) to avoid stale data issues.
For high-frequency access, users should implement their own caching layer.
"""

from typing import Any, Dict, List, Optional

# TODO(subhadipmitra): Add async client variant using httpx
# TODO: Add request/response logging for debugging

from .auth import validate_api_key
from .config import Config
from .http import HTTPClient
from .types import Position, Orbit, Satellite, TimeRange


class RotaStellarClient:
    """Main client for the RotaStellar API.

    This is the primary entry point for interacting with RotaStellar services.

    Attributes:
        config: SDK configuration
        http: HTTP client for making requests

    Example:
        >>> from rotastellar import RotaStellarClient
        >>> client = RotaStellarClient(api_key="rs_live_xxx")
        >>>
        >>> # List satellites
        >>> satellites = client.list_satellites(constellation="starlink")
        >>> for sat in satellites:
        ...     print(f"{sat.name}: {sat.norad_id}")
        >>>
        >>> # Get specific satellite
        >>> iss = client.get_satellite("iss")
        >>> print(f"ISS at {iss.position.latitude}, {iss.position.longitude}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        debug: bool = False,
    ):
        """Initialize the RotaStellar client.

        Args:
            api_key: RotaStellar API key. If not provided, reads from
                     ROTASTELLAR_API_KEY environment variable.
            base_url: API base URL (default: https://api.rotastellar.com/v1)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts (default: 3)
            debug: Enable debug logging

        Raises:
            MissingAPIKeyError: If no API key is provided or found
            InvalidAPIKeyError: If API key format is invalid
        """
        # Create config with provided values
        self.config = Config(api_key=api_key, debug=debug)

        # Override specific settings if provided
        if base_url is not None:
            self.config.base_url = base_url
        if timeout is not None:
            self.config.timeout = timeout
        if max_retries is not None:
            self.config.max_retries = max_retries

        # Validate API key
        validate_api_key(self.config.api_key)

        # Initialize HTTP client
        self.http = HTTPClient(self.config)

    @property
    def environment(self) -> Optional[str]:
        """Get the API key environment (test or live)."""
        return self.config.environment

    # =========================================================================
    # Satellite Operations
    # =========================================================================

    def list_satellites(
        self,
        *,
        constellation: Optional[str] = None,
        operator: Optional[str] = None,
        satellite_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Satellite]:
        """List satellites with optional filtering.

        Args:
            constellation: Filter by constellation name (e.g., "starlink", "oneweb")
            operator: Filter by operator (e.g., "SpaceX", "Amazon")
            satellite_type: Filter by type (e.g., "communication", "observation")
            limit: Maximum number of results (default: 100, max: 1000)
            offset: Pagination offset

        Returns:
            List of Satellite objects

        Example:
            >>> satellites = client.list_satellites(constellation="starlink", limit=10)
            >>> print(f"Found {len(satellites)} Starlink satellites")
        """
        params = {
            "constellation": constellation,
            "operator": operator,
            "type": satellite_type,
            "limit": limit,
            "offset": offset,
        }

        response = self.http.get("/satellites", params=params)
        satellites = response.get("data", [])
        return [Satellite.from_dict(sat) for sat in satellites]

    def get_satellite(self, satellite_id: str) -> Satellite:
        """Get a specific satellite by ID.

        Args:
            satellite_id: Satellite ID or NORAD number

        Returns:
            Satellite object with current position and orbital elements

        Raises:
            NotFoundError: If satellite is not found

        Example:
            >>> iss = client.get_satellite("25544")  # NORAD ID for ISS
            >>> print(f"ISS altitude: {iss.position.altitude_km:.1f} km")
        """
        response = self.http.get(f"/satellites/{satellite_id}")
        return Satellite.from_dict(response)

    def get_satellite_position(
        self, satellite_id: str, at_time: Optional[str] = None
    ) -> Position:
        """Get satellite position at a specific time.

        Args:
            satellite_id: Satellite ID or NORAD number
            at_time: ISO 8601 timestamp (default: now)

        Returns:
            Position object with lat, lon, altitude

        Example:
            >>> pos = client.get_satellite_position("25544")
            >>> print(f"ISS: {pos.latitude:.2f}, {pos.longitude:.2f}")
        """
        params = {"at": at_time} if at_time else None
        response = self.http.get(f"/satellites/{satellite_id}/position", params=params)
        return Position.from_dict(response)

    # =========================================================================
    # Conjunction Analysis
    # =========================================================================

    def list_conjunctions(
        self,
        *,
        satellite_id: Optional[str] = None,
        threshold_km: float = 1.0,
        time_range: Optional[TimeRange] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List close approaches (conjunctions) between space objects.

        Args:
            satellite_id: Filter by specific satellite
            threshold_km: Maximum miss distance in km (default: 1.0)
            time_range: Time range to search
            limit: Maximum results

        Returns:
            List of conjunction events with probability and miss distance

        Example:
            >>> conjunctions = client.list_conjunctions(
            ...     satellite_id="starlink-1234",
            ...     threshold_km=5.0,
            ...     time_range=TimeRange.next_hours(72)
            ... )
            >>> for c in conjunctions:
            ...     print(f"{c['miss_distance_km']:.2f} km at {c['tca']}")
        """
        params: Dict[str, Any] = {
            "satellite_id": satellite_id,
            "threshold_km": threshold_km,
            "limit": limit,
        }
        if time_range:
            params["start"] = time_range.start.isoformat()
            params["end"] = time_range.end.isoformat()

        response = self.http.get("/conjunctions", params=params)
        return response.get("data", [])

    # =========================================================================
    # Planning Operations
    # =========================================================================

    def analyze_feasibility(
        self,
        *,
        workload_type: str,
        compute_tflops: float,
        data_gb: float,
        latency_requirement_ms: Optional[float] = None,
        orbit_altitude_km: float = 550,
    ) -> Dict[str, Any]:
        """Analyze feasibility of orbital compute for a workload.

        Args:
            workload_type: Type of workload (e.g., "inference", "training", "batch")
            compute_tflops: Required compute in TFLOPS
            data_gb: Data volume in GB
            latency_requirement_ms: Maximum acceptable latency in ms
            orbit_altitude_km: Target orbit altitude in km

        Returns:
            Feasibility analysis with recommendations

        Example:
            >>> result = client.analyze_feasibility(
            ...     workload_type="inference",
            ...     compute_tflops=10,
            ...     data_gb=1.5,
            ...     latency_requirement_ms=100
            ... )
            >>> print(f"Feasible: {result['feasible']}")
            >>> print(f"Recommendation: {result['recommendation']}")
        """
        data = {
            "workload_type": workload_type,
            "compute_tflops": compute_tflops,
            "data_gb": data_gb,
            "latency_requirement_ms": latency_requirement_ms,
            "orbit_altitude_km": orbit_altitude_km,
        }
        return self.http.post("/planning/analyze", data=data)

    def simulate_thermal(
        self,
        *,
        power_watts: float,
        orbit_altitude_km: float = 550,
        radiator_area_m2: float = 1.0,
        duration_hours: float = 24,
    ) -> Dict[str, Any]:
        """Simulate thermal conditions for orbital compute.

        Args:
            power_watts: Power dissipation in watts
            orbit_altitude_km: Orbit altitude in km
            radiator_area_m2: Radiator area in square meters
            duration_hours: Simulation duration in hours

        Returns:
            Thermal simulation results with temperature profiles

        Example:
            >>> result = client.simulate_thermal(
            ...     power_watts=500,
            ...     radiator_area_m2=2.0,
            ...     duration_hours=24
            ... )
            >>> print(f"Max temp: {result['max_temperature_c']}C")
        """
        data = {
            "power_watts": power_watts,
            "orbit_altitude_km": orbit_altitude_km,
            "radiator_area_m2": radiator_area_m2,
            "duration_hours": duration_hours,
        }
        return self.http.post("/planning/thermal", data=data)

    def simulate_latency(
        self,
        *,
        source: Position,
        destination: Position,
        orbit_altitude_km: float = 550,
        relay_count: int = 0,
    ) -> Dict[str, Any]:
        """Simulate network latency through orbital infrastructure.

        Args:
            source: Source position on Earth
            destination: Destination position on Earth
            orbit_altitude_km: Satellite orbit altitude
            relay_count: Number of inter-satellite links (0 = direct)

        Returns:
            Latency simulation with breakdown by segment

        Example:
            >>> source = Position(latitude=37.7749, longitude=-122.4194)  # SF
            >>> dest = Position(latitude=51.5074, longitude=-0.1278)  # London
            >>> result = client.simulate_latency(source=source, destination=dest)
            >>> print(f"Total latency: {result['total_latency_ms']:.1f} ms")
        """
        data = {
            "source": source.to_dict(),
            "destination": destination.to_dict(),
            "orbit_altitude_km": orbit_altitude_km,
            "relay_count": relay_count,
        }
        return self.http.post("/planning/latency", data=data)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def ping(self) -> Dict[str, Any]:
        """Check API connectivity and authentication.

        Returns:
            API status information

        Example:
            >>> status = client.ping()
            >>> print(f"API version: {status['version']}")
        """
        return self.http.get("/ping")

    def __repr__(self) -> str:
        """String representation of client."""
        env = self.environment or "unknown"
        return f"RotaStellarClient(environment={env})"
