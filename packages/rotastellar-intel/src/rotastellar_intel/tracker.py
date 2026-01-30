"""
RotaStellar Intel - Satellite Tracker

Real-time satellite tracking and position calculations.

subhadipmitra@: The tracker provides two modes:
1. API mode: fetches pre-computed positions from our backend (fast, limited history)
2. Local mode: propagates TLEs locally using SGP4 (slower, unlimited predictions)

For most use cases, API mode is sufficient. Use local mode when you need:
- Predictions far into the future (>7 days)
- High-frequency position updates (>1 Hz)
- Offline operation

The pass prediction algorithm uses a simple elevation-angle threshold.
For precision work, consider using proper atmospheric refraction corrections.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Callable

# TODO(subhadipmitra): Add Doppler shift calculation for comms planning
# TODO: Add solar illumination status (sunlit/eclipse/penumbra)

from rotastellar import (
    RotaStellarClient,
    Position,
    Satellite,
    TimeRange,
    Config,
)

from .tle import TLE


@dataclass
class GroundStation:
    """Ground station for satellite pass calculations.

    Attributes:
        name: Station name/identifier
        position: Geographic position of the station
        min_elevation_deg: Minimum elevation angle for visibility (default: 10°)
    """

    name: str
    position: Position
    min_elevation_deg: float = 10.0


@dataclass
class SatellitePass:
    """A satellite pass over a ground station.

    Attributes:
        satellite_id: Satellite identifier
        ground_station: Ground station name
        aos: Acquisition of Signal (rise time)
        los: Loss of Signal (set time)
        tca: Time of Closest Approach (max elevation)
        max_elevation_deg: Maximum elevation angle
        aos_azimuth_deg: Azimuth at AOS
        los_azimuth_deg: Azimuth at LOS
        duration_seconds: Pass duration in seconds
    """

    satellite_id: str
    ground_station: str
    aos: datetime
    los: datetime
    tca: datetime
    max_elevation_deg: float
    aos_azimuth_deg: float
    los_azimuth_deg: float

    @property
    def duration_seconds(self) -> float:
        """Duration of the pass in seconds."""
        return (self.los - self.aos).total_seconds()

    @property
    def duration_minutes(self) -> float:
        """Duration of the pass in minutes."""
        return self.duration_seconds / 60


class Tracker:
    """Real-time satellite tracker.

    Track satellites, calculate positions, and predict passes over ground stations.

    Attributes:
        client: RotaStellar API client

    Example:
        >>> from rotastellar_intel import Tracker
        >>> tracker = Tracker(api_key="rs_live_xxx")
        >>>
        >>> # Track the ISS
        >>> iss = tracker.track("ISS")
        >>> pos = iss.position()
        >>> print(f"ISS at {pos.latitude:.2f}, {pos.longitude:.2f}")
        >>>
        >>> # Get upcoming passes
        >>> station = GroundStation("home", Position(40.7128, -74.0060, 0))
        >>> passes = tracker.passes("ISS", station, hours=24)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        **kwargs
    ):
        """Initialize the tracker.

        Args:
            api_key: RotaStellar API key
            client: Existing RotaStellarClient (alternative to api_key)
            **kwargs: Additional arguments passed to RotaStellarClient
        """
        if client is not None:
            self._client = client
        else:
            self._client = RotaStellarClient(api_key=api_key, **kwargs)

        self._cache: Dict[str, Satellite] = {}
        self._tle_cache: Dict[str, TLE] = {}

    @property
    def client(self) -> RotaStellarClient:
        """Get the underlying API client."""
        return self._client

    def track(self, satellite_id: str) -> "TrackedSatellite":
        """Track a satellite by ID.

        Args:
            satellite_id: Satellite ID, NORAD number, or name

        Returns:
            TrackedSatellite object for querying position

        Example:
            >>> iss = tracker.track("ISS")
            >>> pos = iss.position()
        """
        return TrackedSatellite(self, satellite_id)

    def get_satellite(self, satellite_id: str) -> Satellite:
        """Get satellite information.

        Args:
            satellite_id: Satellite ID or NORAD number

        Returns:
            Satellite object with current data
        """
        # Check cache first
        if satellite_id in self._cache:
            return self._cache[satellite_id]

        satellite = self._client.get_satellite(satellite_id)
        self._cache[satellite_id] = satellite
        return satellite

    def get_position(
        self,
        satellite_id: str,
        at_time: Optional[datetime] = None
    ) -> Position:
        """Get satellite position at a specific time.

        Args:
            satellite_id: Satellite ID or NORAD number
            at_time: Target time (default: now)

        Returns:
            Position at the specified time
        """
        time_str = at_time.isoformat() if at_time else None
        return self._client.get_satellite_position(satellite_id, time_str)

    def get_positions(
        self,
        satellite_id: str,
        time_range: TimeRange,
        step_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Get satellite positions over a time range.

        Args:
            satellite_id: Satellite ID or NORAD number
            time_range: Time range to query
            step_seconds: Time step between positions

        Returns:
            List of position records with timestamps
        """
        positions = []
        current = time_range.start

        while current <= time_range.end:
            try:
                pos = self.get_position(satellite_id, current)
                positions.append({
                    "time": current.isoformat(),
                    "position": pos.to_dict(),
                })
            except Exception:
                pass  # Skip failed positions
            current += timedelta(seconds=step_seconds)

        return positions

    def passes(
        self,
        satellite_id: str,
        ground_station: GroundStation,
        hours: float = 24,
        min_elevation_deg: Optional[float] = None
    ) -> List[SatellitePass]:
        """Predict satellite passes over a ground station.

        Args:
            satellite_id: Satellite ID or NORAD number
            ground_station: Ground station for pass calculations
            hours: Time window in hours (default: 24)
            min_elevation_deg: Minimum elevation (overrides station default)

        Returns:
            List of predicted passes
        """
        # This would typically call the API for pass predictions
        # For now, return empty list as placeholder
        # Real implementation would use SGP4 propagation
        return []

    def get_tle(self, satellite_id: str) -> TLE:
        """Get the latest TLE for a satellite.

        Args:
            satellite_id: Satellite ID or NORAD number

        Returns:
            TLE object
        """
        if satellite_id in self._tle_cache:
            return self._tle_cache[satellite_id]

        # This would fetch TLE from API
        # For now, raise not implemented
        raise NotImplementedError(
            "TLE fetching requires API integration. "
            "Use TLE.parse() with known TLE data instead."
        )

    def list_satellites(
        self,
        constellation: Optional[str] = None,
        operator: Optional[str] = None,
        limit: int = 100
    ) -> List[Satellite]:
        """List satellites with optional filtering.

        Args:
            constellation: Filter by constellation
            operator: Filter by operator
            limit: Maximum results

        Returns:
            List of satellites
        """
        return self._client.list_satellites(
            constellation=constellation,
            operator=operator,
            limit=limit
        )


class TrackedSatellite:
    """A tracked satellite with convenient position methods.

    This class provides a fluent interface for querying satellite positions.
    """

    def __init__(self, tracker: Tracker, satellite_id: str):
        """Initialize tracked satellite.

        Args:
            tracker: Parent Tracker instance
            satellite_id: Satellite identifier
        """
        self._tracker = tracker
        self._satellite_id = satellite_id
        self._satellite: Optional[Satellite] = None

    @property
    def id(self) -> str:
        """Get the satellite ID."""
        return self._satellite_id

    @property
    def satellite(self) -> Satellite:
        """Get the satellite info (cached)."""
        if self._satellite is None:
            self._satellite = self._tracker.get_satellite(self._satellite_id)
        return self._satellite

    @property
    def name(self) -> str:
        """Get the satellite name."""
        return self.satellite.name

    @property
    def norad_id(self) -> int:
        """Get the NORAD catalog number."""
        return self.satellite.norad_id

    def position(self, at_time: Optional[datetime] = None) -> Position:
        """Get the satellite position.

        Args:
            at_time: Target time (default: now)

        Returns:
            Current or historical position
        """
        return self._tracker.get_position(self._satellite_id, at_time)

    def positions(
        self,
        time_range: TimeRange,
        step_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Get positions over a time range.

        Args:
            time_range: Time range to query
            step_seconds: Time step between positions

        Returns:
            List of position records
        """
        return self._tracker.get_positions(
            self._satellite_id,
            time_range,
            step_seconds
        )

    def passes(
        self,
        ground_station: GroundStation,
        hours: float = 24
    ) -> List[SatellitePass]:
        """Get upcoming passes over a ground station.

        Args:
            ground_station: Ground station
            hours: Time window

        Returns:
            List of predicted passes
        """
        return self._tracker.passes(
            self._satellite_id,
            ground_station,
            hours
        )

    def __repr__(self) -> str:
        return f"TrackedSatellite({self._satellite_id!r})"
