"""
RotaStellar SDK - Common Types

Core data types used throughout the SDK.

subhadipmitra@: These types are shared across all RotaStellar packages. They use
dataclasses for simplicity but could be migrated to Pydantic for richer validation.

Design decisions:
- Use degrees (not radians) for human-readable I/O
- Use km as the standard distance unit (matches aerospace convention)
- Validate on construction to fail fast on bad data
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, TypeVar, Generic, Callable, AsyncIterator, Iterator
import math

from .errors import ValidationError


T = TypeVar("T")


# Earth constants
# NOTE(subhadipmitra): Using WGS84 equatorial radius. Polar radius is 6356.752 km.
EARTH_RADIUS_KM = 6378.137
# Standard gravitational parameter (GM) for Earth
EARTH_MU = 398600.4418  # km^3/s^2


@dataclass
class Position:
    """Geographic position with altitude.

    Attributes:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        altitude_km: Altitude above sea level in kilometers

    Example:
        >>> pos = Position(latitude=28.5729, longitude=-80.6490, altitude_km=408.0)
        >>> print(f"ISS at {pos.latitude}, {pos.longitude}")
    """

    latitude: float
    longitude: float
    altitude_km: float = 0.0

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate position parameters."""
        if not -90 <= self.latitude <= 90:
            raise ValidationError("latitude", "Must be between -90 and 90 degrees")
        if not -180 <= self.longitude <= 180:
            raise ValidationError("longitude", "Must be between -180 and 180 degrees")
        if self.altitude_km < 0:
            raise ValidationError("altitude_km", "Must be non-negative")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_km": self.altitude_km,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from dictionary."""
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            altitude_km=data.get("altitude_km", 0.0),
        )


@dataclass
class Orbit:
    """Keplerian orbital elements.

    Attributes:
        semi_major_axis_km: Semi-major axis in kilometers
        eccentricity: Orbital eccentricity (0 = circular, 0-1 = elliptical)
        inclination_deg: Inclination in degrees (0-180)
        raan_deg: Right ascension of ascending node in degrees (0-360)
        arg_periapsis_deg: Argument of periapsis in degrees (0-360)
        true_anomaly_deg: True anomaly in degrees (0-360)

    Example:
        >>> orbit = Orbit(
        ...     semi_major_axis_km=6778.0,
        ...     eccentricity=0.0001,
        ...     inclination_deg=51.6,
        ...     raan_deg=100.0,
        ...     arg_periapsis_deg=90.0,
        ...     true_anomaly_deg=0.0
        ... )
        >>> print(f"Period: {orbit.orbital_period_minutes:.1f} minutes")
    """

    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_periapsis_deg: float
    true_anomaly_deg: float

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate orbital parameters."""
        if self.semi_major_axis_km <= EARTH_RADIUS_KM:
            raise ValidationError(
                "semi_major_axis_km",
                f"Must be greater than Earth radius ({EARTH_RADIUS_KM} km)",
            )
        if not 0 <= self.eccentricity < 1:
            raise ValidationError(
                "eccentricity", "Must be between 0 (inclusive) and 1 (exclusive)"
            )
        if not 0 <= self.inclination_deg <= 180:
            raise ValidationError("inclination_deg", "Must be between 0 and 180 degrees")

    @property
    def apogee_km(self) -> float:
        """Apogee altitude above Earth surface in kilometers."""
        return self.semi_major_axis_km * (1 + self.eccentricity) - EARTH_RADIUS_KM

    @property
    def perigee_km(self) -> float:
        """Perigee altitude above Earth surface in kilometers."""
        return self.semi_major_axis_km * (1 - self.eccentricity) - EARTH_RADIUS_KM

    @property
    def orbital_period_seconds(self) -> float:
        """Orbital period in seconds."""
        return 2 * math.pi * math.sqrt(self.semi_major_axis_km**3 / EARTH_MU)

    @property
    def orbital_period_minutes(self) -> float:
        """Orbital period in minutes."""
        return self.orbital_period_seconds / 60

    @property
    def mean_motion(self) -> float:
        """Mean motion in revolutions per day."""
        return 86400 / self.orbital_period_seconds

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "semi_major_axis_km": self.semi_major_axis_km,
            "eccentricity": self.eccentricity,
            "inclination_deg": self.inclination_deg,
            "raan_deg": self.raan_deg,
            "arg_periapsis_deg": self.arg_periapsis_deg,
            "true_anomaly_deg": self.true_anomaly_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Orbit":
        """Create Orbit from dictionary."""
        return cls(
            semi_major_axis_km=data["semi_major_axis_km"],
            eccentricity=data["eccentricity"],
            inclination_deg=data["inclination_deg"],
            raan_deg=data["raan_deg"],
            arg_periapsis_deg=data["arg_periapsis_deg"],
            true_anomaly_deg=data["true_anomaly_deg"],
        )


@dataclass
class TimeRange:
    """Time range for queries.

    Attributes:
        start: Start time (timezone-aware datetime)
        end: End time (timezone-aware datetime)

    Example:
        >>> from datetime import datetime, timezone, timedelta
        >>> now = datetime.now(timezone.utc)
        >>> tr = TimeRange(start=now, end=now + timedelta(hours=24))
        >>> print(f"Duration: {tr.duration_hours} hours")
    """

    start: datetime
    end: datetime

    def __post_init__(self):
        # Ensure timezone-aware
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=timezone.utc)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=timezone.utc)
        self.validate()

    def validate(self) -> None:
        """Validate time range."""
        if self.end <= self.start:
            raise ValidationError("end", "End time must be after start time")

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return (self.end - self.start).total_seconds()

    @property
    def duration_hours(self) -> float:
        """Duration in hours."""
        return self.duration_seconds / 3600

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with ISO format strings."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TimeRange":
        """Create TimeRange from dictionary."""
        return cls(
            start=datetime.fromisoformat(data["start"]),
            end=datetime.fromisoformat(data["end"]),
        )

    @classmethod
    def next_hours(cls, hours: float) -> "TimeRange":
        """Create a time range starting now for the specified hours."""
        now = datetime.now(timezone.utc)
        from datetime import timedelta

        return cls(start=now, end=now + timedelta(hours=hours))


@dataclass
class Satellite:
    """Satellite information.

    Attributes:
        id: RotaStellar satellite ID
        norad_id: NORAD catalog number
        name: Satellite name
        operator: Satellite operator/owner
        constellation: Constellation name (if part of one)
        orbit: Current orbital elements
        position: Current position (if available)
    """

    id: str
    norad_id: int
    name: str
    operator: Optional[str] = None
    constellation: Optional[str] = None
    orbit: Optional[Orbit] = None
    position: Optional[Position] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "norad_id": self.norad_id,
            "name": self.name,
        }
        if self.operator:
            result["operator"] = self.operator
        if self.constellation:
            result["constellation"] = self.constellation
        if self.orbit:
            result["orbit"] = self.orbit.to_dict()
        if self.position:
            result["position"] = self.position.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Satellite":
        """Create Satellite from dictionary."""
        return cls(
            id=data["id"],
            norad_id=data["norad_id"],
            name=data["name"],
            operator=data.get("operator"),
            constellation=data.get("constellation"),
            orbit=Orbit.from_dict(data["orbit"]) if data.get("orbit") else None,
            position=Position.from_dict(data["position"])
            if data.get("position")
            else None,
        )


class PaginatedResponse(Generic[T]):
    """Paginated response for list endpoints.

    Supports iteration, async iteration, and manual pagination.

    Attributes:
        items: Current page of items
        has_more: Whether more pages are available
        total: Total number of items (if known)
        limit: Page size
        offset: Current offset

    Example:
        >>> # Manual pagination
        >>> page = client.list_satellites(constellation="starlink", limit=100)
        >>> for sat in page.items:
        ...     print(sat.name)
        >>> if page.has_more:
        ...     next_page = page.next_page()

        >>> # Iteration (auto-pagination)
        >>> for sat in client.list_satellites(constellation="starlink"):
        ...     print(sat.name)

        >>> # Async iteration
        >>> async for sat in async_client.list_satellites(constellation="starlink"):
        ...     print(sat.name)
    """

    def __init__(
        self,
        items: List[T],
        has_more: bool,
        total: Optional[int],
        limit: int,
        offset: int,
        fetch_next: Optional[Callable[[], "PaginatedResponse[T]"]] = None,
        fetch_next_async: Optional[Callable[[], "PaginatedResponse[T]"]] = None,
    ):
        self.items = items
        self.has_more = has_more
        self.total = total
        self.limit = limit
        self.offset = offset
        self._fetch_next = fetch_next
        self._fetch_next_async = fetch_next_async

    def __len__(self) -> int:
        """Return number of items in current page."""
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        """Iterate through all pages (sync).

        Automatically fetches next pages when current page is exhausted.
        """
        page: Optional[PaginatedResponse[T]] = self
        while page is not None:
            for item in page.items:
                yield item
            if page.has_more and page._fetch_next:
                page = page._fetch_next()
            else:
                page = None

    async def __aiter__(self) -> AsyncIterator[T]:
        """Iterate through all pages (async).

        Automatically fetches next pages when current page is exhausted.
        """
        page: Optional[PaginatedResponse[T]] = self
        while page is not None:
            for item in page.items:
                yield item
            if page.has_more and page._fetch_next_async:
                page = await page._fetch_next_async()
            else:
                page = None

    def next_page(self) -> "PaginatedResponse[T]":
        """Fetch the next page of results.

        Returns:
            Next page of results

        Raises:
            StopIteration: If no more pages are available
        """
        if not self.has_more:
            raise StopIteration("No more pages available")
        if not self._fetch_next:
            raise RuntimeError("No fetch function available for pagination")
        return self._fetch_next()

    async def next_page_async(self) -> "PaginatedResponse[T]":
        """Fetch the next page of results (async).

        Returns:
            Next page of results

        Raises:
            StopIteration: If no more pages are available
        """
        if not self.has_more:
            raise StopIteration("No more pages available")
        if not self._fetch_next_async:
            raise RuntimeError("No async fetch function available for pagination")
        return await self._fetch_next_async()

    def __repr__(self) -> str:
        return (
            f"PaginatedResponse(items={len(self.items)}, "
            f"has_more={self.has_more}, total={self.total})"
        )
