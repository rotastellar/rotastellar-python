"""
RotaStellar Intel - Conjunction Analysis

Space object collision probability and close approach analysis.

subhadipmitra@: Conjunction assessment is one of the most critical SSA functions.
We use the Pc (probability of collision) framework from NASA's CARA team.

Risk thresholds are based on industry standards:
- 1e-4: Red threshold - maneuver decision required
- 1e-5: Yellow threshold - enhanced monitoring
- 1e-7: Green threshold - routine tracking

The miss distance alone is NOT sufficient for risk assessment. A 1km miss with
high covariance uncertainty might be riskier than a 100m miss with low uncertainty.

References:
- NASA CARA Pc Computation: https://www.nasa.gov/cara
- ESA Collision Avoidance: https://www.esa.int/space_debris
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any

# TODO(subhadipmitra): Add Monte Carlo collision probability estimation
# TODO: Integrate with Space-Track CDM (Conjunction Data Messages)
# FIXME: Current Pc calculation assumes spherical covariance (simplification)

from rotastellar import (
    RotaStellarClient,
    Position,
    TimeRange,
)


class RiskLevel(Enum):
    """Conjunction risk level classification."""

    CRITICAL = "critical"  # Immediate action required (P > 1e-4)
    HIGH = "high"          # Close monitoring needed (P > 1e-5)
    MEDIUM = "medium"      # Standard monitoring (P > 1e-6)
    LOW = "low"            # Routine tracking (P > 1e-7)
    NEGLIGIBLE = "negligible"  # No action needed (P <= 1e-7)


@dataclass
class Conjunction:
    """A conjunction (close approach) between two space objects.

    Attributes:
        id: Unique conjunction ID
        primary_id: Primary satellite ID
        primary_name: Primary satellite name
        secondary_id: Secondary object ID (satellite or debris)
        secondary_name: Secondary object name
        tca: Time of Closest Approach
        miss_distance_km: Predicted miss distance in km
        miss_distance_radial_km: Radial component of miss distance
        miss_distance_in_track_km: In-track component of miss distance
        miss_distance_cross_track_km: Cross-track component of miss distance
        relative_velocity_km_s: Relative velocity at TCA in km/s
        collision_probability: Probability of collision
        risk_level: Risk classification
        created_at: When this conjunction was identified
        updated_at: Last update time
    """

    id: str
    primary_id: str
    primary_name: str
    secondary_id: str
    secondary_name: str
    tca: datetime
    miss_distance_km: float
    miss_distance_radial_km: Optional[float] = None
    miss_distance_in_track_km: Optional[float] = None
    miss_distance_cross_track_km: Optional[float] = None
    relative_velocity_km_s: Optional[float] = None
    collision_probability: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.LOW
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conjunction":
        """Create Conjunction from API response dictionary."""
        risk_str = data.get("risk_level", "low")
        try:
            risk_level = RiskLevel(risk_str.lower())
        except ValueError:
            risk_level = RiskLevel.LOW

        return cls(
            id=data["id"],
            primary_id=data["primary_id"],
            primary_name=data.get("primary_name", "Unknown"),
            secondary_id=data["secondary_id"],
            secondary_name=data.get("secondary_name", "Unknown"),
            tca=datetime.fromisoformat(data["tca"].replace("Z", "+00:00")),
            miss_distance_km=data["miss_distance_km"],
            miss_distance_radial_km=data.get("miss_distance_radial_km"),
            miss_distance_in_track_km=data.get("miss_distance_in_track_km"),
            miss_distance_cross_track_km=data.get("miss_distance_cross_track_km"),
            relative_velocity_km_s=data.get("relative_velocity_km_s"),
            collision_probability=data.get("collision_probability"),
            risk_level=risk_level,
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            if data.get("updated_at")
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "primary_id": self.primary_id,
            "primary_name": self.primary_name,
            "secondary_id": self.secondary_id,
            "secondary_name": self.secondary_name,
            "tca": self.tca.isoformat(),
            "miss_distance_km": self.miss_distance_km,
            "risk_level": self.risk_level.value,
        }
        if self.miss_distance_radial_km is not None:
            result["miss_distance_radial_km"] = self.miss_distance_radial_km
        if self.miss_distance_in_track_km is not None:
            result["miss_distance_in_track_km"] = self.miss_distance_in_track_km
        if self.miss_distance_cross_track_km is not None:
            result["miss_distance_cross_track_km"] = self.miss_distance_cross_track_km
        if self.relative_velocity_km_s is not None:
            result["relative_velocity_km_s"] = self.relative_velocity_km_s
        if self.collision_probability is not None:
            result["collision_probability"] = self.collision_probability
        return result

    @property
    def is_critical(self) -> bool:
        """Check if this conjunction is critical risk."""
        return self.risk_level == RiskLevel.CRITICAL

    @property
    def is_high_risk(self) -> bool:
        """Check if this conjunction is high risk or above."""
        return self.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)

    @property
    def time_to_tca(self) -> float:
        """Get time to TCA in hours (negative if past)."""
        now = datetime.now(timezone.utc)
        return (self.tca - now).total_seconds() / 3600


@dataclass
class ManeuverRecommendation:
    """Recommended maneuver to avoid a conjunction.

    Attributes:
        conjunction_id: ID of the conjunction to avoid
        maneuver_time: Recommended maneuver execution time
        delta_v_m_s: Required delta-v in m/s
        direction: Maneuver direction (radial, in-track, cross-track)
        post_maneuver_miss_km: Expected miss distance after maneuver
        post_maneuver_probability: Expected collision probability after maneuver
        fuel_required_kg: Estimated fuel required (if available)
        confidence: Confidence level of the recommendation
    """

    conjunction_id: str
    maneuver_time: datetime
    delta_v_m_s: float
    direction: str
    post_maneuver_miss_km: float
    post_maneuver_probability: float
    fuel_required_kg: Optional[float] = None
    confidence: float = 0.95


class ConjunctionAnalyzer:
    """Analyze conjunctions and collision risks.

    Example:
        >>> from rotastellar_intel import ConjunctionAnalyzer
        >>> analyzer = ConjunctionAnalyzer(api_key="rs_live_xxx")
        >>>
        >>> # Get conjunctions for a satellite
        >>> conjunctions = analyzer.get_conjunctions("starlink-1234")
        >>> for c in conjunctions:
        ...     print(f"{c.miss_distance_km:.2f} km at {c.tca}")
        >>>
        >>> # Get high-risk conjunctions
        >>> critical = analyzer.get_high_risk_conjunctions()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        **kwargs
    ):
        """Initialize the analyzer.

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

    def get_conjunctions(
        self,
        satellite_id: Optional[str] = None,
        *,
        threshold_km: float = 1.0,
        time_range: Optional[TimeRange] = None,
        limit: int = 100
    ) -> List[Conjunction]:
        """Get conjunctions with optional filtering.

        Args:
            satellite_id: Filter by satellite
            threshold_km: Maximum miss distance threshold
            time_range: Time window to search
            limit: Maximum results

        Returns:
            List of Conjunction objects
        """
        raw_conjunctions = self._client.list_conjunctions(
            satellite_id=satellite_id,
            threshold_km=threshold_km,
            time_range=time_range,
            limit=limit
        )

        return [
            Conjunction.from_dict(c) if isinstance(c, dict) else c
            for c in raw_conjunctions
        ]

    def get_high_risk_conjunctions(
        self,
        satellite_id: Optional[str] = None,
        hours: float = 72
    ) -> List[Conjunction]:
        """Get high-risk conjunctions requiring attention.

        Args:
            satellite_id: Filter by satellite
            hours: Time window in hours

        Returns:
            List of high-risk conjunctions sorted by risk
        """
        time_range = TimeRange.next_hours(hours)
        conjunctions = self.get_conjunctions(
            satellite_id=satellite_id,
            threshold_km=5.0,  # Wider threshold for risk assessment
            time_range=time_range,
            limit=500
        )

        # Filter to high risk and sort
        high_risk = [c for c in conjunctions if c.is_high_risk]
        return sorted(
            high_risk,
            key=lambda c: (
                c.risk_level != RiskLevel.CRITICAL,
                c.miss_distance_km
            )
        )

    def get_conjunction(self, conjunction_id: str) -> Conjunction:
        """Get a specific conjunction by ID.

        Args:
            conjunction_id: Conjunction ID

        Returns:
            Conjunction details
        """
        # This would call specific conjunction endpoint
        raise NotImplementedError(
            "Individual conjunction lookup requires API integration"
        )

    def recommend_maneuver(
        self,
        conjunction_id: str,
        target_miss_km: float = 5.0
    ) -> ManeuverRecommendation:
        """Get maneuver recommendation for a conjunction.

        Args:
            conjunction_id: Conjunction to avoid
            target_miss_km: Desired post-maneuver miss distance

        Returns:
            Maneuver recommendation
        """
        # This would call the API for maneuver planning
        raise NotImplementedError(
            "Maneuver recommendations require API integration"
        )

    def analyze_risk(
        self,
        satellite_id: str,
        hours: float = 168  # 7 days
    ) -> Dict[str, Any]:
        """Analyze overall conjunction risk for a satellite.

        Args:
            satellite_id: Satellite to analyze
            hours: Analysis window in hours

        Returns:
            Risk analysis summary
        """
        time_range = TimeRange.next_hours(hours)
        conjunctions = self.get_conjunctions(
            satellite_id=satellite_id,
            threshold_km=10.0,
            time_range=time_range,
            limit=1000
        )

        # Categorize by risk level
        by_risk = {level: [] for level in RiskLevel}
        for c in conjunctions:
            by_risk[c.risk_level].append(c)

        # Find closest approach
        closest = min(conjunctions, key=lambda c: c.miss_distance_km) if conjunctions else None

        return {
            "satellite_id": satellite_id,
            "analysis_window_hours": hours,
            "total_conjunctions": len(conjunctions),
            "by_risk_level": {
                level.value: len(items) for level, items in by_risk.items()
            },
            "critical_count": len(by_risk[RiskLevel.CRITICAL]),
            "high_risk_count": len(by_risk[RiskLevel.HIGH]),
            "closest_approach_km": closest.miss_distance_km if closest else None,
            "closest_approach_tca": closest.tca.isoformat() if closest else None,
            "requires_attention": len(by_risk[RiskLevel.CRITICAL]) > 0 or len(by_risk[RiskLevel.HIGH]) > 0,
        }
