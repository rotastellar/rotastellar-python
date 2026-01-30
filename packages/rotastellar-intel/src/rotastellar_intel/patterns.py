"""
RotaStellar Intel - Pattern Detection

Satellite behavior analysis, anomaly detection, and pattern recognition.

subhadipmitra@: This is where space situational awareness gets interesting.
By analyzing TLE history, we can detect:
- Maneuvers (delta-v events that change orbit)
- Anomalies (unexpected behavior, possible failures)
- Operational patterns (station-keeping schedules, rendezvous operations)

Detection is based on orbit element changes between TLE updates:
- Δa > 1km → altitude maneuver
- Δi > 0.1° → plane change (expensive!)
- Δe change → orbit shaping
- Sudden decay → drag event or deorbit

The challenge is distinguishing real maneuvers from TLE fit noise.
We use filtering and multiple-observation confirmation.

Fun fact: you can often predict commercial satellite maneuvers from their
operational patterns (e.g., station-keeping every 2 weeks).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any

from rotastellar import RotaStellarClient, TimeRange

# TODO(subhadipmitra): Add ML-based anomaly detection using historical patterns
# TODO: Correlate maneuvers with space weather events
# FIXME: Current detection thresholds are tuned for LEO; need adjustment for GEO


class PatternType(Enum):
    """Types of detected patterns/anomalies."""

    MANEUVER = "maneuver"              # Orbital maneuver detected
    ORBIT_RAISE = "orbit_raise"        # Altitude increase
    ORBIT_LOWER = "orbit_lower"        # Altitude decrease
    PLANE_CHANGE = "plane_change"      # Inclination change
    DEORBIT = "deorbit"               # Deorbit maneuver
    STATION_KEEPING = "station_keeping"  # Station-keeping burn
    PROXIMITY_OPS = "proximity_ops"    # Close approach to another object
    RENDEZVOUS = "rendezvous"         # Docking/berthing approach
    DEBRIS_AVOIDANCE = "debris_avoidance"  # Collision avoidance maneuver
    ANOMALY = "anomaly"               # Unexpected behavior
    TUMBLING = "tumbling"             # Loss of attitude control
    FRAGMENTATION = "fragmentation"    # Breakup event
    DEPLOYMENT = "deployment"         # Satellite deployment
    REENTRY = "reentry"               # Atmospheric reentry


class ConfidenceLevel(Enum):
    """Confidence level of pattern detection."""

    CONFIRMED = "confirmed"    # High confidence, multiple data sources
    LIKELY = "likely"          # Good confidence
    POSSIBLE = "possible"      # Moderate confidence
    UNCERTAIN = "uncertain"    # Low confidence, needs more data


@dataclass
class DetectedPattern:
    """A detected pattern or anomaly in satellite behavior.

    Attributes:
        id: Pattern ID
        satellite_id: Satellite that exhibited the pattern
        satellite_name: Satellite name
        pattern_type: Type of pattern detected
        detected_at: When the pattern was detected
        start_time: When the pattern/event started
        end_time: When the pattern/event ended (if known)
        confidence: Detection confidence level
        description: Human-readable description
        delta_v_m_s: Estimated delta-v if maneuver (m/s)
        altitude_change_km: Change in altitude (km)
        inclination_change_deg: Change in inclination (degrees)
        details: Additional pattern-specific details
    """

    id: str
    satellite_id: str
    satellite_name: str
    pattern_type: PatternType
    detected_at: datetime
    start_time: datetime
    end_time: Optional[datetime] = None
    confidence: ConfidenceLevel = ConfidenceLevel.LIKELY
    description: str = ""
    delta_v_m_s: Optional[float] = None
    altitude_change_km: Optional[float] = None
    inclination_change_deg: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectedPattern":
        """Create DetectedPattern from API response."""
        pattern_str = data.get("pattern_type", "anomaly")
        try:
            pattern_type = PatternType(pattern_str.lower())
        except ValueError:
            pattern_type = PatternType.ANOMALY

        confidence_str = data.get("confidence", "likely")
        try:
            confidence = ConfidenceLevel(confidence_str.lower())
        except ValueError:
            confidence = ConfidenceLevel.UNCERTAIN

        return cls(
            id=data["id"],
            satellite_id=data["satellite_id"],
            satellite_name=data.get("satellite_name", "Unknown"),
            pattern_type=pattern_type,
            detected_at=datetime.fromisoformat(
                data["detected_at"].replace("Z", "+00:00")
            ),
            start_time=datetime.fromisoformat(
                data["start_time"].replace("Z", "+00:00")
            ),
            end_time=datetime.fromisoformat(
                data["end_time"].replace("Z", "+00:00")
            ) if data.get("end_time") else None,
            confidence=confidence,
            description=data.get("description", ""),
            delta_v_m_s=data.get("delta_v_m_s"),
            altitude_change_km=data.get("altitude_change_km"),
            inclination_change_deg=data.get("inclination_change_deg"),
            details=data.get("details"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "satellite_id": self.satellite_id,
            "satellite_name": self.satellite_name,
            "pattern_type": self.pattern_type.value,
            "detected_at": self.detected_at.isoformat(),
            "start_time": self.start_time.isoformat(),
            "confidence": self.confidence.value,
            "description": self.description,
        }
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.delta_v_m_s is not None:
            result["delta_v_m_s"] = self.delta_v_m_s
        if self.altitude_change_km is not None:
            result["altitude_change_km"] = self.altitude_change_km
        if self.inclination_change_deg is not None:
            result["inclination_change_deg"] = self.inclination_change_deg
        if self.details:
            result["details"] = self.details
        return result

    @property
    def is_maneuver(self) -> bool:
        """Check if this pattern is any type of maneuver."""
        return self.pattern_type in (
            PatternType.MANEUVER,
            PatternType.ORBIT_RAISE,
            PatternType.ORBIT_LOWER,
            PatternType.PLANE_CHANGE,
            PatternType.DEORBIT,
            PatternType.STATION_KEEPING,
            PatternType.DEBRIS_AVOIDANCE,
        )

    @property
    def is_anomaly(self) -> bool:
        """Check if this is an anomalous pattern."""
        return self.pattern_type in (
            PatternType.ANOMALY,
            PatternType.TUMBLING,
            PatternType.FRAGMENTATION,
        )


class PatternDetector:
    """Detect patterns and anomalies in satellite behavior.

    Example:
        >>> from rotastellar_intel import PatternDetector
        >>> detector = PatternDetector(api_key="rs_live_xxx")
        >>>
        >>> # Get recent maneuvers for a satellite
        >>> maneuvers = detector.get_maneuvers("starlink-1234")
        >>> for m in maneuvers:
        ...     print(f"{m.pattern_type.value}: {m.description}")
        >>>
        >>> # Get all anomalies in the last 24 hours
        >>> anomalies = detector.get_anomalies(hours=24)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[RotaStellarClient] = None,
        **kwargs
    ):
        """Initialize the pattern detector.

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

    def get_patterns(
        self,
        satellite_id: Optional[str] = None,
        *,
        pattern_types: Optional[List[PatternType]] = None,
        time_range: Optional[TimeRange] = None,
        confidence_min: ConfidenceLevel = ConfidenceLevel.POSSIBLE,
        limit: int = 100
    ) -> List[DetectedPattern]:
        """Get detected patterns with filtering.

        Args:
            satellite_id: Filter by satellite
            pattern_types: Filter by pattern types
            time_range: Time window to search
            confidence_min: Minimum confidence level
            limit: Maximum results

        Returns:
            List of detected patterns
        """
        # Build query parameters
        params: Dict[str, Any] = {"limit": limit}

        if satellite_id:
            params["satellite_id"] = satellite_id
        if pattern_types:
            params["pattern_types"] = [p.value for p in pattern_types]
        if time_range:
            params["start"] = time_range.start.isoformat()
            params["end"] = time_range.end.isoformat()

        # Call API
        response = self._client.http.get("/patterns", params=params)
        raw_patterns = response.get("data", [])

        # Parse and filter by confidence
        patterns = [DetectedPattern.from_dict(p) for p in raw_patterns]

        confidence_order = [
            ConfidenceLevel.UNCERTAIN,
            ConfidenceLevel.POSSIBLE,
            ConfidenceLevel.LIKELY,
            ConfidenceLevel.CONFIRMED,
        ]
        min_idx = confidence_order.index(confidence_min)

        return [
            p for p in patterns
            if confidence_order.index(p.confidence) >= min_idx
        ]

    def get_maneuvers(
        self,
        satellite_id: Optional[str] = None,
        hours: float = 168  # 7 days
    ) -> List[DetectedPattern]:
        """Get detected maneuvers.

        Args:
            satellite_id: Filter by satellite
            hours: Time window in hours

        Returns:
            List of detected maneuvers
        """
        maneuver_types = [
            PatternType.MANEUVER,
            PatternType.ORBIT_RAISE,
            PatternType.ORBIT_LOWER,
            PatternType.PLANE_CHANGE,
            PatternType.DEORBIT,
            PatternType.STATION_KEEPING,
            PatternType.DEBRIS_AVOIDANCE,
        ]

        return self.get_patterns(
            satellite_id=satellite_id,
            pattern_types=maneuver_types,
            time_range=TimeRange.next_hours(hours) if hours > 0 else None,
        )

    def get_anomalies(
        self,
        satellite_id: Optional[str] = None,
        hours: float = 24
    ) -> List[DetectedPattern]:
        """Get detected anomalies.

        Args:
            satellite_id: Filter by satellite
            hours: Time window in hours

        Returns:
            List of detected anomalies
        """
        anomaly_types = [
            PatternType.ANOMALY,
            PatternType.TUMBLING,
            PatternType.FRAGMENTATION,
        ]

        return self.get_patterns(
            satellite_id=satellite_id,
            pattern_types=anomaly_types,
            time_range=TimeRange.next_hours(hours) if hours > 0 else None,
        )

    def get_proximity_events(
        self,
        satellite_id: Optional[str] = None,
        hours: float = 168
    ) -> List[DetectedPattern]:
        """Get detected proximity operations.

        Args:
            satellite_id: Filter by satellite
            hours: Time window in hours

        Returns:
            List of proximity events
        """
        proximity_types = [
            PatternType.PROXIMITY_OPS,
            PatternType.RENDEZVOUS,
        ]

        return self.get_patterns(
            satellite_id=satellite_id,
            pattern_types=proximity_types,
            time_range=TimeRange.next_hours(hours) if hours > 0 else None,
        )

    def analyze_behavior(
        self,
        satellite_id: str,
        hours: float = 720  # 30 days
    ) -> Dict[str, Any]:
        """Analyze satellite behavior over time.

        Args:
            satellite_id: Satellite to analyze
            hours: Analysis window in hours

        Returns:
            Behavior analysis summary
        """
        time_range = TimeRange.next_hours(hours)
        patterns = self.get_patterns(
            satellite_id=satellite_id,
            time_range=time_range,
            limit=500
        )

        # Categorize patterns
        by_type: Dict[str, List[DetectedPattern]] = {}
        for p in patterns:
            type_name = p.pattern_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(p)

        # Calculate statistics
        maneuvers = [p for p in patterns if p.is_maneuver]
        anomalies = [p for p in patterns if p.is_anomaly]

        total_delta_v = sum(
            p.delta_v_m_s for p in maneuvers
            if p.delta_v_m_s is not None
        )

        return {
            "satellite_id": satellite_id,
            "analysis_window_hours": hours,
            "total_patterns": len(patterns),
            "patterns_by_type": {
                t: len(items) for t, items in by_type.items()
            },
            "maneuver_count": len(maneuvers),
            "anomaly_count": len(anomalies),
            "total_delta_v_m_s": total_delta_v,
            "average_maneuvers_per_week": len(maneuvers) / (hours / 168) if hours > 0 else 0,
            "has_anomalies": len(anomalies) > 0,
        }
