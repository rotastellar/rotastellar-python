"""
RotaStellar Intel - Orbital Intelligence & Space Situational Awareness

Tools for tracking, analyzing, and monitoring orbital activity.

Documentation: https://rotastellar.com/docs/intel
GitHub: https://github.com/rotastellar/rotastellar-python

Example:
    >>> from rotastellar_intel import Tracker, ConjunctionAnalyzer
    >>>
    >>> # Track a satellite
    >>> tracker = Tracker(api_key="rs_live_xxx")
    >>> iss = tracker.track("ISS")
    >>> pos = iss.position()
    >>> print(f"ISS at {pos.latitude:.2f}, {pos.longitude:.2f}")
    >>>
    >>> # Analyze conjunctions
    >>> analyzer = ConjunctionAnalyzer(api_key="rs_live_xxx")
    >>> conjunctions = analyzer.get_high_risk_conjunctions()
    >>> for c in conjunctions:
    ...     print(f"{c.primary_name} - {c.miss_distance_km:.2f} km")
"""

__version__ = "0.1.0"
__author__ = "Rota, Inc."
__email__ = "hello@rotastellar.com"
__url__ = "https://rotastellar.com"

# Tracker
from .tracker import (
    Tracker,
    TrackedSatellite,
    GroundStation,
    SatellitePass,
)

# TLE
from .tle import (
    TLE,
    parse_tle,
)

# Conjunctions
from .conjunctions import (
    ConjunctionAnalyzer,
    Conjunction,
    ManeuverRecommendation,
    RiskLevel,
)

# Patterns
from .patterns import (
    PatternDetector,
    DetectedPattern,
    PatternType,
    ConfidenceLevel,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    # Tracker
    "Tracker",
    "TrackedSatellite",
    "GroundStation",
    "SatellitePass",
    # TLE
    "TLE",
    "parse_tle",
    # Conjunctions
    "ConjunctionAnalyzer",
    "Conjunction",
    "ManeuverRecommendation",
    "RiskLevel",
    # Patterns
    "PatternDetector",
    "DetectedPattern",
    "PatternType",
    "ConfidenceLevel",
]
