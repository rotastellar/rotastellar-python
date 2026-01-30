"""
RotaStellar - Space Computing Infrastructure

Python SDK for orbital compute planning, simulation, and space intelligence.

Documentation: https://rotastellar.com/docs
GitHub: https://github.com/rotastellar/rotastellar-python

Example:
    >>> from rotastellar import RotaStellarClient
    >>> client = RotaStellarClient(api_key="rs_live_xxx")
    >>>
    >>> # List satellites
    >>> satellites = client.list_satellites(constellation="starlink")
    >>> for sat in satellites:
    ...     print(f"{sat.name}: {sat.norad_id}")
    >>>
    >>> # Analyze orbital compute feasibility
    >>> result = client.analyze_feasibility(
    ...     workload_type="inference",
    ...     compute_tflops=10,
    ...     data_gb=1.5
    ... )
    >>> print(f"Feasible: {result['feasible']}")
"""

__version__ = "0.1.0"
__author__ = "Rota, Inc."
__email__ = "hello@rotastellar.com"
__url__ = "https://rotastellar.com"

# Main client
from .client import RotaStellarClient

# Types
from .types import (
    Position,
    Orbit,
    Satellite,
    TimeRange,
    EARTH_RADIUS_KM,
    EARTH_MU,
)

# Configuration
from .config import Config, get_default_config, set_default_config

# Errors
from .errors import (
    RotaStellarError,
    AuthenticationError,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    APIError,
    RateLimitError,
    NotFoundError,
    ValidationError,
    NetworkError,
    TimeoutError,
)

# Auth utilities
from .auth import validate_api_key, mask_api_key

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    # Main client
    "RotaStellarClient",
    # Types
    "Position",
    "Orbit",
    "Satellite",
    "TimeRange",
    "EARTH_RADIUS_KM",
    "EARTH_MU",
    # Config
    "Config",
    "get_default_config",
    "set_default_config",
    # Errors
    "RotaStellarError",
    "AuthenticationError",
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
    "APIError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    "NetworkError",
    "TimeoutError",
    # Auth
    "validate_api_key",
    "mask_api_key",
]
