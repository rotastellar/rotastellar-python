"""
RotaStellar SDK - Authentication

API key validation and authentication handling.
"""

import re
from typing import Optional, Tuple

from .errors import MissingAPIKeyError, InvalidAPIKeyError


# API key patterns
API_KEY_PATTERN = re.compile(r"^rs_(live|test)_[a-zA-Z0-9]{16,}$")
API_KEY_PREFIX_LIVE = "rs_live_"
API_KEY_PREFIX_TEST = "rs_test_"


def validate_api_key(api_key: Optional[str]) -> Tuple[bool, str]:
    """Validate an API key and return its environment.

    Args:
        api_key: The API key to validate

    Returns:
        Tuple of (is_valid, environment)
        environment is 'live', 'test', or empty string if invalid

    Raises:
        MissingAPIKeyError: If api_key is None or empty
        InvalidAPIKeyError: If api_key format is invalid
    """
    if api_key is None or api_key.strip() == "":
        raise MissingAPIKeyError()

    api_key = api_key.strip()

    # Check prefix
    if api_key.startswith(API_KEY_PREFIX_LIVE):
        environment = "live"
    elif api_key.startswith(API_KEY_PREFIX_TEST):
        environment = "test"
    else:
        raise InvalidAPIKeyError(api_key)

    # Validate full pattern
    if not API_KEY_PATTERN.match(api_key):
        raise InvalidAPIKeyError(api_key)

    return True, environment


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe logging.

    Args:
        api_key: The API key to mask

    Returns:
        Masked API key showing only prefix and last 4 characters

    Example:
        >>> mask_api_key("rs_live_abc123def456xyz")
        'rs_live_****xyz'
    """
    if len(api_key) <= 12:
        return api_key[:8] + "****"

    prefix_end = 8  # "rs_live_" or "rs_test_"
    return api_key[:prefix_end] + "****" + api_key[-4:]


def get_auth_header(api_key: str) -> dict:
    """Get the authorization header for API requests.

    Args:
        api_key: The API key

    Returns:
        Dictionary with Authorization header
    """
    return {"Authorization": f"Bearer {api_key}"}
