"""
RotaStellar SDK - Configuration

SDK configuration and settings.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    """SDK configuration settings.

    Attributes:
        api_key: RotaStellar API key
        base_url: API base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        debug: Enable debug logging

    Example:
        >>> config = Config(api_key="rs_live_xxx")
        >>> print(config.base_url)
        https://api.rotastellar.com/v1
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.rotastellar.com/v1"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    debug: bool = False

    def __post_init__(self):
        # Load from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("ROTASTELLAR_API_KEY")

        # Allow base URL override from environment
        env_base_url = os.environ.get("ROTASTELLAR_BASE_URL")
        if env_base_url:
            self.base_url = env_base_url

        # Debug mode from environment
        if os.environ.get("ROTASTELLAR_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True

    @property
    def is_test_key(self) -> bool:
        """Check if using a test API key."""
        return self.api_key is not None and self.api_key.startswith("rs_test_")

    @property
    def is_live_key(self) -> bool:
        """Check if using a live API key."""
        return self.api_key is not None and self.api_key.startswith("rs_live_")

    @property
    def environment(self) -> Optional[str]:
        """Get the API key environment (test or live)."""
        if self.is_test_key:
            return "test"
        elif self.is_live_key:
            return "live"
        return None


# Default configuration
_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """Get the default SDK configuration."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_default_config(config: Config) -> None:
    """Set the default SDK configuration."""
    global _default_config
    _default_config = config
