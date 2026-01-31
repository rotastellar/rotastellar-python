"""
RotaStellar SDK - Async Client

Async client using httpx for high-performance concurrent requests.
"""

from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for async support. Install with: pip install rotastellar[async]"
    )

from .auth import validate_api_key, get_auth_header
from .config import Config
from .errors import (
    APIError,
    RateLimitError,
    NotFoundError,
    NetworkError,
    TimeoutError as SDKTimeoutError,
)
from .types import Position, Orbit, Satellite, TimeRange


class AsyncRotaStellarClient:
    """Async client for the RotaStellar API.

    This provides async/await support for high-performance concurrent requests.

    Example:
        >>> import asyncio
        >>> from rotastellar import AsyncRotaStellarClient
        >>>
        >>> async def main():
        ...     client = AsyncRotaStellarClient(api_key="rs_live_xxx")
        ...     iss = await client.get_satellite("25544")
        ...     print(f"ISS: {iss.position.lat}, {iss.position.lon}")
        >>>
        >>> asyncio.run(main())
    """

    USER_AGENT = "rotastellar-python/0.1.0 (async)"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        debug: bool = False,
    ):
        """Initialize the async RotaStellar client.

        Args:
            api_key: RotaStellar API key
            base_url: API base URL (default: https://api.rotastellar.com/v1)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts (default: 3)
            debug: Enable debug logging
        """
        self.config = Config(api_key=api_key, debug=debug)

        if base_url is not None:
            self.config.base_url = base_url
        if timeout is not None:
            self.config.timeout = timeout
        if max_retries is not None:
            self.config.max_retries = max_retries

        validate_api_key(self.config.api_key)

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None:
            headers = {
                "User-Agent": self.USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.config.api_key:
                headers.update(get_auth_header(self.config.api_key))

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _parse_error(self, response: httpx.Response) -> APIError:
        """Parse error response into appropriate exception."""
        try:
            data = response.json()
            message = data.get("error", {}).get("message", "Unknown error")
        except Exception:
            message = "Unknown error"

        request_id = response.headers.get("x-request-id")

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            return RateLimitError(
                retry_after=int(retry_after) if retry_after else None,
                request_id=request_id,
            )
        elif response.status_code == 404:
            return NotFoundError(
                resource_type="Resource",
                resource_id="unknown",
                request_id=request_id,
            )
        else:
            return APIError(
                message=message,
                status_code=response.status_code,
                request_id=request_id,
            )

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        client = await self._get_client()

        # Filter None params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        last_error: Optional[Exception] = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(
                    method, path, params=params, json=json_data
                )

                if response.status_code >= 400:
                    error = self._parse_error(response)

                    if isinstance(error, RateLimitError) and attempt < self.config.max_retries:
                        import asyncio
                        wait_time = error.retry_after if error.retry_after else delay
                        await asyncio.sleep(wait_time)
                        delay *= 2
                        last_error = error
                        continue

                    raise error

                return response.json()

            except httpx.TimeoutException:
                last_error = SDKTimeoutError(self.config.timeout)
                if attempt < self.config.max_retries:
                    import asyncio
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise last_error

            except httpx.RequestError as e:
                last_error = NetworkError(str(e), original_error=e)
                if attempt < self.config.max_retries:
                    import asyncio
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise last_error

        if last_error:
            raise last_error
        raise NetworkError("Request failed after all retries")

    # =========================================================================
    # Satellite Operations
    # =========================================================================

    async def list_satellites(
        self,
        *,
        constellation: Optional[str] = None,
        operator: Optional[str] = None,
        satellite_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Satellite]:
        """List satellites with optional filtering."""
        params = {
            "constellation": constellation,
            "operator": operator,
            "type": satellite_type,
            "limit": limit,
            "offset": offset,
        }
        response = await self._request("GET", "/satellites", params=params)
        satellites = response.get("data", [])
        return [Satellite.from_dict(sat) for sat in satellites]

    async def get_satellite(self, satellite_id: str) -> Satellite:
        """Get a specific satellite by ID."""
        response = await self._request("GET", f"/satellites/{satellite_id}")
        return Satellite.from_dict(response)

    async def get_satellite_position(
        self, satellite_id: str, at_time: Optional[str] = None
    ) -> Position:
        """Get satellite position at a specific time."""
        params = {"at": at_time} if at_time else None
        response = await self._request(
            "GET", f"/satellites/{satellite_id}/position", params=params
        )
        return Position.from_dict(response)

    async def get_trajectory(
        self,
        satellite_id: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval_sec: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get predicted trajectory for a satellite."""
        params: Dict[str, Any] = {"interval_sec": interval_sec}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = await self._request(
            "GET", f"/satellites/{satellite_id}/trajectory", params=params
        )
        return response.get("points", [])

    # =========================================================================
    # Conjunction Analysis
    # =========================================================================

    async def list_conjunctions(
        self,
        *,
        satellite_id: Optional[str] = None,
        threshold_km: float = 1.0,
        time_range: Optional[TimeRange] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List close approaches (conjunctions) between space objects."""
        params: Dict[str, Any] = {
            "satellite_id": satellite_id,
            "threshold_km": threshold_km,
            "limit": limit,
        }
        if time_range:
            params["start"] = time_range.start.isoformat()
            params["end"] = time_range.end.isoformat()

        response = await self._request("GET", "/conjunctions", params=params)
        return response.get("data", [])

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    async def list_patterns(
        self,
        *,
        satellite_id: str,
        lookback_days: int = 30,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies and maneuvers in satellite behavior."""
        params: Dict[str, Any] = {
            "satellite": satellite_id,
            "lookback_days": lookback_days,
            "min_confidence": min_confidence,
        }
        if pattern_type:
            params["type"] = pattern_type

        response = await self._request("GET", "/patterns", params=params)
        return response.get("patterns", [])

    # =========================================================================
    # Planning Operations
    # =========================================================================

    async def analyze_feasibility(
        self,
        *,
        workload_type: str,
        compute_tflops: float,
        data_gb: float,
        latency_requirement_ms: Optional[float] = None,
        orbit_altitude_km: float = 550,
    ) -> Dict[str, Any]:
        """Analyze feasibility of orbital compute for a workload."""
        data = {
            "workload_type": workload_type,
            "compute_tflops": compute_tflops,
            "data_gb": data_gb,
            "latency_requirement_ms": latency_requirement_ms,
            "orbit_altitude_km": orbit_altitude_km,
        }
        return await self._request("POST", "/planning/analyze", json_data=data)

    async def simulate_thermal(
        self,
        *,
        power_watts: float,
        orbit_altitude_km: float = 550,
        radiator_area_m2: float = 1.0,
        duration_hours: float = 24,
    ) -> Dict[str, Any]:
        """Simulate thermal conditions for orbital compute."""
        data = {
            "power_watts": power_watts,
            "orbit_altitude_km": orbit_altitude_km,
            "radiator_area_m2": radiator_area_m2,
            "duration_hours": duration_hours,
        }
        return await self._request("POST", "/planning/thermal", json_data=data)

    async def simulate_latency(
        self,
        *,
        source: Position,
        destination: Position,
        orbit_altitude_km: float = 550,
        relay_count: int = 0,
    ) -> Dict[str, Any]:
        """Simulate network latency through orbital infrastructure."""
        data = {
            "source": source.to_dict(),
            "destination": destination.to_dict(),
            "orbit_altitude_km": orbit_altitude_km,
            "relay_count": relay_count,
        }
        return await self._request("POST", "/planning/latency", json_data=data)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def ping(self) -> Dict[str, Any]:
        """Check API connectivity and authentication."""
        return await self._request("GET", "/ping")

    def __repr__(self) -> str:
        """String representation of client."""
        env = self.config.environment or "unknown"
        return f"AsyncRotaStellarClient(environment={env})"
