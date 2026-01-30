"""
RotaStellar SDK - HTTP Client

HTTP client with retries, rate limiting, and error handling.
"""

import json
import time
from typing import Any, Dict, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin

from .auth import get_auth_header, mask_api_key
from .config import Config
from .errors import (
    APIError,
    RateLimitError,
    NotFoundError,
    NetworkError,
    TimeoutError as SDKTimeoutError,
)


class HTTPClient:
    """HTTP client for RotaStellar API.

    Handles authentication, retries, rate limiting, and error responses.

    Attributes:
        config: SDK configuration
        _request_count: Number of requests made
    """

    USER_AGENT = "rotastellar-python/0.1.0"

    def __init__(self, config: Config):
        """Initialize HTTP client.

        Args:
            config: SDK configuration with API key and settings
        """
        self.config = config
        self._request_count = 0

    def _build_url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build full URL with query parameters.

        Args:
            path: API path (e.g., "/satellites")
            params: Query parameters

        Returns:
            Full URL string
        """
        url = urljoin(self.config.base_url, path.lstrip("/"))
        if params:
            # Filter out None values and convert to strings
            filtered = {k: str(v) for k, v in params.items() if v is not None}
            if filtered:
                url = f"{url}?{urlencode(filtered)}"
        return url

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers.

        Returns:
            Dictionary of headers
        """
        headers = {
            "User-Agent": self.USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers.update(get_auth_header(self.config.api_key))
        return headers

    def _parse_error_response(
        self, response_body: bytes, status_code: int, request_id: Optional[str]
    ) -> APIError:
        """Parse error response and return appropriate exception.

        Args:
            response_body: Response body bytes
            status_code: HTTP status code
            request_id: Request ID from response headers

        Returns:
            Appropriate APIError subclass
        """
        try:
            data = json.loads(response_body.decode("utf-8"))
            message = data.get("error", {}).get("message", "Unknown error")
            details = data.get("error", {}).get("details")
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = "Unknown error"
            details = None

        if status_code == 429:
            retry_after = None
            return RateLimitError(retry_after=retry_after, request_id=request_id)
        elif status_code == 404:
            return NotFoundError(
                resource_type="Resource",
                resource_id="unknown",
                request_id=request_id,
            )
        else:
            return APIError(
                message=message,
                status_code=status_code,
                request_id=request_id,
                details=details,
            )

    def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a single HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            params: Query parameters
            data: Request body data

        Returns:
            Parsed JSON response

        Raises:
            APIError: On API error response
            NetworkError: On network error
            SDKTimeoutError: On timeout
        """
        url = self._build_url(path, params)
        headers = self._build_headers()

        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        request = Request(url, data=body, headers=headers, method=method)

        if self.config.debug:
            masked_key = (
                mask_api_key(self.config.api_key)
                if self.config.api_key
                else "None"
            )
            print(f"[DEBUG] {method} {url} (api_key: {masked_key})")

        try:
            timeout = self.config.timeout
            with urlopen(request, timeout=timeout) as response:
                self._request_count += 1
                response_body = response.read()
                return json.loads(response_body.decode("utf-8"))

        except HTTPError as e:
            request_id = e.headers.get("x-request-id")
            response_body = e.read()
            raise self._parse_error_response(response_body, e.code, request_id)

        except URLError as e:
            if "timed out" in str(e.reason).lower():
                raise SDKTimeoutError(self.config.timeout)
            raise NetworkError(f"Network error: {e.reason}", original_error=e)

        except TimeoutError:
            raise SDKTimeoutError(self.config.timeout)

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            data: Request body

        Returns:
            Parsed JSON response

        Raises:
            APIError: On API error after all retries
            NetworkError: On network error after all retries
        """
        last_error: Optional[Exception] = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                return self._make_request(method, path, params, data)

            except RateLimitError as e:
                last_error = e
                # Use retry-after if provided, otherwise exponential backoff
                wait_time = e.retry_after if e.retry_after else delay
                if attempt < self.config.max_retries:
                    if self.config.debug:
                        print(f"[DEBUG] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    delay *= 2

            except (NetworkError, SDKTimeoutError) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    if self.config.debug:
                        print(f"[DEBUG] Network error, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2

            except APIError:
                # Don't retry other API errors
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise NetworkError("Request failed after all retries")

    def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GET request.

        Args:
            path: API path
            params: Query parameters

        Returns:
            Parsed JSON response
        """
        return self.request("GET", path, params=params)

    def post(
        self, path: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make POST request.

        Args:
            path: API path
            data: Request body

        Returns:
            Parsed JSON response
        """
        return self.request("POST", path, data=data)

    def put(
        self, path: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make PUT request.

        Args:
            path: API path
            data: Request body

        Returns:
            Parsed JSON response
        """
        return self.request("PUT", path, data=data)

    def delete(self, path: str) -> Dict[str, Any]:
        """Make DELETE request.

        Args:
            path: API path

        Returns:
            Parsed JSON response
        """
        return self.request("DELETE", path)
