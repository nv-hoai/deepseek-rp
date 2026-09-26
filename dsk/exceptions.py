"""Typed error hierarchy. HTTP/WAF/biz-code mapping lives in transport."""


class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors."""


class AuthenticationError(DeepSeekError):
    """Invalid or expired token (HTTP 401 or code/biz_code 40002/40003)."""


class RateLimitError(DeepSeekError):
    """Rate limited (HTTP 429)."""


class NetworkError(DeepSeekError):
    """Transport-level failure."""


class WafError(DeepSeekError):
    """Blocked by AWS WAF / Cloudflare (challenge/captcha, CloudFront 403)."""


class APIError(DeepSeekError):
    """Any other API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
