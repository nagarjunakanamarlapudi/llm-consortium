"""Token-bucket rate limiter with per-second smoothing for LLM APIs.

Enforces two independent limits:
  1. **Requests per minute (RPM)** — hard cap on call frequency.
  2. **Tokens per minute (TPM)** — optional cap on total tokens consumed.

Both use a token-bucket implementation that refills at a fixed per-second rate
to prevent burst-and-wait behaviour (e.g. firing 500 requests in 1 second then
idling for 59 seconds).

Thread-safe via ``asyncio.Lock``; intended for use inside the
``async_batcher`` flush path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateLimiterStats:
    """Cumulative statistics for a rate limiter instance."""

    total_acquires: int = 0
    total_waits: int = 0
    total_wait_ms: float = 0.0
    peak_wait_ms: float = 0.0

    @property
    def avg_wait_ms(self) -> float:
        return self.total_wait_ms / self.total_waits if self.total_waits else 0.0


@dataclass
class RateLimiterConfig:
    """Rate limit configuration for a single model/provider."""

    requests_per_minute: int = 0
    tokens_per_minute: int = 0


class TokenBucketRateLimiter:
    """Dual token-bucket rate limiter (RPM + TPM).

    Buckets refill at a smoothed per-second rate:
        ``refill_per_second = limit_per_minute / 60``

    This prevents the "thundering herd" pattern where all capacity is
    consumed in the first second of each minute.

    Usage::

        limiter = TokenBucketRateLimiter(
            requests_per_minute=500,
            tokens_per_minute=150_000,
        )

        # Before each LLM call:
        await limiter.acquire(estimated_tokens=2048)
    """

    def __init__(
        self,
        requests_per_minute: int = 0,
        tokens_per_minute: int = 0,
        *,
        name: str = "",
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._name = name

        # Request bucket
        self._req_tokens = float(self._rpm) if self._rpm > 0 else float("inf")
        self._req_refill_rate = self._rpm / 60.0 if self._rpm > 0 else 0.0
        self._req_max = float(self._rpm) if self._rpm > 0 else float("inf")

        # Token bucket
        self._tok_tokens = float(self._tpm) if self._tpm > 0 else float("inf")
        self._tok_refill_rate = self._tpm / 60.0 if self._tpm > 0 else 0.0
        self._tok_max = float(self._tpm) if self._tpm > 0 else float("inf")

        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        self.stats = RateLimiterStats()

        self._log = logger.bind(
            rate_limiter=name or "default",
            rpm=self._rpm,
            tpm=self._tpm,
        )

    @property
    def enabled(self) -> bool:
        """True if any limit is configured."""
        return self._rpm > 0 or self._tpm > 0

    @property
    def config(self) -> RateLimiterConfig:
        return RateLimiterConfig(
            requests_per_minute=self._rpm,
            tokens_per_minute=self._tpm,
        )

    async def acquire(self, estimated_tokens: int = 1) -> None:
        """Wait until capacity is available, then consume one request + tokens.

        Args:
            estimated_tokens: Estimated token consumption for this request.
                Defaults to 1 (request-only limiting). Pass higher values
                for token-based rate limiting.
        """
        if not self.enabled:
            return

        self.stats.total_acquires += 1
        wait_start = time.monotonic()
        waited = False

        async with self._lock:
            while True:
                self._refill()

                # Check request bucket
                req_ok = self._req_tokens >= 1.0
                # Check token bucket
                tok_ok = self._tok_tokens >= estimated_tokens

                if req_ok and tok_ok:
                    # Consume from both buckets
                    if self._rpm > 0:
                        self._req_tokens -= 1.0
                    if self._tpm > 0:
                        self._tok_tokens -= estimated_tokens
                    break

                # Calculate wait time until enough capacity
                wait_s = 0.0
                if not req_ok and self._req_refill_rate > 0:
                    needed = 1.0 - self._req_tokens
                    wait_s = max(wait_s, needed / self._req_refill_rate)
                if not tok_ok and self._tok_refill_rate > 0:
                    needed = estimated_tokens - self._tok_tokens
                    wait_s = max(wait_s, needed / self._tok_refill_rate)

                # Minimum wait to avoid busy-spinning
                wait_s = max(wait_s, 0.01)
                waited = True

                # Release lock while sleeping so other coroutines can check
                self._lock.release()
                try:
                    await asyncio.sleep(wait_s)
                finally:
                    await self._lock.acquire()

        if waited:
            wait_ms = (time.monotonic() - wait_start) * 1000.0
            self.stats.total_waits += 1
            self.stats.total_wait_ms += wait_ms
            self.stats.peak_wait_ms = max(self.stats.peak_wait_ms, wait_ms)

            self._log.debug(
                "rate_limiter.waited",
                wait_ms=round(wait_ms, 1),
                tokens=estimated_tokens,
            )

    async def update_limits(self, rpm: int = 0, tpm: int = 0) -> None:
        """Dynamically adjust limits from provider response headers.

        Only updates if the new values differ from current ones.
        The bucket is re-scaled proportionally: if you had 50% capacity
        remaining and the limit doubles, you'll have 50% of the new limit.

        Args:
            rpm: New requests-per-minute limit (0 = no change).
            tpm: New tokens-per-minute limit (0 = no change).
        """
        async with self._lock:
            changed = False

            if rpm > 0 and rpm != self._rpm:
                # Scale current bucket proportionally
                ratio = (
                    self._req_tokens / self._req_max
                    if self._req_max > 0 and self._req_max != float("inf")
                    else 1.0
                )
                self._rpm = rpm
                self._req_max = float(rpm)
                self._req_refill_rate = rpm / 60.0
                self._req_tokens = min(self._req_max, self._req_max * ratio)
                changed = True

            if tpm > 0 and tpm != self._tpm:
                ratio = (
                    self._tok_tokens / self._tok_max
                    if self._tok_max > 0 and self._tok_max != float("inf")
                    else 1.0
                )
                self._tpm = tpm
                self._tok_max = float(tpm)
                self._tok_refill_rate = tpm / 60.0
                self._tok_tokens = min(self._tok_max, self._tok_max * ratio)
                changed = True

            if changed:
                self._log.info(
                    "rate_limiter.limits_updated",
                    rpm=self._rpm,
                    tpm=self._tpm,
                )

    def _refill(self) -> None:
        """Refill both buckets based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        if self._rpm > 0:
            self._req_tokens = min(
                self._req_max,
                self._req_tokens + elapsed * self._req_refill_rate,
            )
        if self._tpm > 0:
            self._tok_tokens = min(
                self._tok_max,
                self._tok_tokens + elapsed * self._tok_refill_rate,
            )

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot for dashboard display."""
        return {
            "rpm_limit": self._rpm,
            "tpm_limit": self._tpm,
            "rpm_available": round(self._req_tokens, 1) if self._rpm > 0 else None,
            "tpm_available": round(self._tok_tokens, 0) if self._tpm > 0 else None,
            "stats": {
                "total_acquires": self.stats.total_acquires,
                "total_waits": self.stats.total_waits,
                "total_wait_ms": round(self.stats.total_wait_ms, 1),
                "peak_wait_ms": round(self.stats.peak_wait_ms, 1),
                "avg_wait_ms": round(self.stats.avg_wait_ms, 1),
            },
        }
