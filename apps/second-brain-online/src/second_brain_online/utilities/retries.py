# infrastructure/resilience/retries.py
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_exception
)
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout,
    AuthenticationError,
    BadRequestError,
)
from pymongo.errors import ServerSelectionTimeoutError, NetworkTimeout, AutoReconnect
from openai import APITimeoutError, APIConnectionError as OpenAIConnectionError


# ─────────────────────────────────────────────────────────────────────────────
# Loguru-compatible before_sleep callback
# Replaces tenacity's before_sleep_log which only accepts stdlib logging.Logger
# ─────────────────────────────────────────────────────────────────────────────

def _before_sleep_log(retry_state) -> None:
    exc = retry_state.outcome.exception()
    logger.warning(
        f"Retrying {retry_state.fn.__name__} "
        f"(attempt {retry_state.attempt_number}) "
        f"after error: {type(exc).__name__}: {exc} — "
        f"sleeping {retry_state.next_action.sleep:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────────────

def _is_quota_error(e: Exception) -> bool:
    """Quota errors will never succeed on retry — fail fast."""
    msg = str(e).lower()
    return any(kw in msg for kw in [
        "resource_exhausted",
        "quota exceeded",
        "quota_exceeded",
        "billing",
        "insufficient_quota",
        "you exceeded your current quota",
    ])


def _is_retryable_llm_error(e: Exception) -> bool:
    if isinstance(e, (AuthenticationError, BadRequestError)):
        return False
    if _is_quota_error(e):
        return False
    return isinstance(e, (
        RateLimitError,
        ServiceUnavailableError,
        APIConnectionError,
        Timeout,
    ))


def _is_retryable_mongo_error(e: Exception) -> bool:
    return isinstance(e, (
        ServerSelectionTimeoutError,
        NetworkTimeout,
        AutoReconnect,
    ))


def _is_retryable_hf_error(e: Exception) -> bool:
    if _is_quota_error(e):
        return False
    return isinstance(e, (
        APITimeoutError,
        OpenAIConnectionError,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorators
# ─────────────────────────────────────────────────────────────────────────────

gemini_retry = retry(
    retry=retry_if_exception(_is_retryable_llm_error),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    stop=stop_after_attempt(3),
    before_sleep=_before_sleep_log,
    reraise=True,
)

mongo_retry = retry(
    retry=retry_if_exception(_is_retryable_mongo_error),
    wait=wait_fixed(2),
    stop=stop_after_attempt(2),
    before_sleep=_before_sleep_log,
    reraise=True,
)

hf_retry = retry(
    retry=retry_if_exception(_is_retryable_hf_error),
    wait=wait_fixed(3),
    stop=stop_after_attempt(2),
    before_sleep=_before_sleep_log,
    reraise=True,
)