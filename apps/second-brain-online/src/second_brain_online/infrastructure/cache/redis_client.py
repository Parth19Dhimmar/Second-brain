import redis
from loguru import logger
from second_brain_online.config import settings

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """
    Singleton Redis client — reusable anywhere in the codebase.
    Returns None gracefully if Redis is unavailable (caching degrades silently).
    """
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,      # exact cache stores plain strings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            _redis_client.ping()
            logger.info(f"Redis connected: {settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis unavailable — exact caching disabled: {e}")
            _redis_client = None

    return _redis_client