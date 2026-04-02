#!/usr/bin/env python3
"""
warm_redis.py — Redis cache persistence for local development.

TWO COMMANDS:

  python warm_redis.py dump
      Reads all exact-cache keys from Redis and saves them to cache_dump.json.
      Run this BEFORE shutting down Redis to preserve cached answers.

  python warm_redis.py warm
      Reads cache_dump.json and populates Redis with all saved entries.
      Run this AFTER starting Redis to restore the cache instantly.

WHY THIS EXISTS:
  Local Redis is in-memory — all data is lost on restart.
  MongoDB semantic cache survives (it's on disk / Atlas), so semantically
  similar queries will still hit MongoDB and auto-backfill Redis over time.
  But this script gives you instant restoration of exact-match cache
  without waiting for organic queries to rebuild it.

USAGE:
  # Before shutting down Redis:
  python warm_redis.py dump

  # After starting Redis again:
  python warm_redis.py warm

  # Automate: add to your startup script
  python warm_redis.py warm && uvicorn api:app --reload
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import redis
from loguru import logger

# ── Config — adjust to match your setup ───────────────────────────────────────
REDIS_URL = "redis://localhost:6379"
DUMP_FILE = Path(__file__).parent / "cache_dump.json"
KEY_PREFIX = "final_answer_cache:"          # must match CacheManager.EXACT_KEY_PREFIX
# ──────────────────────────────────────────────────────────────────────────────


def get_client() -> redis.Redis:
    client = redis.from_url(REDIS_URL, decode_responses=True)
    client.ping()
    return client


def dump():
    """Read all exact-cache entries from Redis → save to JSON file."""
    client = get_client()

    keys = client.keys(f"{KEY_PREFIX}*")
    if not keys:
        logger.info("No exact-cache keys found in Redis — nothing to dump.")
        return

    entries = []
    for key in keys:
        value = client.get(key)
        ttl = client.ttl(key)   # remaining TTL in seconds (-1 = no expiry, -2 = gone)

        if value and ttl > 0:
            entries.append({
                "key": key,
                "value": value,
                "ttl_remaining": ttl,
                "dumped_at": datetime.utcnow().isoformat(),
            })

    DUMP_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    logger.info(f"Dumped {len(entries)} cache entries → {DUMP_FILE}")


def warm():
    """Read JSON dump → populate Redis, skipping expired entries."""
    if not DUMP_FILE.exists():
        logger.warning(f"No dump file found at {DUMP_FILE}. Run 'dump' first.")
        return

    entries = json.loads(DUMP_FILE.read_text())
    if not entries:
        logger.info("Dump file is empty — nothing to restore.")
        return

    client = get_client()
    restored, skipped = 0, 0

    for entry in entries:
        ttl = entry.get("ttl_remaining", 0)

        # Skip entries that would have expired since the dump was taken
        if ttl <= 0:
            skipped += 1
            continue

        client.setex(entry["key"], ttl, entry["value"])
        restored += 1

    logger.info(
        f"Redis warmed: {restored} entries restored, {skipped} skipped (expired)."
    )
    if skipped > 0:
        logger.info(
            "Skipped entries will be rebuilt organically via MongoDB semantic cache hits."
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "dump":
        dump()
    elif command == "warm":
        warm()
    else:
        print(__doc__)
        sys.exit(1)