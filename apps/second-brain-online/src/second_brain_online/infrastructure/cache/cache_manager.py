import hashlib
import time
from datetime import datetime, timedelta

from loguru import logger
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings

from second_brain_online.config import settings
from .redis_client import get_redis_client


_CACHE_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
_cache_embedder: HuggingFaceEmbeddings | None = None


def get_cache_embedder(device: str = "cpu") -> HuggingFaceEmbeddings:
    global _cache_embedder
    if _cache_embedder is None:
        logger.info(f"Loading cache embedder: {_CACHE_EMBEDDING_MODEL_ID} on {device}")
        _cache_embedder = HuggingFaceEmbeddings(
            model_name=_CACHE_EMBEDDING_MODEL_ID,
            model_kwargs={
                "device": device,
                "backend": "onnx",
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Cache embedder loaded successfully")
    return _cache_embedder


class CacheManager:
    """
    Two-tier cache for FINAL AGENT ANSWERS.

    Tier 1 — Exact match (Redis):
        Key  : SHA256(query)
        Store: Redis string with TTL
        Speed: ~0.1ms, zero embedding cost
        TTL  : 7 days (personal project — answers don't change often)
        Persistence: dumped to JSON file, restored on Redis restart via warm_redis.py

    Tier 2 — Semantic match (MongoDB $vectorSearch):
        Key  : bge-small-en-v1.5 embedding (384 dims)
        Store: CACHE_MONGODB_URI / MONGODB_CACHE_DATABASE_NAME / semantic_cache
        Speed: ~5-15ms via ANN index
        TTL  : 30 days (persistent on disk, survives Redis restarts)
        Threshold: 0.95 cosine similarity

    Why both?
        Redis = speed (sub-ms, in-memory)
        MongoDB = durability (survives Redis restart, acts as natural warm-up source)
        After Redis restart → semantic query hits MongoDB at ~1.0 score → backfills Redis automatically

    Atlas Vector Search index on `semantic_cache`:
        {
          "fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine"
          }]
        }
        Index name: semantic_cache_index
    """

    SEMANTIC_INDEX_NAME = "semantic_cache_index"
    EXACT_KEY_PREFIX = "final_answer_cache:"

    # TTL constants — easy to adjust in one place
    REDIS_TTL_SECONDS = 7 * 24 * 3600       # 7 days
    MONGO_TTL_SECONDS = 30 * 24 * 3600      # 30 days

    def __init__(
        self,
        ttl_seconds: int | None = None,          # None = use class defaults above
        semantic_threshold: float = 0.92,
        device: str = "cpu",
    ):
        # Allow override but default to the longer TTLs defined above
        self.redis_ttl = ttl_seconds or self.REDIS_TTL_SECONDS
        self.mongo_ttl = ttl_seconds or self.MONGO_TTL_SECONDS
        self.semantic_threshold = semantic_threshold

        self._redis = get_redis_client()

        _mongo_client = MongoClient(settings.CACHE_MONGODB_URI)
        _db = _mongo_client[settings.MONGODB_CACHE_DATABASE_NAME]
        self._semantic_col = _db["semantic_cache"]
        self._ensure_mongo_indexes()

        self._embedder = get_cache_embedder(device)

    def _ensure_mongo_indexes(self) -> None:
        self._semantic_col.create_index(
            "expires_at",
            expireAfterSeconds=0,
            background=True,
        )

    def _mongo_expires_at(self) -> datetime:
        return datetime.utcnow() + timedelta(seconds=self.mongo_ttl)

    def embed_query(self, query: str) -> list[float]:
        return self._embedder.embed_query(query)

    # ─────────────────────────────────────────────
    # TIER 1 — Redis exact cache
    # ─────────────────────────────────────────────

    @property
    def _redis_enabled(self) -> bool:
        return self._redis is not None

    def _build_exact_key(self, query: str) -> str:
        return f"{self.EXACT_KEY_PREFIX}{hashlib.sha256(query.encode()).hexdigest()}"

    def get_exact(self, query: str) -> str | None:
        if not self._redis_enabled:
            return None
        try:
            t_start = time.perf_counter()
            result = self._redis.get(self._build_exact_key(query))
            t_end = time.perf_counter()
            logger.debug(f"[TIMING] Redis GET: {(t_end - t_start) * 1000:.1f}ms | hit={result is not None}")
            if result:
                logger.debug("Cache HIT [Tier 1 — exact / Redis]")
                return result
        except Exception as e:
            logger.warning(f"Redis GET failed: {e}")
        return None

    def set_exact(self, query: str, result: str) -> None:
        if not self._redis_enabled:
            return
        try:
            t_start = time.perf_counter()
            self._redis.setex(
                self._build_exact_key(query),
                self.redis_ttl,
                result,
            )
            t_end = time.perf_counter()
            logger.debug(f"[TIMING] Redis SET: {(t_end - t_start) * 1000:.1f}ms")
        except Exception as e:
            logger.warning(f"Redis SET failed: {e}")

    # ─────────────────────────────────────────────
    # TIER 2 — MongoDB semantic cache
    # ─────────────────────────────────────────────

    def get_semantic(self, query_embedding: list[float]) -> str | None:
        try:
            t_start = time.perf_counter()
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.SEMANTIC_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 1,
                    }
                },
                {
                    "$project": {
                        "result": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "_id": 0,
                    }
                },
            ]
            docs = list(self._semantic_col.aggregate(pipeline))
            t_end = time.perf_counter()
            logger.info(
                f"[TIMING] MongoDB semantic $vectorSearch: {(t_end - t_start) * 1000:.1f}ms "
                f"| hit={bool(docs and docs[0]['score'] >= self.semantic_threshold)}"
                + (f" | score={docs[0]['score']:.4f}" if docs else "")
            )
            if docs and docs[0]["score"] >= self.semantic_threshold:
                logger.debug(
                    f"Cache HIT [Tier 2 — semantic / MongoDB] score={docs[0]['score']:.4f}"
                )
                return docs[0]["result"]
        except Exception as e:
            logger.warning(f"MongoDB semantic GET failed: {e}")
        return None

    def set_semantic(self, query_embedding: list[float], result: str) -> None:
        try:
            t_start = time.perf_counter()
            self._semantic_col.insert_one({
                "embedding": query_embedding,
                "result": result,
                "expires_at": self._mongo_expires_at(),
            })
            t_end = time.perf_counter()
            logger.debug(f"[TIMING] MongoDB semantic SET: {(t_end - t_start) * 1000:.1f}ms")
        except Exception as e:
            logger.warning(f"MongoDB semantic SET failed: {e}")