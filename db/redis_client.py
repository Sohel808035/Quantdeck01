"""
db/redis_client.py
──────────────────
High-Speed Redis Cache Layer for QuantSphereX V2.
Provides connection pooling, binary DataFrame serialization (MsgPack/Parquet),
key namespacing, and graceful in-memory fallback if Redis server is offline.
"""

from __future__ import annotations
import io
import json
import logging
import os
import threading
from typing import Optional, Any, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing redis optional dependency
try:
    import redis  # type: ignore
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class InMemoryFallbackCache:
    """In-memory dictionary cache used when Redis server is offline or uninstalled."""

    def __init__(self):
        self._store: Dict[str, bytes] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        with self._lock:
            self._store[key] = value
        return True

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._store.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
        return False

    def flush(self) -> None:
        with self._lock:
            self._store.clear()


class RedisCacheManager:
    """
    Manages Redis L1 cache connections for ultra-fast price & metadata retrieval.
    Falls back gracefully to memory cache if Redis is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        namespace: str = "quantspherex",
    ):
        self.namespace = namespace
        self.redis_client = None
        self.fallback = InMemoryFallbackCache()
        self.is_connected = False

        env_host = os.environ.get("REDIS_HOST", host)
        env_port = int(os.environ.get("REDIS_PORT", port))

        if _REDIS_AVAILABLE:
            try:
                client = redis.Redis(
                    host=env_host,
                    port=env_port,
                    db=db,
                    password=password,
                    socket_timeout=1.5,
                    socket_connect_timeout=1.5,
                )
                client.ping()
                self.redis_client = client
                self.is_connected = True
                logger.info(f"[Redis] Connected to Redis server at {env_host}:{env_port} (DB {db})")
            except Exception as exc:
                logger.debug(f"[Redis] Server unavailable ({exc}). Using in-memory fallback cache.")

    def _format_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def set(self, key: str, value: str | bytes, ttl: Optional[int] = 3600) -> bool:
        """Stores string or bytes in Redis with optional TTL in seconds."""
        full_key = self._format_key(key)
        data_bytes = value.encode("utf-8") if isinstance(value, str) else value

        if self.is_connected and self.redis_client:
            try:
                return bool(self.redis_client.set(full_key, data_bytes, ex=ttl))
            except Exception as exc:
                logger.warning(f"[Redis] Set error ({exc}). Routing to fallback cache.")

        return self.fallback.set(full_key, data_bytes, ttl)

    def get(self, key: str) -> Optional[bytes]:
        """Retrieves raw bytes from Redis or fallback cache."""
        full_key = self._format_key(key)

        if self.is_connected and self.redis_client:
            try:
                res = self.redis_client.get(full_key)
                if res is not None:
                    return res
            except Exception as exc:
                logger.warning(f"[Redis] Get error ({exc}). Querying fallback cache.")

        return self.fallback.get(full_key)

    def set_dataframe(self, key: str, df: pd.DataFrame, ttl: Optional[int] = 3600) -> bool:
        """Serializes DataFrame to binary Parquet in RAM and caches it."""
        if df.empty:
            return False
        buf = io.BytesIO()
        df.to_parquet(buf, compression="snappy")
        return self.set(key, buf.getvalue(), ttl=ttl)

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Deserializes DataFrame from RAM cache."""
        raw_bytes = self.get(key)
        if raw_bytes is None:
            return None
        buf = io.BytesIO(raw_bytes)
        return pd.read_parquet(buf)

    def delete(self, key: str) -> bool:
        """Deletes key from cache."""
        full_key = self._format_key(key)
        if self.is_connected and self.redis_client:
            try:
                return bool(self.redis_client.delete(full_key))
            except Exception:
                pass
        return self.fallback.delete(full_key)
