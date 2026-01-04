import os
import json
import logging
import ssl
import socket
from typing import Optional, Any, Dict, List
from datetime import datetime, date
from decimal import Decimal
import redis
from redis.connection import ConnectionPool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
REDIS_HOST = os.getenv("AZURE_REDIS_HOST")
REDIS_PORT = int(os.getenv("AZURE_REDIS_PORT", "6380"))
REDIS_PASSWORD = os.getenv("AZURE_REDIS_PASSWORD")
REDIS_SSL = os.getenv("AZURE_REDIS_SSL", "true").lower() == "true"

# Default TTL values (in seconds)
DEFAULT_TTL = 3600  # 1 hour
ANALYTICS_TTL = 1800  # 30 minutes
FILE_LIST_TTL = 300  # 5 minutes
USER_TTL = 7200  # 2 hours
FORECAST_TTL = 3600  # 1 hour

# Connection settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CONNECT_TIMEOUT = 15  # increased for Azure SSL
SOCKET_TIMEOUT = 10

if not REDIS_HOST or not REDIS_PASSWORD:
    logger.warning("Redis credentials not found. Cache will be disabled.")
    REDIS_ENABLED = False
else:
    REDIS_ENABLED = True


class CustomJSONEncoder(json.JSONEncoder):
    """Handle non-serializable types common in analytics"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        # Handle numpy types if numpy is available
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


class RedisCache:
    """Azure Redis Cache manager with automatic fallback and retry logic"""

    def __init__(self):
        self.client = None
        self.pool = None
        self.enabled = REDIS_ENABLED

        if self.enabled:
            self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize Redis connection with retry logic"""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Build keepalive options (Linux/Unix only)
                keepalive_opts = None
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    keepalive_opts = {
                        socket.TCP_KEEPIDLE: 60,
                        socket.TCP_KEEPINTVL: 10,
                        socket.TCP_KEEPCNT: 3
                    }

                # Redis 7.x SSL configuration - different API than 4.x/5.x
                connection_kwargs = {
                    "host": REDIS_HOST,
                    "port": REDIS_PORT,
                    "password": REDIS_PASSWORD,
                    "decode_responses": True,
                    "socket_connect_timeout": CONNECT_TIMEOUT,
                    "socket_timeout": SOCKET_TIMEOUT,
                    "socket_keepalive": True,
                    "socket_keepalive_options": keepalive_opts,
                    "retry_on_timeout": True,
                    "retry_on_error": [redis.ConnectionError, redis.TimeoutError],
                    "retry": redis.retry.Retry(redis.backoff.ExponentialBackoff(), 3),
                    "health_check_interval": 30,
                    "max_connections": 50,
                    "encoding": "utf-8"
                }

                # Add SSL configuration for Azure Redis
                if REDIS_SSL:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                    
                    # Redis 7.x uses ssl_context parameter instead of ssl
                    connection_kwargs["ssl"] = True
                    connection_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
                    connection_kwargs["ssl_check_hostname"] = False

                # Create connection pool
                self.pool = ConnectionPool(**connection_kwargs)
                self.client = redis.Redis(connection_pool=self.pool)
                
                # Test connection with timeout
                self.client.ping()
                logger.info(f"✓ Redis connected: {REDIS_HOST} (attempt {attempt})")
                return

            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis connection attempt {attempt}/{MAX_RETRIES} failed: {str(e)}")
                if attempt < MAX_RETRIES:
                    import time
                    time.sleep(RETRY_DELAY * attempt)  # Exponential backoff
                else:
                    logger.error(f"Redis connection failed after {MAX_RETRIES} attempts. Disabling cache.")
                    self.enabled = False
                    self.client = None
                    self.pool = None

            except Exception as e:
                logger.error(f"Unexpected Redis error: {str(e)}")
                self.enabled = False
                self.client = None
                self.pool = None
                break

    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)

    def _serialize(self, value: Any) -> str:
        """Serialize value with custom encoder"""
        return json.dumps(value, cls=CustomJSONEncoder)

    def _deserialize(self, value: str) -> Any:
        """Deserialize value"""
        return json.loads(value)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling"""
        if not self.enabled or not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return self._deserialize(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis connection error for GET {key}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {key}: {str(e)}")
            self.delete(key)  # Remove corrupted data
            return None
        except Exception as e:
            logger.error(f"Redis GET error for {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """Set value in cache with TTL"""
        if not self.enabled or not self.client:
            return False

        try:
            serialized = self._serialize(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis connection error for SET {key}: {str(e)}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error for {key}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Redis SET error for {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return False

        try:
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for {key}: {str(e)}")
            return False

    def delete_pattern(self, pattern: str, batch_size: int = 100) -> int:
        """
        Delete all keys matching pattern using SCAN (non-blocking)
        
        CRITICAL FIX: Uses SCAN instead of KEYS to avoid blocking Redis
        """
        if not self.enabled or not self.client:
            return 0

        try:
            count = 0
            cursor = 0
            
            while True:
                # Use SCAN for non-blocking iteration
                cursor, keys = self.client.scan(
                    cursor=cursor, 
                    match=pattern, 
                    count=batch_size
                )
                
                if keys:
                    # Delete in batches using pipeline
                    pipe = self.client.pipeline()
                    for key in keys:
                        pipe.delete(key)
                    pipe.execute()
                    count += len(keys)
                
                # cursor returns to 0 when iteration is complete
                if cursor == 0:
                    break
            
            logger.debug(f"Cache DELETE pattern {pattern}: {count} keys")
            return count
            
        except Exception as e:
            logger.error(f"Redis DELETE pattern error for {pattern}: {str(e)}")
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.enabled or not self.client:
            return False

        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for {key}: {str(e)}")
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        if not self.enabled or not self.client:
            return None

        try:
            ttl = self.client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Redis TTL error for {key}: {str(e)}")
            return None

    def increment(self, key: str, amount: int = 1, ttl: int = DEFAULT_TTL) -> Optional[int]:
        """Increment counter atomically"""
        if not self.enabled or not self.client:
            return None

        try:
            # Use pipeline for atomic operation
            pipe = self.client.pipeline()
            pipe.incr(key, amount)
            pipe.expire(key, ttl)
            results = pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Redis INCR error for {key}: {str(e)}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False, "status": "disabled"}

        try:
            info = self.client.info()
            return {
                "enabled": True,
                "status": "connected",
                "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "used_memory_peak_mb": round(info.get("used_memory_peak", 0) / (1024 * 1024), 2),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {str(e)}")
            return {"enabled": True, "status": "error", "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        return round((hits / total * 100) if total > 0 else 0, 2)

    def flush_all(self) -> bool:
        """Clear entire cache (use with caution!)"""
        if not self.enabled or not self.client:
            return False

        try:
            self.client.flushdb()
            logger.warning("Cache FLUSHED: All keys deleted")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSH error: {str(e)}")
            return False

    def close(self) -> None:
        """Properly close Redis connection and pool"""
        if self.client:
            try:
                self.client.close()
            except:
                pass
        if self.pool:
            try:
                self.pool.disconnect()
            except:
                pass
        logger.info("Redis connection closed")


# ========== GLOBAL INSTANCE ==========
cache = RedisCache()


# ========== HELPER FUNCTIONS ==========

def cache_analytics(user_id: str, blob_name: str, analytics_data: Dict) -> bool:
    """Cache analytics results"""
    key = cache._generate_key("analytics", user_id, blob_name)
    return cache.set(key, analytics_data, ttl=ANALYTICS_TTL)


def get_cached_analytics(user_id: str, blob_name: str) -> Optional[Dict]:
    """Get cached analytics results"""
    key = cache._generate_key("analytics", user_id, blob_name)
    return cache.get(key)


def invalidate_analytics(user_id: str, blob_name: str = None) -> int:
    """Invalidate analytics cache for user or specific file"""
    if blob_name:
        key = cache._generate_key("analytics", user_id, blob_name)
        return 1 if cache.delete(key) else 0
    else:
        pattern = cache._generate_key("analytics", user_id, "*")
        return cache.delete_pattern(pattern)


def cache_file_list(user_id: str, file_list: list) -> bool:
    """Cache user's file list"""
    key = cache._generate_key("files", user_id)
    return cache.set(key, file_list, ttl=FILE_LIST_TTL)


def get_cached_file_list(user_id: str) -> Optional[list]:
    """Get cached file list"""
    key = cache._generate_key("files", user_id)
    return cache.get(key)


def invalidate_file_list(user_id: str) -> bool:
    """Invalidate file list cache"""
    key = cache._generate_key("files", user_id)
    return cache.delete(key)


def cache_forecast(user_id: str, blob_name: str, periods: int, forecast_data: Dict) -> bool:
    """Cache forecast results"""
    key = cache._generate_key("forecast", user_id, blob_name, periods)
    return cache.set(key, forecast_data, ttl=FORECAST_TTL)


def get_cached_forecast(user_id: str, blob_name: str, periods: int) -> Optional[Dict]:
    """Get cached forecast results"""
    key = cache._generate_key("forecast", user_id, blob_name, periods)
    return cache.get(key)


def invalidate_forecast(user_id: str, blob_name: str = None) -> int:
    """Invalidate forecast cache"""
    if blob_name:
        pattern = cache._generate_key("forecast", user_id, blob_name, "*")
    else:
        pattern = cache._generate_key("forecast", user_id, "*")
    return cache.delete_pattern(pattern)


def cache_user(user_id: str, user_data: Dict) -> bool:
    """Cache user data"""
    key = cache._generate_key("user", user_id)
    return cache.set(key, user_data, ttl=USER_TTL)


def get_cached_user(user_id: str) -> Optional[Dict]:
    """Get cached user data"""
    key = cache._generate_key("user", user_id)
    return cache.get(key)


def invalidate_user(user_id: str) -> bool:
    """Invalidate user cache"""
    key = cache._generate_key("user", user_id)
    return cache.delete(key)


def track_api_usage(user_id: str, endpoint: str) -> Optional[int]:
    """Track API usage with rate limiting"""
    key = cache._generate_key("api_usage", user_id, endpoint)
    return cache.increment(key, ttl=3600)


def get_api_usage(user_id: str, endpoint: str) -> int:
    """Get current API usage count"""
    key = cache._generate_key("api_usage", user_id, endpoint)
    value = cache.get(key)
    return int(value) if value else 0
