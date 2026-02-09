"""
Cache Manager for EDA Results
INTEGRATED: Works with existing FastAPI structure
Supports Redis with fallback to in-memory cache
"""

import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# ============================================================================
# REDIS CLIENT (Optional)
# ============================================================================

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available, using in-memory cache")

# ============================================================================
# CACHE MANAGER
# ============================================================================

class EDACacheManager:
    """
    EDA Cache Manager
    - Uses Redis if available
    - Falls back to in-memory cache
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize cache"""
        self.in_memory_cache: Dict[str, tuple] = {}  # (value, expiry_time)
        self.redis_client = None
        
        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {str(e)}")
                self.redis_client = None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set cache value with TTL
        
        Args:
            key: Cache key
            value: Value to cache (dict, list, or string)
            ttl: Time to live in seconds (default: 1 hour)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert dict/list to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            # Try Redis first
            if self.redis_client:
                self.redis_client.setex(key, ttl, value)
                logger.debug(f"✅ Redis SET: {key} (TTL: {ttl}s)")
                return True
            
            # Fallback to in-memory cache
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.in_memory_cache[key] = (value, expiry)
            logger.debug(f"✅ Memory SET: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache SET failed: {key} - {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get cache value
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    logger.debug(f"✅ Redis HIT: {key}")
                    return value
                logger.debug(f"⏭️ Redis MISS: {key}")
                return None
            
            # Fallback to in-memory cache
            if key not in self.in_memory_cache:
                logger.debug(f"⏭️ Memory MISS: {key}")
                return None
            
            value, expiry = self.in_memory_cache[key]
            
            # Check if expired
            if datetime.utcnow() > expiry:
                logger.debug(f"⏰ Memory EXPIRED: {key}")
                del self.in_memory_cache[key]
                return None
            
            logger.debug(f"✅ Memory HIT: {key}")
            return value
            
        except Exception as e:
            logger.error(f"❌ Cache GET failed: {key} - {str(e)}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
                logger.debug(f"🗑️ Redis DELETE: {key}")
                return True
            
            if key in self.in_memory_cache:
                del self.in_memory_cache[key]
                logger.debug(f"🗑️ Memory DELETE: {key}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Cache DELETE failed: {key} - {str(e)}")
            return False
    
    async def ping(self) -> bool:
        """Check cache connectivity"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            # In-memory cache is always available
            return True
        except:
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
                logger.info("🗑️ Redis cache cleared")
            self.in_memory_cache.clear()
            logger.info("🗑️ Memory cache cleared")
            return True
        except Exception as e:
            logger.error(f"❌ Cache CLEAR failed: {str(e)}")
            return False

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

cache_manager = EDACacheManager()
