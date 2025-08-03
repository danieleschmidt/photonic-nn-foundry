"""
Caching layer for photonic circuits and components.
"""

import json
import pickle
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    file_path: Path
    created_at: datetime
    accessed_at: datetime
    access_count: int
    size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'file_path': str(self.file_path),
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'size_bytes': self.size_bytes
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            key=data['key'],
            file_path=Path(data['file_path']),
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            access_count=data['access_count'],
            size_bytes=data['size_bytes']
        )


class BaseCache:
    """Base caching functionality."""
    
    def __init__(self, cache_dir: str, max_entries: int = 1000, 
                 max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.max_age = timedelta(hours=max_age_hours)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._lock = threading.Lock()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Clean up old entries
        self._cleanup_expired()
        
    def _load_metadata(self) -> Dict[str, CacheEntry]:
        """Load cache metadata from disk."""
        if not self.metadata_file.exists():
            return {}
            
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                
            metadata = {}
            for key, entry_data in data.items():
                metadata[key] = CacheEntry.from_dict(entry_data)
                
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return {}
            
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                key: entry.to_dict() 
                for key, entry in self.metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
            
        return hashlib.sha256(content.encode()).hexdigest()[:16]
        
    def _get_file_path(self, key: str, extension: str = ".pkl") -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}{extension}"
        
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        with self._lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.metadata.items():
                if current_time - entry.created_at > self.max_age:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                self._remove_entry(key)
                
            # Also remove entries if we exceed max count
            if len(self.metadata) > self.max_entries:
                # Sort by access count and remove least accessed
                sorted_entries = sorted(
                    self.metadata.items(),
                    key=lambda x: (x[1].access_count, x[1].accessed_at)
                )
                
                excess_count = len(self.metadata) - self.max_entries
                for key, _ in sorted_entries[:excess_count]:
                    self._remove_entry(key)
                    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        if key in self.metadata:
            entry = self.metadata[key]
            if entry.file_path.exists():
                try:
                    entry.file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {entry.file_path}: {e}")
                    
            del self.metadata[key]
            
    def _update_access(self, key: str):
        """Update access statistics for cache entry."""
        if key in self.metadata:
            entry = self.metadata[key]
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.metadata.values())
            
            return {
                'total_entries': len(self.metadata),
                'max_entries': self.max_entries,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
                'oldest_entry': min(
                    (entry.created_at for entry in self.metadata.values()),
                    default=None
                ),
                'newest_entry': max(
                    (entry.created_at for entry in self.metadata.values()),
                    default=None
                )
            }
            
    def clear_cache(self):
        """Clear all cache entries."""
        with self._lock:
            for key in list(self.metadata.keys()):
                self._remove_entry(key)
            self._save_metadata()
            
    def remove_expired(self):
        """Manually trigger cleanup of expired entries."""
        self._cleanup_expired()
        self._save_metadata()


class CircuitCache(BaseCache):
    """Cache for photonic circuits."""
    
    def __init__(self, cache_dir: str = ".cache/circuits", 
                 max_entries: int = 1000, max_age_hours: int = 24):
        super().__init__(cache_dir, max_entries, max_age_hours)
        
    def put_circuit(self, circuit_data: Dict[str, Any], 
                   verilog_code: Optional[str] = None,
                   metrics: Optional[Dict[str, Any]] = None) -> str:
        """Cache circuit data."""
        key = self._generate_key(circuit_data)
        
        with self._lock:
            cache_data = {
                'circuit_data': circuit_data,
                'verilog_code': verilog_code,
                'metrics': metrics,
                'cached_at': datetime.now().isoformat()
            }
            
            file_path = self._get_file_path(key, ".json")
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                    
                # Update metadata
                size_bytes = file_path.stat().st_size
                self.metadata[key] = CacheEntry(
                    key=key,
                    file_path=file_path,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes
                )
                
                self._save_metadata()
                logger.debug(f"Cached circuit with key: {key}")
                
            except Exception as e:
                logger.error(f"Failed to cache circuit: {e}")
                
        return key
        
    def get_circuit(self, circuit_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached circuit data."""
        key = self._generate_key(circuit_data)
        
        with self._lock:
            if key not in self.metadata:
                return None
                
            entry = self.metadata[key]
            if not entry.file_path.exists():
                self._remove_entry(key)
                return None
                
            try:
                with open(entry.file_path, 'r') as f:
                    cache_data = json.load(f)
                    
                self._update_access(key)
                self._save_metadata()
                
                logger.debug(f"Retrieved cached circuit with key: {key}")
                return cache_data
                
            except Exception as e:
                logger.error(f"Failed to read cached circuit: {e}")
                self._remove_entry(key)
                return None
                
    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached circuit by key."""
        with self._lock:
            if key not in self.metadata:
                return None
                
            entry = self.metadata[key]
            if not entry.file_path.exists():
                self._remove_entry(key)
                return None
                
            try:
                with open(entry.file_path, 'r') as f:
                    cache_data = json.load(f)
                    
                self._update_access(key)
                return cache_data
                
            except Exception as e:
                logger.error(f"Failed to read cached circuit: {e}")
                return None
                
    def list_cached_circuits(self) -> List[Dict[str, Any]]:
        """List all cached circuits with metadata."""
        with self._lock:
            circuits = []
            for key, entry in self.metadata.items():
                circuits.append({
                    'key': key,
                    'created_at': entry.created_at.isoformat(),
                    'accessed_at': entry.accessed_at.isoformat(),
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes
                })
            return circuits


class ComponentCache(BaseCache):
    """Cache for photonic components."""
    
    def __init__(self, cache_dir: str = ".cache/components",
                 max_entries: int = 500, max_age_hours: int = 48):
        super().__init__(cache_dir, max_entries, max_age_hours)
        
    def put_component(self, component_spec: Dict[str, Any]) -> str:
        """Cache component specification."""
        key = self._generate_key(component_spec)
        
        with self._lock:
            file_path = self._get_file_path(key, ".json")
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(component_spec, f, indent=2)
                    
                # Update metadata
                size_bytes = file_path.stat().st_size
                self.metadata[key] = CacheEntry(
                    key=key,
                    file_path=file_path,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes
                )
                
                self._save_metadata()
                logger.debug(f"Cached component with key: {key}")
                
            except Exception as e:
                logger.error(f"Failed to cache component: {e}")
                
        return key
        
    def get_component(self, component_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached component specification."""
        key = self._generate_key(component_spec)
        
        with self._lock:
            if key not in self.metadata:
                return None
                
            entry = self.metadata[key]
            if not entry.file_path.exists():
                self._remove_entry(key)
                return None
                
            try:
                with open(entry.file_path, 'r') as f:
                    component_data = json.load(f)
                    
                self._update_access(key)
                self._save_metadata()
                
                return component_data
                
            except Exception as e:
                logger.error(f"Failed to read cached component: {e}")
                self._remove_entry(key)
                return None


# Global cache instances
_circuit_cache: Optional[CircuitCache] = None
_component_cache: Optional[ComponentCache] = None


def get_circuit_cache() -> CircuitCache:
    """Get global circuit cache instance."""
    global _circuit_cache
    
    if _circuit_cache is None:
        cache_dir = os.getenv('CIRCUIT_CACHE_DIR', '.cache/circuits')
        max_entries = int(os.getenv('MAX_CACHED_CIRCUITS', '1000'))
        _circuit_cache = CircuitCache(cache_dir, max_entries)
        
    return _circuit_cache


def get_component_cache() -> ComponentCache:
    """Get global component cache instance."""
    global _component_cache
    
    if _component_cache is None:
        cache_dir = os.getenv('COMPONENT_CACHE_DIR', '.cache/components')
        _component_cache = ComponentCache(cache_dir)
        
    return _component_cache


def clear_all_caches():
    """Clear all cache instances."""
    global _circuit_cache, _component_cache
    
    if _circuit_cache:
        _circuit_cache.clear_cache()
        
    if _component_cache:
        _component_cache.clear_cache()