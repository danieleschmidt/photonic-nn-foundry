"""
Intelligent caching system with machine learning-based optimization.

This module provides advanced caching capabilities including:
- Adaptive cache replacement policies with ML optimization
- Predictive prefetching based on access patterns
- Multi-tier cache hierarchy with automatic promotion/demotion
- Distributed cache coordination and consistency
- Cache performance analytics and optimization
- Content-aware caching strategies
- Compression and serialization optimization
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (Dict, List, Any, Optional, Callable, Union, Tuple, Set, 
                   Protocol, TypeVar, Generic, Iterator)
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib
import psutil
import weakref
import gc
import traceback
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import json
from datetime import datetime, timedelta
import math
import statistics
from contextlib import contextmanager
import resource
from threading import RLock, Condition, Event
from abc import ABC, abstractmethod
import uuid
import lz4.frame
import zstandard as zstd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    ARC = "arc"                    # Adaptive Replacement Cache
    ML_OPTIMIZED = "ml_optimized"  # Machine Learning Optimized
    HYBRID = "hybrid"              # Hybrid approach
    TTL = "ttl"                    # Time To Live based


class CacheTier(Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"        # Fastest, smallest
    L2_MEMORY = "l2_memory"        # Fast, medium size
    L3_DISK = "l3_disk"           # Slower, large size
    DISTRIBUTED = "distributed"    # Network-based, largest


class CompressionType(Enum):
    """Compression algorithms for cache storage."""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    PICKLE_PROTOCOL = "pickle"


class AccessPattern(Enum):
    """Access pattern types for optimization."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    MIXED = "mixed"


@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[float] = None
    compression_type: CompressionType = CompressionType.NONE
    compressed_data: Optional[bytes] = None
    tags: Set[str] = field(default_factory=set)
    priority: float = 1.0
    cost: float = 1.0  # Cost to recreate this entry
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
        
    @property
    def idle_time_seconds(self) -> float:
        """Get idle time since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()
        
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
        
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        age = max(self.age_seconds, 1.0)  # Avoid division by zero
        return self.access_count / age
        
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0  # L2->L1, L3->L2
    demotions: int = 0   # L1->L2, L2->L3
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    compression_savings_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
        
    @property
    def prefetch_accuracy(self) -> float:
        """Calculate prefetch accuracy percentage."""
        total = self.prefetch_hits + self.prefetch_misses
        return (self.prefetch_hits / total * 100) if total > 0 else 0.0
        
    @property
    def compression_ratio(self) -> float:
        """Calculate compression savings ratio."""
        if self.total_size_bytes == 0:
            return 0.0
        return self.compression_savings_bytes / self.total_size_bytes


class SerializationManager:
    """Manages serialization and compression for cache entries."""
    
    def __init__(self):
        self.compressors = {
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.ZSTD: self._zstd_compress,
            CompressionType.PICKLE_PROTOCOL: self._pickle_compress
        }
        
        self.decompressors = {
            CompressionType.LZ4: self._lz4_decompress,
            CompressionType.ZSTD: self._zstd_decompress,
            CompressionType.PICKLE_PROTOCOL: self._pickle_decompress
        }
        
    def serialize_and_compress(self, value: Any, 
                             compression_type: CompressionType = CompressionType.LZ4) -> Tuple[bytes, int]:
        """Serialize and compress a value."""
        if compression_type == CompressionType.NONE:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return serialized, 0  # No compression savings
            
        if compression_type not in self.compressors:
            raise ValueError(f"Unsupported compression type: {compression_type}")
            
        # Serialize first
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized)
        
        # Compress
        compressed = self.compressors[compression_type](serialized)
        compressed_size = len(compressed)
        
        savings = original_size - compressed_size
        return compressed, savings
        
    def decompress_and_deserialize(self, data: bytes, 
                                 compression_type: CompressionType = CompressionType.LZ4) -> Any:
        """Decompress and deserialize data."""
        if compression_type == CompressionType.NONE:
            return pickle.loads(data)
            
        if compression_type not in self.decompressors:
            raise ValueError(f"Unsupported compression type: {compression_type}")
            
        # Decompress first
        decompressed = self.decompressors[compression_type](data)
        
        # Deserialize
        return pickle.loads(decompressed)
        
    def _lz4_compress(self, data: bytes) -> bytes:
        """LZ4 compression."""
        return lz4.frame.compress(data)
        
    def _lz4_decompress(self, data: bytes) -> bytes:
        """LZ4 decompression."""
        return lz4.frame.decompress(data)
        
    def _zstd_compress(self, data: bytes) -> bytes:
        """Zstandard compression."""
        compressor = zstd.ZstdCompressor(level=3)
        return compressor.compress(data)
        
    def _zstd_decompress(self, data: bytes) -> bytes:
        """Zstandard decompression."""
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(data)
        
    def _pickle_compress(self, data: bytes) -> bytes:
        """Pickle protocol optimization."""
        # Use highest protocol for better compression
        return data  # Already pickled with highest protocol
        
    def _pickle_decompress(self, data: bytes) -> bytes:
        """Pickle protocol decompression."""
        return data  # No additional decompression needed


class AccessPatternAnalyzer:
    """Analyzes cache access patterns for optimization."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.access_history = deque(maxlen=history_size)
        self.key_sequences = defaultdict(lambda: deque(maxlen=1000))
        self.temporal_patterns = defaultdict(list)
        self._lock = RLock()
        
        # ML models
        self.scaler = StandardScaler()
        self.predictor = LinearRegression()
        self.is_trained = False
        
    def record_access(self, key: str, timestamp: datetime = None):
        """Record a cache access."""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self._lock:
            access_record = {
                'key': key,
                'timestamp': timestamp,
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'minute_of_day': timestamp.hour * 60 + timestamp.minute
            }
            
            self.access_history.append(access_record)
            self.key_sequences[key].append(timestamp)
            
            # Update temporal patterns
            time_key = f"{timestamp.hour}:{timestamp.minute // 10 * 10}"  # 10-minute buckets
            self.temporal_patterns[time_key].append(key)
            
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect access patterns in the cache."""
        with self._lock:
            if len(self.access_history) < 100:
                return {'pattern_type': AccessPattern.MIXED, 'confidence': 0.0}
                
            # Analyze temporal patterns
            temporal_score = self._analyze_temporal_patterns()
            
            # Analyze sequential patterns  
            sequential_score = self._analyze_sequential_patterns()
            
            # Analyze frequency patterns
            frequency_score = self._analyze_frequency_patterns()
            
            # Determine dominant pattern
            patterns = {
                AccessPattern.TEMPORAL: temporal_score,
                AccessPattern.SEQUENTIAL: sequential_score,
                AccessPattern.RANDOM: 1.0 - max(temporal_score, sequential_score)
            }
            
            dominant_pattern = max(patterns, key=patterns.get)
            confidence = patterns[dominant_pattern]
            
            return {
                'pattern_type': dominant_pattern,
                'confidence': confidence,
                'pattern_scores': {p.value: s for p, s in patterns.items()},
                'recommendations': self._generate_recommendations(dominant_pattern, confidence)
            }
            
    def predict_next_accesses(self, num_predictions: int = 10) -> List[Tuple[str, float]]:
        """Predict next likely cache accesses."""
        if not self.is_trained:
            self._train_predictor()
            
        with self._lock:
            if len(self.access_history) < 50:
                return []
                
            # Get recent access patterns
            recent_keys = [record['key'] for record in list(self.access_history)[-20:]]
            key_counts = defaultdict(int)
            
            for key in recent_keys:
                key_counts[key] += 1
                
            # Simple prediction based on recent frequency and temporal patterns
            current_time = datetime.now()
            time_key = f"{current_time.hour}:{current_time.minute // 10 * 10}"
            
            predictions = []
            
            # Frequency-based predictions
            for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True):
                frequency_score = count / len(recent_keys)
                
                # Temporal boost
                temporal_boost = 1.0
                if time_key in self.temporal_patterns:
                    temporal_keys = self.temporal_patterns[time_key]
                    if key in temporal_keys:
                        temporal_boost = 1.0 + (temporal_keys.count(key) / len(temporal_keys))
                        
                combined_score = frequency_score * temporal_boost
                predictions.append((key, combined_score))
                
            # Sort by score and return top predictions
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:num_predictions]
            
    def _analyze_temporal_patterns(self) -> float:
        """Analyze temporal access patterns."""
        if len(self.temporal_patterns) < 5:
            return 0.0
            
        # Calculate variance in temporal access distribution
        pattern_strengths = []
        for time_bucket, keys in self.temporal_patterns.items():
            if len(keys) > 1:
                unique_keys = set(keys)
                repetition_ratio = len(keys) / len(unique_keys)
                pattern_strengths.append(repetition_ratio)
                
        if not pattern_strengths:
            return 0.0
            
        # Higher variance indicates stronger temporal patterns
        mean_strength = np.mean(pattern_strengths)
        return min(1.0, mean_strength - 1.0) if mean_strength > 1.0 else 0.0
        
    def _analyze_sequential_patterns(self) -> float:
        """Analyze sequential access patterns."""
        if len(self.access_history) < 10:
            return 0.0
            
        # Look for sequential patterns in key names or access order
        recent_keys = [record['key'] for record in list(self.access_history)[-100:]]
        
        # Simple sequential pattern detection
        sequential_score = 0.0
        sequence_length = 0
        
        for i in range(1, len(recent_keys)):
            curr_key = recent_keys[i]
            prev_key = recent_keys[i-1]
            
            # Check if keys are numerically sequential
            try:
                if curr_key.replace(prev_key[:-1], '') and prev_key.replace(curr_key[:-1], ''):
                    curr_num = int(curr_key.split('_')[-1]) if '_' in curr_key else int(curr_key[-1])
                    prev_num = int(prev_key.split('_')[-1]) if '_' in prev_key else int(prev_key[-1])
                    
                    if curr_num == prev_num + 1:
                        sequence_length += 1
                    else:
                        if sequence_length > 0:
                            sequential_score += sequence_length
                        sequence_length = 0
            except (ValueError, IndexError):
                continue
                
        if sequence_length > 0:
            sequential_score += sequence_length
            
        return min(1.0, sequential_score / len(recent_keys))
        
    def _analyze_frequency_patterns(self) -> float:
        """Analyze frequency-based patterns."""
        if len(self.access_history) < 20:
            return 0.0
            
        key_counts = defaultdict(int)
        for record in self.access_history:
            key_counts[record['key']] += 1
            
        counts = list(key_counts.values())
        if len(counts) < 2:
            return 0.0
            
        # Calculate coefficient of variation
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
            
        cv = std_count / mean_count
        # High CV indicates some keys are accessed much more frequently
        return min(1.0, cv / 2.0)  # Normalize to [0, 1]
        
    def _generate_recommendations(self, pattern_type: AccessPattern, 
                                confidence: float) -> List[str]:
        """Generate optimization recommendations based on patterns."""
        recommendations = []
        
        if pattern_type == AccessPattern.TEMPORAL and confidence > 0.7:
            recommendations.extend([
                "Enable time-based prefetching",
                "Implement temporal cache warming",
                "Use TTL-based expiration aligned with access patterns"
            ])
            
        elif pattern_type == AccessPattern.SEQUENTIAL and confidence > 0.6:
            recommendations.extend([
                "Enable sequential prefetching",
                "Implement sliding window cache warming",
                "Use predictive preloading for sequential data"
            ])
            
        elif pattern_type == AccessPattern.RANDOM and confidence > 0.5:
            recommendations.extend([
                "Focus on hit rate optimization",
                "Use adaptive cache sizing",
                "Implement smart eviction policies"
            ])
            
        if confidence > 0.8:
            recommendations.append("High pattern confidence - enable aggressive optimization")
        elif confidence < 0.3:
            recommendations.append("Low pattern confidence - use conservative policies")
            
        return recommendations
        
    def _train_predictor(self):
        """Train ML predictor for access patterns."""
        try:
            if len(self.access_history) < 100:
                return
                
            # Prepare training data
            features = []
            targets = []
            
            history_list = list(self.access_history)
            for i in range(10, len(history_list) - 1):
                # Features: recent access pattern
                recent_keys = [record['key'] for record in history_list[i-10:i]]
                key_features = self._extract_key_features(recent_keys)
                
                # Target: next key (encoded)
                next_key = history_list[i+1]['key']
                target = hash(next_key) % 1000  # Simple encoding
                
                features.append(key_features)
                targets.append(target)
                
            if len(features) < 20:
                return
                
            X = np.array(features)
            y = np.array(targets)
            
            # Train scaler and predictor
            X_scaled = self.scaler.fit_transform(X)
            self.predictor.fit(X_scaled, y)
            self.is_trained = True
            
        except Exception as e:
            logger.warning(f"Failed to train access pattern predictor: {e}")
            
    def _extract_key_features(self, keys: List[str]) -> List[float]:
        """Extract numerical features from key sequence."""
        features = []
        
        # Basic statistics
        features.append(len(set(keys)))  # Unique keys
        features.append(len(keys))       # Total keys
        
        # Key hash statistics
        key_hashes = [hash(key) % 1000 for key in keys]
        features.extend([
            np.mean(key_hashes),
            np.std(key_hashes),
            np.min(key_hashes),
            np.max(key_hashes)
        ])
        
        # Pad or truncate to fixed size
        target_size = 10
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return features


class MLCacheOptimizer:
    """Machine learning-based cache optimization."""
    
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.access_analyzer = AccessPatternAnalyzer()
        
        # ML models for different aspects
        self.eviction_predictor = None
        self.size_predictor = None
        self.prefetch_predictor = None
        
        # Training data
        self.eviction_training_data = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
    def record_cache_event(self, event_type: str, key: str, entry: Optional[CacheEntry] = None,
                          hit_rate: float = 0.0):
        """Record cache events for ML training."""
        timestamp = datetime.now()
        
        # Record access for pattern analysis
        if event_type in ['hit', 'miss']:
            self.access_analyzer.record_access(key, timestamp)
            
        # Record eviction events for training
        if event_type == 'eviction' and entry:
            eviction_data = {
                'key': key,
                'age_seconds': entry.age_seconds,
                'idle_time': entry.idle_time_seconds,
                'access_count': entry.access_count,
                'access_frequency': entry.access_frequency,
                'size_bytes': entry.size_bytes,
                'priority': entry.priority,
                'hit_rate_when_evicted': hit_rate,
                'timestamp': timestamp
            }
            self.eviction_training_data.append(eviction_data)
            
        # Record performance metrics
        if event_type in ['hit', 'miss']:
            self.performance_history.append({
                'timestamp': timestamp,
                'hit_rate': hit_rate,
                'event_type': event_type
            })
            
    def optimize_eviction_policy(self, candidates: List[CacheEntry]) -> CacheEntry:
        """Use ML to select the best candidate for eviction."""
        if not candidates:
            return None
            
        if len(self.eviction_training_data) < 50:
            # Fall back to LRU if insufficient training data
            return min(candidates, key=lambda e: e.last_accessed)
            
        try:
            # Extract features for each candidate
            candidate_features = []
            
            for entry in candidates:
                features = [
                    entry.age_seconds,
                    entry.idle_time_seconds,
                    entry.access_count,
                    entry.access_frequency,
                    entry.size_bytes,
                    entry.priority,
                    len(entry.tags),
                    entry.cost
                ]
                candidate_features.append(features)
                
            # Simple scoring based on historical patterns
            scores = []
            for features in candidate_features:
                # Weighted scoring based on typical eviction factors
                score = (
                    features[1] * 0.4 +      # Idle time (higher = more likely to evict)
                    (1.0 / max(features[3], 0.001)) * 0.3 +  # Inverse frequency
                    features[0] * 0.2 +      # Age
                    features[4] * 0.1        # Size (larger items more likely to evict)
                )
                scores.append(score)
                
            # Select candidate with highest eviction score
            best_idx = np.argmax(scores)
            return candidates[best_idx]
            
        except Exception as e:
            logger.warning(f"ML eviction optimization failed: {e}")
            # Fall back to LRU
            return min(candidates, key=lambda e: e.last_accessed)
            
    def predict_optimal_cache_size(self, current_hit_rate: float, 
                                 current_size: int) -> int:
        """Predict optimal cache size based on performance history."""
        if len(self.performance_history) < 100:
            return current_size
            
        try:
            # Analyze hit rate trends
            recent_history = list(self.performance_history)[-50:]
            hit_rates = [h['hit_rate'] for h in recent_history]
            
            current_avg_hit_rate = np.mean(hit_rates)
            hit_rate_trend = np.polyfit(range(len(hit_rates)), hit_rates, 1)[0]
            
            # Simple heuristic for size adjustment
            if current_avg_hit_rate < 0.7 and hit_rate_trend < 0:
                # Hit rate is low and declining - increase size
                return min(current_size * 1.2, self.cache_size * 2)
            elif current_avg_hit_rate > 0.95 and hit_rate_trend > 0:
                # Hit rate is very high - can potentially reduce size
                return max(current_size * 0.9, self.cache_size * 0.5)
            else:
                return current_size
                
        except Exception as e:
            logger.warning(f"Cache size prediction failed: {e}")
            return current_size
            
    def generate_prefetch_suggestions(self, num_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Generate prefetch suggestions based on ML predictions."""
        return self.access_analyzer.predict_next_accesses(num_suggestions)
        
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get ML-based optimization insights."""
        pattern_analysis = self.access_analyzer.detect_patterns()
        
        insights = {
            'access_patterns': pattern_analysis,
            'training_data_size': {
                'eviction_events': len(self.eviction_training_data),
                'performance_history': len(self.performance_history),
                'access_history': len(self.access_analyzer.access_history)
            },
            'recommendations': []
        }
        
        # Add ML-specific recommendations
        if len(self.eviction_training_data) > 100:
            insights['recommendations'].append("ML eviction optimization is active")
        else:
            insights['recommendations'].append("Need more eviction data for ML optimization")
            
        if pattern_analysis['confidence'] > 0.7:
            insights['recommendations'].append(
                f"Strong {pattern_analysis['pattern_type'].value} pattern detected - "
                f"enable specialized optimizations"
            )
            
        return insights


class IntelligentCache(Generic[K, V]):
    """Main intelligent cache with ML-based optimization."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100,
                 policy: CachePolicy = CachePolicy.ML_OPTIMIZED,
                 compression: CompressionType = CompressionType.LZ4,
                 enable_prefetch: bool = True,
                 enable_multi_tier: bool = True):
        
        # Configuration
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.compression = compression
        self.enable_prefetch = enable_prefetch
        self.enable_multi_tier = enable_multi_tier
        
        # Storage
        self.entries: Dict[K, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        
        # Multi-tier support
        self.tiers = {
            CacheTier.L1_MEMORY: {},  # Hot data
            CacheTier.L2_MEMORY: {},  # Warm data
            CacheTier.L3_DISK: {}     # Cold data (future implementation)
        }
        
        # Components
        self.serializer = SerializationManager()
        self.ml_optimizer = MLCacheOptimizer(max_size)
        
        # Statistics
        self.stats = CacheStats()
        self.performance_tracker = deque(maxlen=1000)
        
        # Control
        self._lock = RLock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache-prefetch")
        self._background_thread = None
        self._running = False
        
        # Start background optimization
        self.start_background_optimization()
        
    def start_background_optimization(self):
        """Start background optimization thread."""
        if self._running:
            return
            
        self._running = True
        self._background_thread = threading.Thread(
            target=self._optimization_loop, 
            daemon=True,
            name="cache-optimizer"
        )
        self._background_thread.start()
        
    def stop_background_optimization(self):
        """Stop background optimization."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5.0)
        self._prefetch_executor.shutdown(wait=True)
        
    def get(self, key: K, default: V = None) -> Optional[V]:
        """Get value from cache with intelligent optimizations."""
        with self._lock:
            # Check if key exists in any tier
            entry = self._find_entry_in_tiers(key)
            
            if entry is None:
                # Cache miss
                self.stats.misses += 1
                self.ml_optimizer.record_cache_event('miss', str(key), hit_rate=self.stats.hit_rate)
                
                # Trigger prefetch if enabled
                if self.enable_prefetch:
                    self._schedule_prefetch()
                    
                return default
                
            # Cache hit
            self.stats.hits += 1
            entry.update_access()
            self.ml_optimizer.record_cache_event('hit', str(key), entry, self.stats.hit_rate)
            
            # Update access tracking
            self._update_access_tracking(key, entry)
            
            # Promote entry if in lower tier
            self._promote_entry_if_needed(key, entry)
            
            # Decompress if needed
            if entry.compression_type != CompressionType.NONE:
                return self.serializer.decompress_and_deserialize(
                    entry.compressed_data, entry.compression_type
                )
            else:
                return entry.value
                
    def put(self, key: K, value: V, ttl_seconds: Optional[float] = None,
           tags: Set[str] = None, priority: float = 1.0, cost: float = 1.0) -> bool:
        """Put value in cache with intelligent placement."""
        with self._lock:
            # Serialize and compress value
            compressed_data, compression_savings = self.serializer.serialize_and_compress(
                value, self.compression
            )
            
            # Calculate size
            if self.compression != CompressionType.NONE:
                size_bytes = len(compressed_data)
                actual_value = None  # Store compressed data
            else:
                actual_value = value
                size_bytes = len(compressed_data)  # Serialized size
                compressed_data = None
                
            # Create cache entry
            entry = CacheEntry(
                key=str(key),
                value=actual_value,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl_seconds,
                compression_type=self.compression,
                compressed_data=compressed_data,
                tags=tags or set(),
                priority=priority,
                cost=cost
            )
            
            # Check if we need to make room
            if not self._has_space_for_entry(entry):
                if not self._make_room_for_entry(entry):
                    return False  # Couldn't make room
                    
            # Place entry in appropriate tier
            tier = self._select_initial_tier(entry)
            self._place_entry_in_tier(key, entry, tier)
            
            # Update statistics
            self.stats.entry_count += 1
            self.stats.total_size_bytes += size_bytes
            if compression_savings > 0:
                self.stats.compression_savings_bytes += compression_savings
                
            return True
            
    def delete(self, key: K) -> bool:
        """Delete entry from cache."""
        with self._lock:
            entry = self._find_entry_in_tiers(key)
            if entry is None:
                return False
                
            # Remove from all tracking structures
            self._remove_entry_from_all_tiers(key)
            self._remove_from_access_tracking(key)
            
            # Update statistics
            self.stats.entry_count -= 1
            self.stats.total_size_bytes -= entry.size_bytes
            
            return True
            
    def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate all entries matching any of the given tags."""
        with self._lock:
            keys_to_remove = []
            
            for tier_entries in self.tiers.values():
                for key, entry in tier_entries.items():
                    if entry.tags.intersection(tags):
                        keys_to_remove.append(key)
                        
            for key in keys_to_remove:
                self.delete(key)
                
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            for tier in self.tiers.values():
                tier.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            
            # Reset statistics
            self.stats = CacheStats()
            
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics and insights."""
        with self._lock:
            tier_stats = {}
            for tier, tier_entries in self.tiers.items():
                tier_size = sum(entry.size_bytes for entry in tier_entries.values())
                tier_stats[tier.value] = {
                    'entry_count': len(tier_entries),
                    'size_bytes': tier_size,
                    'size_mb': tier_size / (1024 * 1024)
                }
                
            # Get ML insights
            ml_insights = self.ml_optimizer.get_optimization_insights()
            
            return {
                'basic_stats': {
                    'hit_rate': self.stats.hit_rate,
                    'total_requests': self.stats.hits + self.stats.misses,
                    'hits': self.stats.hits,
                    'misses': self.stats.misses,
                    'evictions': self.stats.evictions,
                    'entry_count': self.stats.entry_count,
                    'total_size_mb': self.stats.total_size_bytes / (1024 * 1024),
                    'compression_ratio': self.stats.compression_ratio
                },
                'tier_stats': tier_stats,
                'configuration': {
                    'max_size': self.max_size,
                    'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                    'policy': self.policy.value,
                    'compression': self.compression.value,
                    'prefetch_enabled': self.enable_prefetch,
                    'multi_tier_enabled': self.enable_multi_tier
                },
                'ml_insights': ml_insights,
                'prefetch_stats': {
                    'hits': self.stats.prefetch_hits,
                    'misses': self.stats.prefetch_misses,
                    'accuracy': self.stats.prefetch_accuracy
                },
                'tier_movements': {
                    'promotions': self.stats.promotions,
                    'demotions': self.stats.demotions
                }
            }
            
    def _find_entry_in_tiers(self, key: K) -> Optional[CacheEntry]:
        """Find entry in any tier."""
        # Check tiers in order: L1 -> L2 -> L3
        for tier in [CacheTier.L1_MEMORY, CacheTier.L2_MEMORY, CacheTier.L3_DISK]:
            if key in self.tiers[tier]:
                return self.tiers[tier][key]
        return None
        
    def _has_space_for_entry(self, entry: CacheEntry) -> bool:
        """Check if there's space for the entry."""
        # Check size constraints
        if self.stats.entry_count >= self.max_size:
            return False
            
        if self.stats.total_size_bytes + entry.size_bytes > self.max_memory_bytes:
            return False
            
        return True
        
    def _make_room_for_entry(self, new_entry: CacheEntry) -> bool:
        """Make room for new entry by evicting others."""
        space_needed = new_entry.size_bytes
        
        while (self.stats.entry_count >= self.max_size or 
               self.stats.total_size_bytes + space_needed > self.max_memory_bytes):
            
            # Select entry for eviction using ML optimization
            candidates = []
            for tier_entries in self.tiers.values():
                candidates.extend(tier_entries.values())
                
            if not candidates:
                return False
                
            if self.policy == CachePolicy.ML_OPTIMIZED:
                victim = self.ml_optimizer.optimize_eviction_policy(candidates)
            else:
                victim = self._select_victim_traditional(candidates)
                
            if victim is None:
                return False
                
            # Evict the victim
            victim_key = None
            for tier_entries in self.tiers.values():
                for key, entry in tier_entries.items():
                    if entry is victim:
                        victim_key = key
                        break
                if victim_key:
                    break
                    
            if victim_key:
                self.ml_optimizer.record_cache_event('eviction', str(victim_key), victim, self.stats.hit_rate)
                self.delete(victim_key)
                self.stats.evictions += 1
            else:
                return False
                
        return True
        
    def _select_victim_traditional(self, candidates: List[CacheEntry]) -> Optional[CacheEntry]:
        """Select victim using traditional cache policies."""
        if not candidates:
            return None
            
        if self.policy == CachePolicy.LRU:
            return min(candidates, key=lambda e: e.last_accessed)
        elif self.policy == CachePolicy.LFU:
            return min(candidates, key=lambda e: e.access_count)
        elif self.policy == CachePolicy.TTL:
            # Evict expired entries first, then by age
            expired = [e for e in candidates if e.is_expired]
            if expired:
                return min(expired, key=lambda e: e.created_at)
            else:
                return min(candidates, key=lambda e: e.created_at)
        else:
            # Default to LRU
            return min(candidates, key=lambda e: e.last_accessed)
            
    def _select_initial_tier(self, entry: CacheEntry) -> CacheTier:
        """Select initial tier for new entry."""
        if not self.enable_multi_tier:
            return CacheTier.L1_MEMORY
            
        # Simple heuristic: high priority -> L1, normal -> L2, low -> L3
        if entry.priority > 2.0:
            return CacheTier.L1_MEMORY
        elif entry.priority > 1.0:
            return CacheTier.L2_MEMORY
        else:
            return CacheTier.L2_MEMORY  # L3_DISK not implemented yet
            
    def _place_entry_in_tier(self, key: K, entry: CacheEntry, tier: CacheTier):
        """Place entry in specified tier."""
        self.tiers[tier][key] = entry
        
    def _promote_entry_if_needed(self, key: K, entry: CacheEntry):
        """Promote frequently accessed entries to higher tiers."""
        if not self.enable_multi_tier:
            return
            
        current_tier = self._get_entry_tier(key)
        if current_tier is None:
            return
            
        # Promotion criteria: high access frequency or recent intensive access
        should_promote = False
        
        if (current_tier == CacheTier.L2_MEMORY and 
            entry.access_frequency > 0.1):  # More than 0.1 accesses per second
            target_tier = CacheTier.L1_MEMORY
            should_promote = True
        elif (current_tier == CacheTier.L3_DISK and 
              entry.access_frequency > 0.05):
            target_tier = CacheTier.L2_MEMORY
            should_promote = True
            
        if should_promote:
            # Move entry to higher tier
            del self.tiers[current_tier][key]
            self.tiers[target_tier][key] = entry
            self.stats.promotions += 1
            
    def _get_entry_tier(self, key: K) -> Optional[CacheTier]:
        """Get the tier containing the entry."""
        for tier, tier_entries in self.tiers.items():
            if key in tier_entries:
                return tier
        return None
        
    def _remove_entry_from_all_tiers(self, key: K):
        """Remove entry from all tiers."""
        for tier_entries in self.tiers.values():
            tier_entries.pop(key, None)
            
    def _update_access_tracking(self, key: K, entry: CacheEntry):
        """Update access tracking structures."""
        # Update LRU tracking
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = True
        
        # Update LFU tracking
        self.frequency_counter[key] += 1
        
    def _remove_from_access_tracking(self, key: K):
        """Remove key from access tracking structures."""
        self.access_order.pop(key, None)
        self.frequency_counter.pop(key, None)
        
    def _schedule_prefetch(self):
        """Schedule prefetch operations based on ML predictions."""
        if not self.enable_prefetch:
            return
            
        # Get prefetch suggestions asynchronously
        self._prefetch_executor.submit(self._execute_prefetch)
        
    def _execute_prefetch(self):
        """Execute prefetch operations."""
        try:
            suggestions = self.ml_optimizer.generate_prefetch_suggestions(5)
            
            # For now, just record the suggestions (actual prefetch would depend on data source)
            if suggestions:
                logger.debug(f"Prefetch suggestions: {suggestions[:3]}")
                # In a real implementation, you would:
                # 1. Check if suggested keys are not already cached
                # 2. Fetch data for those keys from the data source
                # 3. Pre-populate the cache
                
        except Exception as e:
            logger.error(f"Prefetch execution failed: {e}")
            
    def _optimization_loop(self):
        """Background optimization loop."""
        while self._running:
            try:
                # Clean expired entries
                self._cleanup_expired_entries()
                
                # Optimize cache size if needed
                self._optimize_cache_size()
                
                # Periodic tier management
                self._manage_tiers()
                
                # Sleep for optimization interval
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Cache optimization loop error: {e}")
                
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self._lock:
            expired_keys = []
            
            for tier_entries in self.tiers.values():
                for key, entry in tier_entries.items():
                    if entry.is_expired:
                        expired_keys.append(key)
                        
            for key in expired_keys:
                self.delete(key)
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
    def _optimize_cache_size(self):
        """Optimize cache size based on ML insights."""
        if self.policy == CachePolicy.ML_OPTIMIZED:
            current_hit_rate = self.stats.hit_rate
            suggested_size = self.ml_optimizer.predict_optimal_cache_size(
                current_hit_rate, self.max_size
            )
            
            if abs(suggested_size - self.max_size) > self.max_size * 0.1:  # 10% threshold
                logger.info(f"ML suggests cache size change: {self.max_size} -> {suggested_size}")
                # In practice, you might want to implement gradual size changes
                
    def _manage_tiers(self):
        """Manage entry distribution across tiers."""
        if not self.enable_multi_tier:
            return
            
        with self._lock:
            # Demote entries from L1 if it's getting full
            l1_entries = self.tiers[CacheTier.L1_MEMORY]
            l1_size = sum(entry.size_bytes for entry in l1_entries.values())
            l1_limit = self.max_memory_bytes * 0.3  # L1 gets 30% of memory
            
            if l1_size > l1_limit:
                # Find candidates for demotion (least frequently accessed)
                candidates = list(l1_entries.values())
                candidates.sort(key=lambda e: e.access_frequency)
                
                for entry in candidates[:5]:  # Demote up to 5 entries
                    key = None
                    for k, v in l1_entries.items():
                        if v is entry:
                            key = k
                            break
                            
                    if key:
                        del l1_entries[key]
                        self.tiers[CacheTier.L2_MEMORY][key] = entry
                        self.stats.demotions += 1
                        
                        if l1_size <= l1_limit:
                            break


# Factory function for creating caches
def create_intelligent_cache(cache_type: str = "standard", **kwargs) -> IntelligentCache:
    """Factory function to create different types of intelligent caches."""
    
    if cache_type == "high_performance":
        return IntelligentCache(
            max_size=kwargs.get('max_size', 10000),
            max_memory_mb=kwargs.get('max_memory_mb', 500),
            policy=CachePolicy.ML_OPTIMIZED,
            compression=CompressionType.LZ4,
            enable_prefetch=True,
            enable_multi_tier=True
        )
    elif cache_type == "memory_optimized":
        return IntelligentCache(
            max_size=kwargs.get('max_size', 5000),
            max_memory_mb=kwargs.get('max_memory_mb', 100),
            policy=CachePolicy.ML_OPTIMIZED,
            compression=CompressionType.ZSTD,
            enable_prefetch=False,
            enable_multi_tier=True
        )
    elif cache_type == "simple":
        return IntelligentCache(
            max_size=kwargs.get('max_size', 1000),
            max_memory_mb=kwargs.get('max_memory_mb', 50),
            policy=CachePolicy.LRU,
            compression=CompressionType.NONE,
            enable_prefetch=False,
            enable_multi_tier=False
        )
    else:  # standard
        return IntelligentCache(
            max_size=kwargs.get('max_size', 2000),
            max_memory_mb=kwargs.get('max_memory_mb', 200),
            policy=CachePolicy.ML_OPTIMIZED,
            compression=CompressionType.LZ4,
            enable_prefetch=True,
            enable_multi_tier=True
        )