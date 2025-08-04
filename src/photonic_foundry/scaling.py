"""
Auto-scaling and performance optimization for photonic systems.
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Union
import queue
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib
import psutil

from .monitoring import get_metrics_collector, get_performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 1
    max_workers: int = multiprocessing.cpu_count()
    target_cpu_utilization: float = 70.0  # Percentage
    scale_up_threshold: float = 80.0      # CPU % to scale up
    scale_down_threshold: float = 40.0    # CPU % to scale down
    scale_up_cooldown: int = 60           # Seconds before scaling up again
    scale_down_cooldown: int = 300        # Seconds before scaling down again
    queue_threshold: int = 10             # Queue size to trigger scaling
    enable_process_pool: bool = True      # Use processes instead of threads for CPU-bound tasks


class AdaptiveCache:
    """Intelligent caching system that adapts to access patterns."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.Lock()
        
        # Adaptive parameters
        self.hit_rate_threshold = 0.3
        self.access_pattern_window = 1000
        self.recent_accesses = queue.deque(maxlen=self.access_pattern_window)
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if (key in self.cache and 
                current_time - self.creation_times.get(key, 0) < self.ttl_seconds):
                
                # Update access statistics
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.access_times[key] = current_time
                self.recent_accesses.append(key)
                
                return self.cache[key]
                
            # Remove expired items
            elif key in self.cache:
                self._remove_item(key)
                
            return None
            
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # If cache is full, make room
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_item()
                
            # Store the item
            self.cache[key] = value
            self.access_counts[key] = 1
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            self.recent_accesses.append(key)
            
            return True
            
    def _evict_item(self):
        """Intelligently evict items based on access patterns."""
        if not self.cache:
            return
            
        current_time = time.time()
        
        # Calculate scores for eviction (lower score = more likely to evict)
        scores = {}
        for key in self.cache:
            access_count = self.access_counts.get(key, 1)
            last_access = self.access_times.get(key, 0)
            age = current_time - self.creation_times.get(key, current_time)
            
            # Score based on frequency, recency, and age
            frequency_score = access_count / 10.0
            recency_score = max(0, 1 - (current_time - last_access) / self.ttl_seconds)
            age_penalty = age / self.ttl_seconds
            
            scores[key] = frequency_score + recency_score - age_penalty
            
        # Evict item with lowest score
        evict_key = min(scores, key=scores.get)
        self._remove_item(evict_key)
        
    def _remove_item(self, key: str):
        """Remove item from cache and all tracking structures."""
        self.cache.pop(key, None)
        self.access_counts.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics and performance metrics."""
        with self._lock:
            total_accesses = sum(self.access_counts.values())
            unique_keys = len(self.access_counts)
            
            # Calculate hit rate from recent accesses
            recent_keys = list(self.recent_accesses)
            hits = sum(1 for key in recent_keys if key in self.cache)
            hit_rate = hits / len(recent_keys) if recent_keys else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_accesses': total_accesses,
                'unique_keys': unique_keys,
                'avg_accesses_per_key': total_accesses / max(unique_keys, 1),
                'utilization': len(self.cache) / self.max_size
            }


class AutoScaler:
    """Automatic scaling system for photonic processing workloads."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.executor = None
        self.task_queue = queue.Queue()
        self.last_scale_up = 0
        self.last_scale_down = 0
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._running = False
        
        # Performance tracking
        self.metrics_collector = get_metrics_collector()
        self.performance_monitor = get_performance_monitor()
        
        # Initialize executor
        self._create_executor()
        
    def _create_executor(self):
        """Create appropriate executor based on configuration."""
        if self.executor:
            self.executor.shutdown(wait=False)
            
        if self.config.enable_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
            
        logger.info(f"Created executor with {self.current_workers} workers")
        
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started auto-scaling monitoring")
        
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Stopped auto-scaling monitoring")
        
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task for auto-scaled execution."""
        return self.executor.submit(func, *args, **kwargs)
        
    def submit_batch(self, func: Callable, items: List[Any], **kwargs):
        """Submit batch of tasks for parallel processing."""
        futures = []
        for item in items:
            future = self.executor.submit(func, item, **kwargs)
            futures.append(future)
        return futures
        
    def _monitoring_loop(self):
        """Monitor system metrics and adjust scaling."""
        while self._running:
            try:
                # Get current system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                queue_size = self.task_queue.qsize()
                current_time = time.time()
                
                # Record metrics
                self.metrics_collector.record_metric("cpu_utilization", cpu_percent, "percent")
                self.metrics_collector.record_metric("task_queue_size", queue_size, "count")
                self.metrics_collector.record_metric("active_workers", self.current_workers, "count")
                
                # Check scaling conditions
                should_scale_up = (
                    cpu_percent > self.config.scale_up_threshold or
                    queue_size > self.config.queue_threshold
                ) and (current_time - self.last_scale_up > self.config.scale_up_cooldown)
                
                should_scale_down = (
                    cpu_percent < self.config.scale_down_threshold and
                    queue_size < 2
                ) and (current_time - self.last_scale_down > self.config.scale_down_cooldown)
                
                # Execute scaling decisions
                if should_scale_up and self.current_workers < self.config.max_workers:
                    self._scale_up()
                elif should_scale_down and self.current_workers > self.config.min_workers:
                    self._scale_down()
                    
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(30)
                
    def _scale_up(self):
        """Scale up the number of workers."""
        with self._lock:
            new_workers = min(self.current_workers + 1, self.config.max_workers)
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                self.last_scale_up = time.time()
                self._create_executor()
                logger.info(f"Scaled up to {self.current_workers} workers")
                
    def _scale_down(self):
        """Scale down the number of workers."""
        with self._lock:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                self.last_scale_down = time.time()
                self._create_executor()
                logger.info(f"Scaled down to {self.current_workers} workers")
                
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down,
            'executor_type': 'ProcessPoolExecutor' if self.config.enable_process_pool else 'ThreadPoolExecutor',
            'task_queue_size': self.task_queue.qsize()
        }


class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self):
        self.adaptive_cache = AdaptiveCache(max_size=2000)
        self.optimization_strategies = {
            'batch_processing': self._optimize_batch_processing,
            'memory_pooling': self._optimize_memory_pooling,
            'algorithm_selection': self._optimize_algorithm_selection,
            'caching_strategy': self._optimize_caching_strategy
        }
        
        # Performance baselines
        self.baselines = {}
        self.optimization_history = []
        
    def optimize_workload(self, workload_type: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Apply performance optimizations to a workload."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(workload_type, data, kwargs)
        cached_result = self.adaptive_cache.get(cache_key)
        
        if cached_result:
            return {
                'result': cached_result,
                'cache_hit': True,
                'processing_time': time.time() - start_time,
                'optimizations_applied': ['caching']
            }
            
        # Apply optimizations
        optimizations_applied = []
        optimized_data = data
        
        for strategy, optimizer in self.optimization_strategies.items():
            try:
                optimized_data, applied = optimizer(optimized_data, workload_type, **kwargs)
                if applied:
                    optimizations_applied.append(strategy)
            except Exception as e:
                logger.warning(f"Optimization strategy {strategy} failed: {e}")
                
        # Cache the result
        self.adaptive_cache.put(cache_key, optimized_data)
        
        processing_time = time.time() - start_time
        
        # Record optimization effectiveness
        self.optimization_history.append({
            'workload_type': workload_type,
            'processing_time': processing_time,
            'optimizations_applied': optimizations_applied,
            'timestamp': time.time()
        })
        
        return {
            'result': optimized_data,
            'cache_hit': False,
            'processing_time': processing_time,
            'optimizations_applied': optimizations_applied
        }
        
    def _optimize_batch_processing(self, data: Any, workload_type: str, **kwargs) -> tuple:
        """Optimize for batch processing patterns."""
        if isinstance(data, list) and len(data) > 10:
            # Sort data for better cache locality
            if all(hasattr(item, '__lt__') for item in data):
                try:
                    data = sorted(data)
                    return data, True
                except:
                    pass
        return data, False
        
    def _optimize_memory_pooling(self, data: Any, workload_type: str, **kwargs) -> tuple:
        """Optimize memory usage patterns."""
        # For large numpy arrays, ensure optimal memory layout
        if isinstance(data, np.ndarray) and data.size > 1000:
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)
                return data, True
        return data, False
        
    def _optimize_algorithm_selection(self, data: Any, workload_type: str, **kwargs) -> tuple:
        """Select optimal algorithms based on data characteristics."""
        # This would contain logic to choose different algorithms
        # based on data size, type, and historical performance
        return data, False
        
    def _optimize_caching_strategy(self, data: Any, workload_type: str, **kwargs) -> tuple:
        """Optimize caching based on access patterns."""
        # Adjust cache parameters based on workload
        cache_stats = self.adaptive_cache.get_stats()
        
        if cache_stats['hit_rate'] < 0.3 and cache_stats['utilization'] > 0.8:
            # Increase cache size if hit rate is low but cache is full
            self.adaptive_cache.max_size = min(self.adaptive_cache.max_size * 1.2, 5000)
            return data, True
            
        return data, False
        
    def _generate_cache_key(self, workload_type: str, data: Any, kwargs: Dict) -> str:
        """Generate cache key for workload."""
        key_components = [workload_type]
        
        # Add data hash
        if isinstance(data, np.ndarray):
            key_components.append(str(data.shape) + str(data.dtype))
        elif isinstance(data, (list, tuple)):
            key_components.append(f"list_{len(data)}")
        else:
            key_components.append(str(type(data)))
            
        # Add relevant kwargs
        for k, v in sorted(kwargs.items()):
            key_components.append(f"{k}:{v}")
            
        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        recent_history = [
            h for h in self.optimization_history 
            if time.time() - h['timestamp'] < 3600  # Last hour
        ]
        
        if not recent_history:
            return {'no_data': True}
            
        avg_processing_time = np.mean([h['processing_time'] for h in recent_history])
        optimization_counts = {}
        
        for history in recent_history:
            for opt in history['optimizations_applied']:
                optimization_counts[opt] = optimization_counts.get(opt, 0) + 1
                
        return {
            'total_optimizations': len(recent_history),
            'avg_processing_time': avg_processing_time,
            'optimization_effectiveness': optimization_counts,
            'cache_stats': self.adaptive_cache.get_stats(),
            'most_effective_optimization': max(optimization_counts, key=optimization_counts.get) if optimization_counts else None
        }


# Global instances
_auto_scaler = None
_performance_optimizer = None


def get_auto_scaler(config: Optional[ScalingConfig] = None) -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        if config is None:
            config = ScalingConfig()
        _auto_scaler = AutoScaler(config)
    return _auto_scaler


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def start_scaling_services():
    """Start all scaling and optimization services."""
    scaler = get_auto_scaler()
    scaler.start_monitoring()
    logger.info("Auto-scaling services started")


def stop_scaling_services():
    """Stop all scaling services."""
    if _auto_scaler:
        _auto_scaler.stop_monitoring()
    logger.info("Auto-scaling services stopped")


def get_scaling_status() -> Dict[str, Any]:
    """Get comprehensive scaling and optimization status."""
    scaler = get_auto_scaler()
    optimizer = get_performance_optimizer()
    
    return {
        'auto_scaling': scaler.get_scaling_stats(),
        'performance_optimization': optimizer.get_optimization_stats(),
        'system_resources': {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1)
        },
        'timestamp': time.time()
    }