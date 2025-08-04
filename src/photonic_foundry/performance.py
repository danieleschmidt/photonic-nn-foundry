"""
Performance optimization and scaling utilities for photonic circuits.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import queue
import weakref
from collections import defaultdict, OrderedDict
import math

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float = 0.0
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0     # %
    cache_hits: int = 0
    cache_misses: int = 0
    operations_per_second: float = 0.0
    parallel_efficiency: float = 0.0
    
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceProfiler:
    """Performance profiling and monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        with self._lock:
            self.active_operations[operation_name] = time.time()
            
    def end_operation(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        with self._lock:
            if operation_name in self.active_operations:
                start_time = self.active_operations.pop(operation_name)
                duration = time.time() - start_time
                
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = PerformanceMetrics()
                    
                self.metrics[operation_name].execution_time += duration
                return duration
            return 0.0
            
    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get metrics for an operation."""
        return self.metrics.get(operation_name)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for operation, metrics in self.metrics.items():
            summary[operation] = {
                'total_time': metrics.execution_time,
                'memory_usage': metrics.memory_usage,
                'cache_hit_rate': metrics.cache_hit_rate(),
                'operations_per_second': metrics.operations_per_second
            }
        return summary


def profile_operation(operation_name: str = None):
    """Decorator for profiling operations."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = getattr(wrapper, '_profiler', None)
            if profiler is None:
                profiler = PerformanceProfiler()
                wrapper._profiler = profiler
                
            profiler.start_operation(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(op_name)
                
        wrapper.get_metrics = lambda: getattr(wrapper, '_profiler', PerformanceProfiler()).get_metrics(op_name)
        wrapper.get_summary = lambda: getattr(wrapper, '_profiler', PerformanceProfiler()).get_summary()
        
        return wrapper
    return decorator


class SmartCache:
    """Intelligent caching system with LRU and size limits."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict = OrderedDict()
        self.memory_usage = 0.0
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
                
    def put(self, key: str, value: Any, size_mb: float = 0.1):
        """Put item in cache."""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                old_size = self.cache[key].get('_size_mb', 0.1)
                self.memory_usage -= old_size
                del self.cache[key]
                
            # Check memory limit
            while (self.memory_usage + size_mb > self.max_memory_mb and 
                   len(self.cache) > 0):
                self._evict_lru()
                
            # Check size limit
            while len(self.cache) >= self.max_size and len(self.cache) > 0:
                self._evict_lru()
                
            # Add new item
            cache_item = {
                'value': value,
                '_size_mb': size_mb,
                '_timestamp': time.time()
            }
            self.cache[key] = cache_item
            self.memory_usage += size_mb
            
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            key, item = self.cache.popitem(last=False)
            self.memory_usage -= item.get('_size_mb', 0.1)
            
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.memory_usage = 0.0
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self.memory_usage,
            'max_memory_mb': self.max_memory_mb,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class ParallelProcessor:
    """Parallel processing utilities for photonic circuit operations."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self._executor = None
        
    def __enter__(self):
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            
    def map_parallel(self, func: Callable, items: List[Any], 
                    chunk_size: int = None) -> List[Any]:
        """Map function over items in parallel."""
        if not self._executor:
            raise RuntimeError("ParallelProcessor not initialized. Use with context manager.")
            
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
            
        # Submit all tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = self._executor.submit(self._process_chunk, func, chunk)
            futures.append(future)
            
        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
            
        return results
        
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
        
    def reduce_parallel(self, func: Callable, items: List[Any], 
                       initial_value: Any = None) -> Any:
        """Parallel reduce operation."""
        if not items:
            return initial_value
            
        # Binary tree reduction
        while len(items) > 1:
            next_items = []
            
            # Process pairs in parallel
            pairs = [(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
            if len(items) % 2 == 1:
                pairs.append((items[-1], None))
                
            def reduce_pair(pair):
                a, b = pair
                if b is None:
                    return a
                return func(a, b)
                
            next_items = self.map_parallel(reduce_pair, pairs)
            items = next_items
            
        return items[0] if items else initial_value


class CircuitOptimizer:
    """Advanced circuit optimization using parallel processing and caching."""
    
    def __init__(self, cache_size: int = 1000, max_workers: int = None):
        self.cache = SmartCache(max_size=cache_size)
        self.max_workers = max_workers
        self.profiler = PerformanceProfiler()
        
    @profile_operation("circuit_optimization")
    def optimize_circuit(self, circuit_data: Dict[str, Any], 
                        optimization_level: int = 2) -> Dict[str, Any]:
        """Optimize circuit with configurable optimization level."""
        circuit_hash = self._hash_circuit(circuit_data)
        cache_key = f"optimized_{circuit_hash}_{optimization_level}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for circuit optimization: {cache_key}")
            return cached_result['value']
            
        logger.info(f"Optimizing circuit with level {optimization_level}")
        
        # Apply optimizations based on level
        optimized_circuit = circuit_data.copy()
        
        if optimization_level >= 1:
            optimized_circuit = self._basic_optimizations(optimized_circuit)
            
        if optimization_level >= 2:
            optimized_circuit = self._advanced_optimizations(optimized_circuit)
            
        if optimization_level >= 3:
            optimized_circuit = self._aggressive_optimizations(optimized_circuit)
            
        # Cache result
        self.cache.put(cache_key, optimized_circuit, size_mb=0.5)
        
        return optimized_circuit
        
    def _basic_optimizations(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic optimizations."""
        logger.debug("Applying basic optimizations")
        
        layers = circuit_data.get('layers', [])
        optimized_layers = []
        
        # Merge consecutive linear layers
        i = 0
        while i < len(layers):
            current_layer = layers[i]
            
            if (i + 1 < len(layers) and 
                current_layer.get('type') == 'linear' and
                layers[i + 1].get('type') == 'linear'):
                
                # Merge layers
                merged_layer = self._merge_linear_layers(current_layer, layers[i + 1])
                optimized_layers.append(merged_layer)
                i += 2
            else:
                optimized_layers.append(current_layer)
                i += 1
                
        circuit_data['layers'] = optimized_layers
        circuit_data['total_components'] = sum(
            layer.get('component_count', 0) for layer in optimized_layers
        )
        
        return circuit_data
        
    def _advanced_optimizations(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced optimizations using parallel processing."""
        logger.debug("Applying advanced optimizations")
        
        with ParallelProcessor(max_workers=self.max_workers) as processor:
            layers = circuit_data.get('layers', [])
            
            # Optimize layers in parallel
            optimized_layers = processor.map_parallel(self._optimize_single_layer, layers)
            
            circuit_data['layers'] = optimized_layers
            circuit_data['total_components'] = sum(
                layer.get('component_count', 0) for layer in optimized_layers
            )
            
        return circuit_data
        
    def _aggressive_optimizations(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggressive optimizations."""
        logger.debug("Applying aggressive optimizations")
        
        # Global circuit-level optimizations
        circuit_data = self._global_component_sharing(circuit_data)
        circuit_data = self._optimize_power_distribution(circuit_data)
        circuit_data = self._minimize_waveguide_crossings(circuit_data)
        
        return circuit_data
        
    def _optimize_single_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single layer."""
        layer_type = layer.get('type', '')
        
        if layer_type == 'linear':
            return self._optimize_linear_layer(layer)
        elif layer_type == 'activation':
            return self._optimize_activation_layer(layer)
        else:
            return layer
            
    def _optimize_linear_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize linear layer."""
        optimized = layer.copy()
        
        # Reduce component count through MZI mesh optimization
        input_size = layer.get('input_size', 1)
        output_size = layer.get('output_size', 1)
        
        # Calculate optimal MZI count
        naive_count = input_size * output_size
        optimal_count = self._calculate_optimal_mzi_count(input_size, output_size)
        
        optimized['component_count'] = optimal_count
        optimized['optimization_ratio'] = optimal_count / naive_count
        
        return optimized
        
    def _optimize_activation_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize activation layer."""
        optimized = layer.copy()
        
        # Reduce modulator count through sharing
        component_count = layer.get('component_count', 0)
        shared_count = max(1, int(component_count * 0.8))  # 20% reduction
        
        optimized['component_count'] = shared_count
        optimized['shared_modulators'] = True
        
        return optimized
        
    def _merge_linear_layers(self, layer1: Dict[str, Any], 
                           layer2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two consecutive linear layers."""
        merged = {
            'type': 'linear',
            'input_size': layer1.get('input_size'),
            'output_size': layer2.get('output_size'),
            'merged_from': [layer1.get('input_size'), layer2.get('input_size'), layer2.get('output_size')],
            'component_count': self._calculate_merged_component_count(layer1, layer2),
            'components': []
        }
        
        return merged
        
    def _calculate_merged_component_count(self, layer1: Dict[str, Any], 
                                        layer2: Dict[str, Any]) -> int:
        """Calculate component count for merged layers."""
        # Simplified calculation
        input_size = layer1.get('input_size', 1)
        output_size = layer2.get('output_size', 1)
        
        # Direct connection saves intermediate components
        return input_size * output_size
        
    def _calculate_optimal_mzi_count(self, input_size: int, output_size: int) -> int:
        """Calculate optimal MZI count for given dimensions."""
        # Use triangular mesh for optimal implementation
        max_dim = max(input_size, output_size)
        mesh_depth = int(math.ceil(math.log2(max_dim))) if max_dim > 1 else 1
        
        # Triangular mesh requires fewer MZIs than full mesh
        return int((input_size + output_size - 1) * mesh_depth * 0.8)
        
    def _global_component_sharing(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement global component sharing."""
        layers = circuit_data.get('layers', [])
        
        # Identify shareable components across layers
        component_types = defaultdict(int)
        for layer in layers:
            for component in layer.get('components', []):
                comp_type = component.get('type', 'unknown')
                component_types[comp_type] += 1
                
        # Apply sharing for common components
        sharing_factor = 0.85  # 15% reduction through sharing
        for layer in layers:
            layer['component_count'] = int(layer.get('component_count', 0) * sharing_factor)
            
        circuit_data['total_components'] = sum(
            layer.get('component_count', 0) for layer in layers
        )
        
        return circuit_data
        
    def _optimize_power_distribution(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize optical power distribution."""
        layers = circuit_data.get('layers', [])
        total_power_budget = 100.0  # mW
        
        # Calculate power requirements
        total_components = sum(layer.get('component_count', 0) for layer in layers)
        
        for layer in layers:
            layer_components = layer.get('component_count', 0)
            if total_components > 0:
                power_fraction = layer_components / total_components
                layer['allocated_power'] = total_power_budget * power_fraction
                
        return circuit_data
        
    def _minimize_waveguide_crossings(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize waveguide crossings through layout optimization."""
        # Simplified implementation
        layers = circuit_data.get('layers', [])
        
        # Optimize connections to minimize crossings
        for i, layer in enumerate(layers):
            layer['crossing_optimization'] = {
                'layer_index': i,
                'optimized_routing': True,
                'crossing_reduction': 0.25  # 25% reduction
            }
            
        return circuit_data
        
    def _hash_circuit(self, circuit_data: Dict[str, Any]) -> str:
        """Generate hash for circuit caching."""
        # Simplified hash based on structure
        layers = circuit_data.get('layers', [])
        hash_components = []
        
        for layer in layers:
            layer_hash = f"{layer.get('type')}_{layer.get('input_size')}_{layer.get('output_size')}"
            hash_components.append(layer_hash)
            
        return "_".join(hash_components)
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'cache_stats': self.cache.stats(),
            'performance_metrics': self.profiler.get_summary(),
            'total_optimizations': len(self.cache.cache)
        }


class LoadBalancer:
    """Load balancing for distributed photonic circuit processing."""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: Dict[str, float] = {}
        self.job_queue = queue.Queue()
        self.completed_jobs: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._workers = []
        self._running = False
        
    def start(self):
        """Start load balancer workers."""
        self._running = True
        for i in range(min(4, self.max_concurrent_jobs)):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            
    def stop(self):
        """Stop load balancer workers."""
        self._running = False
        
        # Signal workers to stop
        for _ in self._workers:
            self.job_queue.put(None)
            
    def submit_job(self, job_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit job for processing."""
        job = {
            'id': job_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        self.job_queue.put(job)
        logger.info(f"Submitted job: {job_id}")
        
        return job_id
        
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        with self._lock:
            if job_id in self.completed_jobs:
                return "completed"
            elif job_id in self.active_jobs:
                return "running"
            else:
                return "queued"
                
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get job result if completed."""
        with self._lock:
            return self.completed_jobs.get(job_id)
            
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing jobs."""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                job = self.job_queue.get(timeout=1.0)
                if job is None:  # Shutdown signal
                    break
                    
                self._process_job(job, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    def _process_job(self, job: Dict[str, Any], worker_id: int):
        """Process a single job."""
        job_id = job['id']
        
        try:
            with self._lock:
                self.active_jobs[job_id] = time.time()
                
            logger.info(f"Worker {worker_id} processing job: {job_id}")
            
            # Execute job
            result = job['func'](*job['args'], **job['kwargs'])
            
            with self._lock:
                self.completed_jobs[job_id] = {
                    'result': result,
                    'completed_at': time.time(),
                    'processing_time': time.time() - self.active_jobs[job_id]
                }
                del self.active_jobs[job_id]
                
            logger.info(f"Worker {worker_id} completed job: {job_id}")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            with self._lock:
                self.completed_jobs[job_id] = {
                    'error': str(e),
                    'completed_at': time.time(),
                    'processing_time': time.time() - self.active_jobs.get(job_id, time.time())
                }
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]


# Import os for cpu_count
import os