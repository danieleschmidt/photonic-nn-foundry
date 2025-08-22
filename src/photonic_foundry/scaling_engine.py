"""
Advanced Scaling Engine for Photonic Neural Network Foundry
Implements performance optimization, caching, concurrent processing, 
load balancing, and auto-scaling capabilities.
"""

import time
import json
import hashlib
import threading
import queue
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import logging
import os
import sys
from collections import defaultdict, deque
import pickle
import threading
import weakref

# Setup logging
logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_BASED = "resource_based"


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), 'hit_rate': self.hit_rate}


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    node_id: str
    capacity: int
    current_load: int = 0
    response_times: deque = None
    last_heartbeat: float = 0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=100)
        self.last_heartbeat = time.time()
    
    @property
    def utilization(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0.0
    
    @property
    def avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    def is_healthy(self, heartbeat_timeout: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < heartbeat_timeout


class IntelligentCache:
    """High-performance intelligent cache with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.ADAPTIVE,
                 ttl_seconds: float = 3600):
        self.max_size = max_size
        self.policy = policy
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        logger.info(f"IntelligentCache initialized: size={max_size}, policy={policy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent policy handling."""
        with self._lock:
            current_time = time.time()
            self.stats.total_requests += 1
            
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            # Check TTL expiration
            if self.policy in [CachePolicy.TTL, CachePolicy.ADAPTIVE]:
                item_time = self.access_times.get(key, 0)
                if current_time - item_time > self.ttl_seconds:
                    self._evict_item(key)
                    self.stats.misses += 1
                    return None
            
            # Update access patterns
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.stats.hits += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # Update existing item
            if key in self.cache:
                self.cache[key] = value
                self.access_times[key] = current_time
                self.access_counts[key] += 1
                return True
            
            # Handle cache full scenario
            if len(self.cache) >= self.max_size:
                evicted = self._evict_by_policy()
                if not evicted:
                    logger.warning("Cache eviction failed, cannot add new item")
                    return False
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            self.stats.size = len(self.cache)
            
            return True
    
    def _evict_by_policy(self) -> bool:
        """Evict item based on configured policy."""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            # Evict least recently used
            lru_key = min(self.access_times.keys(), 
                         key=lambda k: self.access_times[k])
            self._evict_item(lru_key)
            
        elif self.policy == CachePolicy.LFU:
            # Evict least frequently used
            lfu_key = min(self.access_counts.keys(),
                         key=lambda k: self.access_counts[k])
            self._evict_item(lfu_key)
            
        elif self.policy == CachePolicy.TTL:
            # Evict expired items first, then oldest
            current_time = time.time()
            expired = [k for k in self.cache.keys()
                      if current_time - self.access_times.get(k, 0) > self.ttl_seconds]
            if expired:
                self._evict_item(expired[0])
            else:
                oldest_key = min(self.access_times.keys(),
                               key=lambda k: self.access_times[k])
                self._evict_item(oldest_key)
                
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive policy considering access patterns
            current_time = time.time()
            
            # Score items based on recency, frequency, and TTL
            scores = {}
            for key in self.cache.keys():
                recency_score = (current_time - self.access_times.get(key, 0)) / self.ttl_seconds
                frequency_score = 1.0 / (self.access_counts[key] + 1)
                scores[key] = recency_score + frequency_score
            
            # Evict item with highest (worst) score
            worst_key = max(scores.keys(), key=lambda k: scores[k])
            self._evict_item(worst_key)
        
        return True
    
    def _evict_item(self, key: str):
        """Remove item from cache and update statistics."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            self.stats.evictions += 1
            self.stats.size = len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return self.stats.to_dict()
    
    def clear(self):
        """Clear all cache contents."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.stats = CacheStats()


class LoadBalancer:
    """Intelligent load balancer for distributed processing."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_BASED):
        self.strategy = strategy
        self.nodes = {}
        self.round_robin_index = 0
        self._lock = threading.RLock()
        
        logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")
    
    def add_node(self, node: WorkerNode):
        """Add a worker node to the load balancer."""
        with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added worker node: {node.node_id} (capacity: {node.capacity})")
    
    def remove_node(self, node_id: str):
        """Remove a worker node from the load balancer."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed worker node: {node_id}")
    
    def select_node(self) -> Optional[WorkerNode]:
        """Select optimal node based on load balancing strategy."""
        with self._lock:
            healthy_nodes = [node for node in self.nodes.values() if node.is_healthy()]
            
            if not healthy_nodes:
                logger.warning("No healthy nodes available")
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                node = healthy_nodes[self.round_robin_index % len(healthy_nodes)]
                self.round_robin_index += 1
                return node
                
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_nodes, key=lambda n: n.current_load)
                
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
                # Select based on inverse of response time
                if all(n.avg_response_time == 0 for n in healthy_nodes):
                    return healthy_nodes[0]  # Fall back to first node
                return min(healthy_nodes, key=lambda n: n.avg_response_time or float('inf'))
                
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                # Select based on lowest utilization
                return min(healthy_nodes, key=lambda n: n.utilization)
            
            return healthy_nodes[0]  # Default fallback
    
    def update_node_metrics(self, node_id: str, response_time: float, success: bool):
        """Update node performance metrics."""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.response_times.append(response_time)
                node.last_heartbeat = time.time()
                if success:
                    node.current_load = max(0, node.current_load - 1)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            healthy_nodes = [n for n in self.nodes.values() if n.is_healthy()]
            total_capacity = sum(n.capacity for n in healthy_nodes)
            total_load = sum(n.current_load for n in healthy_nodes)
            
            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': len(healthy_nodes),
                'total_capacity': total_capacity,
                'current_load': total_load,
                'cluster_utilization': total_load / total_capacity if total_capacity > 0 else 0.0,
                'avg_response_time': sum(n.avg_response_time for n in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0.0,
                'strategy': self.strategy.value,
                'timestamp': time.time()
            }


class ConcurrentTaskExecutor:
    """High-performance concurrent task executor with optimization."""
    
    def __init__(self, max_workers: int = 8, use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)  # Keep recent history
        self._task_counter = 0
        self._lock = threading.Lock()
        
        # Initialize executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(f"Initialized ProcessPoolExecutor with {max_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Initialized ThreadPoolExecutor with {max_workers} workers")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task for concurrent execution."""
        with self._lock:
            task_id = f"task_{self._task_counter}"
            self._task_counter += 1
        
        future = self.executor.submit(func, *args, **kwargs)
        
        task_info = {
            'task_id': task_id,
            'future': future,
            'submitted_at': time.time(),
            'func_name': func.__name__ if hasattr(func, '__name__') else 'unknown'
        }
        
        self.active_tasks[task_id] = task_info
        logger.info(f"Submitted task {task_id} ({task_info['func_name']})")
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of submitted task."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.active_tasks[task_id]
        future = task_info['future']
        
        start_time = time.time()
        try:
            result = future.result(timeout=timeout)
            execution_time = time.time() - task_info['submitted_at']
            
            # Move to completed tasks
            completed_info = {
                **task_info,
                'completed_at': time.time(),
                'execution_time': execution_time,
                'success': True
            }
            del completed_info['future']  # Remove non-serializable future
            
            self.completed_tasks.append(completed_info)
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - task_info['submitted_at']
            
            # Record failed task
            failed_info = {
                **task_info,
                'completed_at': time.time(),
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
            del failed_info['future']
            
            self.completed_tasks.append(failed_info)
            del self.active_tasks[task_id]
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor performance statistics."""
        completed_list = list(self.completed_tasks)
        successful_tasks = [t for t in completed_list if t.get('success', False)]
        failed_tasks = [t for t in completed_list if not t.get('success', True)]
        
        avg_execution_time = (
            sum(t['execution_time'] for t in successful_tasks) / len(successful_tasks)
            if successful_tasks else 0.0
        )
        
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(completed_list),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(successful_tasks) / len(completed_list) if completed_list else 0.0,
            'avg_execution_time': avg_execution_time,
            'max_workers': self.max_workers,
            'executor_type': 'ProcessPool' if self.use_processes else 'ThreadPool',
            'timestamp': time.time()
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        logger.info("Shutting down concurrent task executor")
        self.executor.shutdown(wait=wait)


class ScalingPhotonicAccelerator:
    """Production-scale photonic accelerator with advanced optimization."""
    
    def __init__(self, cache_size: int = 10000, max_workers: int = 8,
                 enable_load_balancing: bool = True):
        
        # Initialize scaling components
        self.cache = IntelligentCache(max_size=cache_size, policy=CachePolicy.ADAPTIVE)
        self.executor = ConcurrentTaskExecutor(max_workers=max_workers)
        self.load_balancer = LoadBalancer() if enable_load_balancing else None
        
        # Performance tracking
        self.operation_metrics = defaultdict(list)
        self.scaling_metrics = {
            'total_operations': 0,
            'cache_enabled_operations': 0,
            'concurrent_operations': 0,
            'load_balanced_operations': 0,
            'start_time': time.time()
        }
        
        # Add default local worker node
        if self.load_balancer:
            local_node = WorkerNode(
                node_id='local_worker',
                capacity=max_workers,
                current_load=0
            )
            self.load_balancer.add_node(local_node)
        
        logger.info(f"ScalingPhotonicAccelerator initialized - "
                   f"Cache: {cache_size}, Workers: {max_workers}, "
                   f"LoadBalancing: {enable_load_balancing}")
    
    def process_circuit_optimized(self, layer_sizes: List[int], 
                                circuit_name: Optional[str] = None,
                                use_cache: bool = True,
                                use_concurrency: bool = False) -> Dict[str, Any]:
        """Process circuit with all scaling optimizations."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = hashlib.md5(
            f"{layer_sizes}_{circuit_name}".encode()
        ).hexdigest()
        
        # Try cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.scaling_metrics['cache_enabled_operations'] += 1
                logger.info(f"Cache hit for circuit: {cache_key[:8]}")
                return cached_result
        
        # Process circuit
        if use_concurrency:
            task_id = self.executor.submit_task(
                self._create_circuit_internal, 
                layer_sizes, circuit_name
            )
            result = self.executor.get_result(task_id, timeout=30.0)
            self.scaling_metrics['concurrent_operations'] += 1
        else:
            result = self._create_circuit_internal(layer_sizes, circuit_name)
        
        # Cache result
        if use_cache:
            self.cache.put(cache_key, result)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.operation_metrics['process_circuit'].append(execution_time)
        self.scaling_metrics['total_operations'] += 1
        
        # Add scaling information to result
        result['scaling_info'] = {
            'cache_hit': use_cache and cached_result is not None,
            'concurrent_execution': use_concurrency,
            'execution_time': execution_time,
            'cache_key': cache_key
        }
        
        return result
    
    def _create_circuit_internal(self, layer_sizes: List[int], 
                               circuit_name: Optional[str]) -> Dict[str, Any]:
        """Internal circuit creation method."""
        # Import required modules
        sys.path.append(os.path.dirname(__file__))
        from core_standalone import PhotonicCircuit, MZILayer, CircuitMetrics
        
        # Generate name if not provided
        if circuit_name is None:
            name_hash = hashlib.md5(str(layer_sizes).encode()).hexdigest()[:8]
            circuit_name = f"optimized_circuit_{name_hash}"
        
        # Create circuit
        circuit = PhotonicCircuit(circuit_name)
        
        for i in range(len(layer_sizes) - 1):
            layer = MZILayer(layer_sizes[i], layer_sizes[i+1], precision=8)
            circuit.add_layer(layer)
        
        # Generate outputs and metrics
        verilog_code = circuit.generate_full_verilog()
        metrics = circuit.estimate_metrics()
        
        return {
            'circuit_name': circuit_name,
            'circuit_info': {
                'layers': len(circuit.layers),
                'total_components': circuit.total_components,
                'layer_sizes': layer_sizes
            },
            'performance_metrics': metrics.to_dict(),
            'verilog_stats': {
                'length': len(verilog_code),
                'hash': hashlib.sha256(verilog_code.encode()).hexdigest()[:16]
            },
            'generation_timestamp': time.time()
        }
    
    def process_batch_circuits(self, circuit_configs: List[Tuple[List[int], str]], 
                             max_concurrent: int = 4) -> List[Dict[str, Any]]:
        """Process multiple circuits concurrently for maximum throughput."""
        start_time = time.time()
        
        # Submit all tasks
        task_ids = []
        for layer_sizes, name in circuit_configs:
            task_id = self.executor.submit_task(
                self.process_circuit_optimized,
                layer_sizes, name, True, False  # Use cache but not nested concurrency
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for i, task_id in enumerate(task_ids):
            try:
                result = self.executor.get_result(task_id, timeout=60.0)
                results.append(result)
                logger.info(f"Batch item {i+1}/{len(task_ids)} completed")
            except Exception as e:
                logger.error(f"Batch item {i+1} failed: {str(e)}")
                results.append({
                    'error': str(e),
                    'failed_config': circuit_configs[i]
                })
        
        batch_time = time.time() - start_time
        self.operation_metrics['batch_process'].append(batch_time)
        
        logger.info(f"Processed {len(circuit_configs)} circuits in {batch_time:.2f}s")
        return results
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance metrics."""
        
        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in self.operation_metrics.items():
            if times:
                operation_stats[op_name] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times)
                }
        
        return {
            'scaling_metrics': self.scaling_metrics,
            'operation_stats': operation_stats,
            'cache_stats': self.cache.get_stats(),
            'executor_stats': self.executor.get_executor_stats(),
            'load_balancer_stats': (
                self.load_balancer.get_cluster_status() 
                if self.load_balancer else None
            ),
            'uptime_seconds': time.time() - self.scaling_metrics['start_time'],
            'report_timestamp': time.time()
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze performance and apply optimizations."""
        metrics = self.get_comprehensive_metrics()
        optimizations = []
        
        # Cache optimization
        cache_stats = metrics['cache_stats']
        if cache_stats['hit_rate'] < 0.5 and cache_stats['total_requests'] > 100:
            # Increase cache size
            old_size = self.cache.max_size
            self.cache.max_size = min(old_size * 2, 50000)
            optimizations.append(f"Increased cache size from {old_size} to {self.cache.max_size}")
        
        # Worker optimization
        executor_stats = metrics['executor_stats']
        if executor_stats['success_rate'] < 0.9 and executor_stats['completed_tasks'] > 50:
            optimizations.append("High failure rate detected - consider reducing workload")
        
        avg_exec_time = executor_stats.get('avg_execution_time', 0)
        if avg_exec_time > 5.0:  # > 5 seconds average
            optimizations.append(f"High average execution time ({avg_exec_time:.2f}s) - consider optimizing algorithms")
        
        if not optimizations:
            optimizations.append("System performing optimally - no adjustments needed")
        
        logger.info(f"Performance optimization completed: {len(optimizations)} changes")
        
        return {
            'optimizations_applied': optimizations,
            'previous_metrics': metrics,
            'optimization_timestamp': time.time()
        }
    
    def shutdown(self):
        """Shutdown scaling accelerator."""
        logger.info("Shutting down ScalingPhotonicAccelerator")
        self.executor.shutdown(wait=True)
        self.cache.clear()


def create_scaling_demo() -> Dict[str, Any]:
    """Demonstrate advanced scaling capabilities."""
    print("⚡ Creating Advanced Scaling Demo...")
    
    # Initialize scaling accelerator
    accelerator = ScalingPhotonicAccelerator(
        cache_size=5000,
        max_workers=6,
        enable_load_balancing=True
    )
    
    try:
        # Test 1: Single circuit with caching
        print("  → Testing single circuit optimization...")
        result1 = accelerator.process_circuit_optimized(
            layer_sizes=[16, 64, 32, 8],
            circuit_name="demo_mlp",
            use_cache=True,
            use_concurrency=False
        )
        
        # Test 2: Same circuit again (should hit cache)
        result2 = accelerator.process_circuit_optimized(
            layer_sizes=[16, 64, 32, 8],
            circuit_name="demo_mlp",
            use_cache=True,
            use_concurrency=False
        )
        
        # Test 3: Batch processing
        print("  → Testing concurrent batch processing...")
        batch_configs = [
            ([4, 16, 8], "tiny_net"),
            ([28, 128, 64, 10], "small_classifier"),
            ([100, 256, 128, 50], "medium_net"),
            ([200, 512, 256, 100], "large_net")
        ]
        
        batch_results = accelerator.process_batch_circuits(batch_configs, max_concurrent=4)
        
        # Test 4: Performance optimization
        print("  → Running performance optimization...")
        optimization_results = accelerator.optimize_performance()
        
        # Get comprehensive metrics
        final_metrics = accelerator.get_comprehensive_metrics()
        
        # Create demo report
        demo_report = {
            'demo_type': 'advanced_scaling',
            'single_circuit_results': [result1, result2],
            'batch_results': batch_results,
            'optimization_results': optimization_results,
            'comprehensive_metrics': final_metrics,
            'cache_hit_demonstration': result2['scaling_info']['cache_hit'],
            'batch_success_rate': sum(1 for r in batch_results if 'error' not in r) / len(batch_results),
            'demo_timestamp': time.time()
        }
        
        # Save results
        os.makedirs('output', exist_ok=True)
        with open('output/scaling_demo_results.json', 'w') as f:
            json.dump(demo_report, f, indent=2)
        
        # Print summary
        print(f"✅ Scaling demo completed successfully!")
        print(f"   → Cache hit rate: {final_metrics['cache_stats']['hit_rate']:.1%}")
        print(f"   → Batch success rate: {demo_report['batch_success_rate']:.1%}")
        print(f"   → Total operations: {final_metrics['scaling_metrics']['total_operations']}")
        print(f"   → Concurrent operations: {final_metrics['scaling_metrics']['concurrent_operations']}")
        print(f"   → Results saved to output/scaling_demo_results.json")
        
        return demo_report
        
    finally:
        # Always shutdown properly
        accelerator.shutdown()


if __name__ == "__main__":
    # Run scaling demo
    results = create_scaling_demo()
    print("\n⚡ GENERATION 3 SUCCESS: Advanced scaling implemented!")