"""
Enterprise Scaling & Optimization Engine

High-performance distributed system with intelligent caching, load balancing,
auto-scaling, and performance optimization for production deployments.
"""

import json
import time
import threading
import asyncio
import logging
import hashlib
import pickle
import sys
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import os
import traceback

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive_scaling"
    PREDICTIVE = "predictive_scaling"
    HYBRID = "hybrid_scaling"
    COST_OPTIMIZED = "cost_optimized_scaling"

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    PERFORMANCE_BASED = "performance_based"
    LOCALITY_AWARE = "locality_aware"

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive_policy"

@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    hostname: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    current_load: float
    active_tasks: int
    health_status: str
    last_heartbeat: float
    capabilities: List[str]
    performance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TaskRequest:
    """Represents a task request for processing."""
    task_id: str
    task_type: str
    priority: int
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    payload: Dict[str, Any]
    created_at: float
    deadline: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    disk_io: float
    active_connections: int
    tasks_per_second: float
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float

class IntelligentCache:
    """
    High-performance intelligent cache with adaptive policies and compression.
    """
    
    def __init__(self, max_size: int = 10000, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.ttl: Dict[str, float] = {}
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Adaptive policy parameters
        self.performance_history = deque(maxlen=1000)
        self.policy_weights = {
            CachePolicy.LRU: 0.4,
            CachePolicy.LFU: 0.3,
            CachePolicy.TTL: 0.3
        }
        
        logger.info(f"IntelligentCache initialized with policy {policy.value}, max_size {max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tracking."""
        with self.cache_lock:
            if key in self.cache:
                # Check TTL if applicable
                if key in self.ttl and time.time() > self.ttl[key]:
                    del self.cache[key]
                    del self.ttl[key]
                    self.miss_count += 1
                    return None
                
                # Update access patterns
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hit_count += 1
                
                # Decompress if needed
                value = self.cache[key]
                if isinstance(value, bytes):
                    try:
                        value = pickle.loads(value)
                    except:
                        pass
                
                return value
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache with intelligent eviction."""
        with self.cache_lock:
            # Compress large objects
            if sys.getsizeof(value) > 1024:  # 1KB threshold
                try:
                    compressed_value = pickle.dumps(value)
                    if len(compressed_value) < sys.getsizeof(value):
                        value = compressed_value
                except:
                    pass
            
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_items(1)
            
            # Store value
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            # Set TTL if specified
            if ttl:
                self.ttl[key] = time.time() + ttl
    
    def _evict_items(self, count: int) -> None:
        """Evict items based on adaptive policy."""
        if not self.cache:
            return
        
        current_time = time.time()
        candidates = []
        
        for key in self.cache.keys():
            score = self._calculate_eviction_score(key, current_time)
            candidates.append((score, key))
        
        # Sort by eviction score (higher score = more likely to evict)
        candidates.sort(reverse=True)
        
        # Evict top candidates
        for i in range(min(count, len(candidates))):
            _, key = candidates[i]
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.ttl:
                del self.ttl[key]
            self.eviction_count += 1
    
    def _calculate_eviction_score(self, key: str, current_time: float) -> float:
        """Calculate eviction score using adaptive policy."""
        scores = {}
        
        # LRU score (time since last access)
        last_access = self.access_times.get(key, 0)
        lru_score = current_time - last_access
        scores[CachePolicy.LRU] = lru_score
        
        # LFU score (inverse of access frequency)
        access_count = self.access_counts.get(key, 1)
        lfu_score = 1.0 / access_count
        scores[CachePolicy.LFU] = lfu_score
        
        # TTL score (time until expiration)
        if key in self.ttl:
            ttl_score = max(0, self.ttl[key] - current_time)
        else:
            ttl_score = 0
        scores[CachePolicy.TTL] = ttl_score
        
        # Weighted combination
        if self.policy == CachePolicy.ADAPTIVE:
            combined_score = sum(self.policy_weights[policy] * scores[policy] 
                               for policy in scores.keys())
        else:
            combined_score = scores.get(self.policy, lru_score)
        
        return combined_score
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.cache_lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': self.get_hit_rate(),
                'eviction_count': self.eviction_count,
                'policy': self.policy.value,
                'policy_weights': self.policy_weights.copy()
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.ttl.clear()

class LoadBalancer:
    """
    Intelligent load balancer with multiple algorithms and health monitoring.
    """
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.PERFORMANCE_BASED):
        self.algorithm = algorithm
        self.nodes: Dict[str, WorkerNode] = {}
        self.node_lock = threading.RLock()
        self.round_robin_index = 0
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info(f"LoadBalancer initialized with algorithm {algorithm.value}")
    
    def add_node(self, node: WorkerNode) -> None:
        """Add a worker node to the load balancer."""
        with self.node_lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added node {node.node_id} to load balancer")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a worker node from the load balancer."""
        with self.node_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                if node_id in self.performance_history:
                    del self.performance_history[node_id]
                logger.info(f"Removed node {node_id} from load balancer")
    
    def update_node_metrics(self, node_id: str, metrics: PerformanceMetrics) -> None:
        """Update performance metrics for a node."""
        with self.node_lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.current_load = metrics.cpu_usage
                node.last_heartbeat = time.time()
                node.health_status = "healthy" if metrics.error_rate < 0.05 else "degraded"
                
                # Update performance score
                performance_score = self._calculate_performance_score(metrics)
                node.performance_score = performance_score
                
                # Store performance history
                self.performance_history[node_id].append({
                    'timestamp': metrics.timestamp,
                    'score': performance_score,
                    'cpu_usage': metrics.cpu_usage,
                    'response_time': metrics.avg_response_time
                })
    
    def select_node(self, task: TaskRequest) -> Optional[WorkerNode]:
        """Select the best node for a task based on load balancing algorithm."""
        with self.node_lock:
            healthy_nodes = [node for node in self.nodes.values() 
                           if node.health_status == "healthy" and 
                           self._node_can_handle_task(node, task)]
            
            if not healthy_nodes:
                return None
            
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin_selection(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.PERFORMANCE_BASED:
                return self._performance_based_selection(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.LOCALITY_AWARE:
                return self._locality_aware_selection(healthy_nodes, task)
            else:
                return healthy_nodes[0]  # Fallback
    
    def _node_can_handle_task(self, node: WorkerNode, task: TaskRequest) -> bool:
        """Check if a node can handle a specific task."""
        # Check resource requirements
        req = task.resource_requirements
        
        if req.get('min_cpu_cores', 0) > node.cpu_cores:
            return False
        if req.get('min_memory_gb', 0) > node.memory_gb:
            return False
        if req.get('requires_gpu', False) and node.gpu_count == 0:
            return False
        
        # Check capabilities
        required_capabilities = req.get('capabilities', [])
        if not all(cap in node.capabilities for cap in required_capabilities):
            return False
        
        # Check current load
        if node.current_load > 0.9:  # 90% load threshold
            return False
        
        return True
    
    def _round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node using round-robin algorithm."""
        selected_node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected_node
    
    def _least_connections_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: n.active_tasks)
    
    def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node using weighted round-robin based on capacity."""
        # Weight by available capacity
        weights = []
        for node in nodes:
            available_capacity = (node.cpu_cores * node.memory_gb) * (1.0 - node.current_load)
            weights.append(max(available_capacity, 0.1))
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0]
        
        # Normalize weights and select
        import random
        r = random.random() * total_weight
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _performance_based_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on performance score."""
        return max(nodes, key=lambda n: n.performance_score)
    
    def _locality_aware_selection(self, nodes: List[WorkerNode], task: TaskRequest) -> WorkerNode:
        """Select node considering data locality and network proximity."""
        # Simplified locality awareness (in real implementation, consider network topology)
        preferred_location = task.payload.get('preferred_location', '')
        
        if preferred_location:
            local_nodes = [n for n in nodes if preferred_location in n.hostname]
            if local_nodes:
                return self._performance_based_selection(local_nodes)
        
        return self._performance_based_selection(nodes)
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate performance score for a node."""
        # Weighted combination of performance factors
        cpu_score = 1.0 - metrics.cpu_usage
        memory_score = 1.0 - metrics.memory_usage
        response_time_score = max(0, 1.0 - metrics.avg_response_time / 1000.0)  # Normalize to 1s
        error_score = 1.0 - metrics.error_rate
        
        # Weighted average
        performance_score = (
            0.3 * cpu_score +
            0.2 * memory_score +
            0.3 * response_time_score +
            0.2 * error_score
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        with self.node_lock:
            if not self.nodes:
                return {}
            
            total_nodes = len(self.nodes)
            healthy_nodes = len([n for n in self.nodes.values() if n.health_status == "healthy"])
            total_cpu_cores = sum(n.cpu_cores for n in self.nodes.values())
            total_memory = sum(n.memory_gb for n in self.nodes.values())
            total_gpus = sum(n.gpu_count for n in self.nodes.values())
            avg_load = sum(n.current_load for n in self.nodes.values()) / total_nodes
            total_active_tasks = sum(n.active_tasks for n in self.nodes.values())
            
            return {
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'cluster_health': healthy_nodes / total_nodes if total_nodes > 0 else 0,
                'total_cpu_cores': total_cpu_cores,
                'total_memory_gb': total_memory,
                'total_gpus': total_gpus,
                'average_load': avg_load,
                'total_active_tasks': total_active_tasks,
                'load_balancing_algorithm': self.algorithm.value
            }

class AutoScaler:
    """
    Intelligent auto-scaler with predictive scaling and cost optimization.
    """
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.load_balancer: Optional[LoadBalancer] = None
        self.scaling_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=1000)
        
        # Scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.min_nodes = 1
        self.max_nodes = 100
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_action = 0
        
        # Predictive parameters
        self.prediction_window = 600  # 10 minutes
        self.trend_sensitivity = 0.1
        
        logger.info(f"AutoScaler initialized with strategy {strategy.value}")
    
    def set_load_balancer(self, load_balancer: LoadBalancer) -> None:
        """Set the load balancer for scaling decisions."""
        self.load_balancer = load_balancer
    
    def evaluate_scaling(self, current_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Evaluate whether scaling action is needed."""
        if not self.load_balancer:
            return {'action': 'none', 'reason': 'No load balancer configured'}
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.cooldown_period:
            return {'action': 'none', 'reason': 'In cooldown period'}
        
        # Store current metrics
        self.metrics_history.append({
            'timestamp': current_time,
            'metrics': current_metrics
        })
        
        # Calculate cluster load
        cluster_stats = self.load_balancer.get_cluster_stats()
        current_load = cluster_stats.get('average_load', 0)
        current_nodes = cluster_stats.get('healthy_nodes', 0)
        
        # Apply scaling strategy
        if self.strategy == ScalingStrategy.REACTIVE:
            return self._reactive_scaling(current_load, current_nodes)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._predictive_scaling(current_load, current_nodes)
        elif self.strategy == ScalingStrategy.HYBRID:
            return self._hybrid_scaling(current_load, current_nodes)
        elif self.strategy == ScalingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_scaling(current_load, current_nodes)
        else:
            return {'action': 'none', 'reason': 'Unknown strategy'}
    
    def _reactive_scaling(self, current_load: float, current_nodes: int) -> Dict[str, Any]:
        """Reactive scaling based on current load."""
        if current_load > self.scale_up_threshold and current_nodes < self.max_nodes:
            nodes_to_add = min(2, self.max_nodes - current_nodes)
            return {
                'action': 'scale_up',
                'nodes_to_add': nodes_to_add,
                'reason': f'High load: {current_load:.2f} > {self.scale_up_threshold}',
                'strategy': 'reactive'
            }
        elif current_load < self.scale_down_threshold and current_nodes > self.min_nodes:
            nodes_to_remove = min(1, current_nodes - self.min_nodes)
            return {
                'action': 'scale_down',
                'nodes_to_remove': nodes_to_remove,
                'reason': f'Low load: {current_load:.2f} < {self.scale_down_threshold}',
                'strategy': 'reactive'
            }
        else:
            return {'action': 'none', 'reason': 'Load within acceptable range'}
    
    def _predictive_scaling(self, current_load: float, current_nodes: int) -> Dict[str, Any]:
        """Predictive scaling based on load trends."""
        if len(self.metrics_history) < 5:
            return self._reactive_scaling(current_load, current_nodes)
        
        # Calculate load trend
        recent_loads = [entry['metrics'] for entry in list(self.metrics_history)[-5:]]
        load_values = []
        
        for metrics_dict in recent_loads:
            avg_load = sum(m.cpu_usage for m in metrics_dict.values()) / len(metrics_dict) if metrics_dict else 0
            load_values.append(avg_load)
        
        # Simple linear trend calculation
        if len(load_values) >= 2:
            trend = (load_values[-1] - load_values[0]) / (len(load_values) - 1)
        else:
            trend = 0
        
        # Predict future load
        predicted_load = current_load + trend * 5  # 5-step prediction
        
        if predicted_load > self.scale_up_threshold and current_nodes < self.max_nodes:
            nodes_to_add = min(2, self.max_nodes - current_nodes)
            return {
                'action': 'scale_up',
                'nodes_to_add': nodes_to_add,
                'reason': f'Predicted high load: {predicted_load:.2f}',
                'strategy': 'predictive',
                'trend': trend
            }
        elif predicted_load < self.scale_down_threshold and current_nodes > self.min_nodes:
            nodes_to_remove = min(1, current_nodes - self.min_nodes)
            return {
                'action': 'scale_down',
                'nodes_to_remove': nodes_to_remove,
                'reason': f'Predicted low load: {predicted_load:.2f}',
                'strategy': 'predictive',
                'trend': trend
            }
        else:
            return {'action': 'none', 'reason': 'Predicted load within range'}
    
    def _hybrid_scaling(self, current_load: float, current_nodes: int) -> Dict[str, Any]:
        """Hybrid scaling combining reactive and predictive approaches."""
        reactive_decision = self._reactive_scaling(current_load, current_nodes)
        predictive_decision = self._predictive_scaling(current_load, current_nodes)
        
        # If both suggest scaling in same direction, be more aggressive
        if (reactive_decision['action'] == 'scale_up' and 
            predictive_decision['action'] == 'scale_up'):
            nodes_to_add = min(
                reactive_decision.get('nodes_to_add', 1) + 1,
                self.max_nodes - current_nodes
            )
            return {
                'action': 'scale_up',
                'nodes_to_add': nodes_to_add,
                'reason': 'Both reactive and predictive suggest scale up',
                'strategy': 'hybrid'
            }
        elif (reactive_decision['action'] == 'scale_down' and 
              predictive_decision['action'] == 'scale_down'):
            nodes_to_remove = min(
                reactive_decision.get('nodes_to_remove', 1),
                current_nodes - self.min_nodes
            )
            return {
                'action': 'scale_down',
                'nodes_to_remove': nodes_to_remove,
                'reason': 'Both reactive and predictive suggest scale down',
                'strategy': 'hybrid'
            }
        # If only one suggests scaling, use reactive (more conservative)
        elif reactive_decision['action'] != 'none':
            return reactive_decision
        else:
            return {'action': 'none', 'reason': 'No consensus on scaling action'}
    
    def _cost_optimized_scaling(self, current_load: float, current_nodes: int) -> Dict[str, Any]:
        """Cost-optimized scaling considering resource efficiency."""
        # Calculate cost efficiency (simplified)
        if current_nodes == 0:
            efficiency = 0
        else:
            efficiency = current_load / current_nodes
        
        # More conservative scaling for cost optimization
        scale_up_threshold = self.scale_up_threshold + 0.1
        scale_down_threshold = self.scale_down_threshold - 0.1
        
        if current_load > scale_up_threshold and efficiency > 0.7 and current_nodes < self.max_nodes:
            # Only scale up if efficiency is good
            nodes_to_add = 1  # Conservative scaling
            return {
                'action': 'scale_up',
                'nodes_to_add': nodes_to_add,
                'reason': f'High load with good efficiency: {efficiency:.2f}',
                'strategy': 'cost_optimized'
            }
        elif current_load < scale_down_threshold and current_nodes > self.min_nodes:
            nodes_to_remove = 1
            return {
                'action': 'scale_down',
                'nodes_to_remove': nodes_to_remove,
                'reason': f'Low load, improving efficiency',
                'strategy': 'cost_optimized'
            }
        else:
            return {'action': 'none', 'reason': 'Cost optimization - no action needed'}
    
    def record_scaling_action(self, action: Dict[str, Any]) -> None:
        """Record a scaling action in history."""
        self.last_scaling_action = time.time()
        self.scaling_history.append({
            'timestamp': self.last_scaling_action,
            'action': action
        })
        logger.info(f"Scaling action recorded: {action}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and history."""
        if not self.scaling_history:
            return {
                'total_scaling_actions': 0,
                'scale_up_actions': 0,
                'scale_down_actions': 0,
                'strategy': self.strategy.value
            }
        
        scale_up_count = sum(1 for entry in self.scaling_history 
                           if entry['action'].get('action') == 'scale_up')
        scale_down_count = sum(1 for entry in self.scaling_history 
                             if entry['action'].get('action') == 'scale_down')
        
        return {
            'total_scaling_actions': len(self.scaling_history),
            'scale_up_actions': scale_up_count,
            'scale_down_actions': scale_down_count,
            'last_scaling_action': self.last_scaling_action,
            'strategy': self.strategy.value,
            'recent_actions': list(self.scaling_history)[-5:]  # Last 5 actions
        }

class DistributedTaskExecutor:
    """
    High-performance distributed task execution engine with intelligent scheduling.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.cache = IntelligentCache(max_size=10000)
        
        # Task management
        self.pending_tasks: Dict[str, TaskRequest] = {}
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.RLock()
        
        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers//2)
        
        # Monitoring
        self.metrics_collector = threading.Thread(target=self._collect_metrics, daemon=True)
        self.is_running = True
        
        # Performance tracking
        self.task_completion_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_tasks = 0
        
        # Initialize components
        self.auto_scaler.set_load_balancer(self.load_balancer)
        
        logger.info(f"DistributedTaskExecutor initialized with {max_workers} workers")
    
    def start(self) -> None:
        """Start the distributed task executor."""
        self.metrics_collector.start()
        logger.info("DistributedTaskExecutor started")
    
    def stop(self) -> None:
        """Stop the distributed task executor."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("DistributedTaskExecutor stopped")
    
    def add_worker_node(self, node: WorkerNode) -> None:
        """Add a worker node to the cluster."""
        self.load_balancer.add_node(node)
    
    def remove_worker_node(self, node_id: str) -> None:
        """Remove a worker node from the cluster."""
        self.load_balancer.remove_node(node_id)
    
    async def submit_task(self, task: TaskRequest) -> str:
        """Submit a task for distributed execution."""
        task_id = task.task_id
        
        with self.task_lock:
            self.pending_tasks[task_id] = task
            self.total_tasks += 1
        
        # Try to execute immediately
        await self._schedule_task(task)
        
        logger.info(f"Task {task_id} submitted for execution")
        return task_id
    
    async def _schedule_task(self, task: TaskRequest) -> None:
        """Schedule a task for execution on an appropriate node."""
        # Check cache first
        cache_key = self._generate_cache_key(task)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            await self._complete_task(task.task_id, cached_result, from_cache=True)
            return
        
        # Select appropriate worker node
        selected_node = self.load_balancer.select_node(task)
        
        if not selected_node:
            # No available nodes, task remains pending
            logger.warning(f"No available nodes for task {task.task_id}")
            return
        
        # Execute task
        with self.task_lock:
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
                self.running_tasks[task.task_id] = {
                    'task': task,
                    'node': selected_node,
                    'start_time': time.time()
                }
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_task, task, selected_node)
        
        # Handle completion
        def handle_completion(fut):
            try:
                result = fut.result()
                asyncio.create_task(self._complete_task(task.task_id, result))
            except Exception as e:
                asyncio.create_task(self._handle_task_error(task.task_id, e))
        
        future.add_done_callback(handle_completion)
    
    def _execute_task(self, task: TaskRequest, node: WorkerNode) -> Any:
        """Execute a task on a specific node."""
        start_time = time.time()
        
        try:
            # Simulate task execution (in real implementation, this would dispatch to actual node)
            result = self._simulate_task_execution(task, node)
            
            # Cache result if appropriate
            if task.task_type in ['computation', 'analysis']:
                cache_key = self._generate_cache_key(task)
                self.cache.put(cache_key, result, ttl=3600)  # 1 hour TTL
            
            execution_time = time.time() - start_time
            logger.info(f"Task {task.task_id} completed in {execution_time:.3f}s on node {node.node_id}")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Task {task.task_id} failed on node {node.node_id}: {e}")
            raise
    
    def _simulate_task_execution(self, task: TaskRequest, node: WorkerNode) -> Dict[str, Any]:
        """Simulate task execution (replace with actual implementation)."""
        import random
        import time
        
        # Simulate processing time based on task complexity
        base_time = task.estimated_duration
        actual_time = base_time * random.uniform(0.8, 1.2)  # ±20% variance
        
        # Simulate node performance impact
        performance_factor = node.performance_score
        adjusted_time = actual_time / max(performance_factor, 0.1)
        
        time.sleep(min(adjusted_time, 0.1))  # Cap at 100ms for demo
        
        # Generate simulated result
        result = {
            'task_id': task.task_id,
            'status': 'completed',
            'execution_time': adjusted_time,
            'node_id': node.node_id,
            'result_data': {
                'computation_result': random.random(),
                'metrics': {
                    'cpu_time': adjusted_time * 0.8,
                    'memory_used': random.uniform(100, 1000),  # MB
                    'accuracy': random.uniform(0.9, 0.99)
                }
            },
            'timestamp': time.time()
        }
        
        return result
    
    async def _complete_task(self, task_id: str, result: Any, from_cache: bool = False) -> None:
        """Handle task completion."""
        with self.task_lock:
            if task_id in self.running_tasks:
                task_info = self.running_tasks[task_id]
                execution_time = time.time() - task_info['start_time']
                
                self.completed_tasks[task_id] = {
                    'task': task_info['task'],
                    'result': result,
                    'execution_time': execution_time,
                    'from_cache': from_cache,
                    'completed_at': time.time()
                }
                
                del self.running_tasks[task_id]
                
                if not from_cache:
                    self.task_completion_times.append(execution_time)
            
            elif task_id in self.pending_tasks:
                # Task completed from cache before scheduling
                task = self.pending_tasks[task_id]
                self.completed_tasks[task_id] = {
                    'task': task,
                    'result': result,
                    'execution_time': 0.0,
                    'from_cache': True,
                    'completed_at': time.time()
                }
                del self.pending_tasks[task_id]
    
    async def _handle_task_error(self, task_id: str, error: Exception) -> None:
        """Handle task execution error."""
        with self.task_lock:
            if task_id in self.running_tasks:
                task_info = self.running_tasks[task_id]
                
                self.completed_tasks[task_id] = {
                    'task': task_info['task'],
                    'result': None,
                    'error': str(error),
                    'execution_time': time.time() - task_info['start_time'],
                    'from_cache': False,
                    'completed_at': time.time()
                }
                
                del self.running_tasks[task_id]
    
    def _generate_cache_key(self, task: TaskRequest) -> str:
        """Generate cache key for a task."""
        # Create deterministic key based on task content
        content = {
            'task_type': task.task_type,
            'payload': task.payload
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _collect_metrics(self) -> None:
        """Collect performance metrics continuously."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Create performance metrics for each node
                node_metrics = {}
                for node_id, node in self.load_balancer.nodes.items():
                    # Simulate metrics collection (in real implementation, get from actual nodes)
                    metrics = PerformanceMetrics(
                        timestamp=current_time,
                        cpu_usage=node.current_load,
                        memory_usage=min(0.9, node.current_load + 0.1),
                        gpu_usage=node.current_load * 0.8 if node.gpu_count > 0 else 0,
                        network_io=50.0 + node.current_load * 100,
                        disk_io=10.0 + node.current_load * 20,
                        active_connections=node.active_tasks,
                        tasks_per_second=len(self.completed_tasks) / max(current_time - 3600, 1),
                        avg_response_time=sum(self.task_completion_times) / len(self.task_completion_times) if self.task_completion_times else 0,
                        error_rate=self.error_count / max(self.total_tasks, 1),
                        cache_hit_rate=self.cache.get_hit_rate()
                    )
                    node_metrics[node_id] = metrics
                    
                    # Update load balancer with metrics
                    self.load_balancer.update_node_metrics(node_id, metrics)
                
                # Evaluate auto-scaling
                scaling_decision = self.auto_scaler.evaluate_scaling(node_metrics)
                if scaling_decision['action'] != 'none':
                    self.auto_scaler.record_scaling_action(scaling_decision)
                    # In real implementation, execute scaling action
                    logger.info(f"Auto-scaling decision: {scaling_decision}")
                
                time.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(30)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the distributed executor."""
        with self.task_lock:
            cluster_stats = self.load_balancer.get_cluster_stats()
            cache_stats = self.cache.get_stats()
            scaling_stats = self.auto_scaler.get_scaling_stats()
            
            avg_completion_time = (sum(self.task_completion_times) / len(self.task_completion_times) 
                                 if self.task_completion_times else 0)
            
            return {
                'cluster': cluster_stats,
                'cache': cache_stats,
                'scaling': scaling_stats,
                'tasks': {
                    'total_submitted': self.total_tasks,
                    'pending': len(self.pending_tasks),
                    'running': len(self.running_tasks),
                    'completed': len(self.completed_tasks),
                    'error_count': self.error_count,
                    'error_rate': self.error_count / max(self.total_tasks, 1),
                    'avg_completion_time': avg_completion_time
                },
                'performance': {
                    'cache_hit_rate': cache_stats['hit_rate'],
                    'avg_task_duration': avg_completion_time,
                    'tasks_per_minute': len(self.task_completion_times),
                    'cluster_utilization': cluster_stats.get('average_load', 0)
                }
            }

# Factory functions
def create_enterprise_executor(max_workers: int = 20, 
                             scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
                             load_balancing: LoadBalancingAlgorithm = LoadBalancingAlgorithm.PERFORMANCE_BASED) -> DistributedTaskExecutor:
    """Create an enterprise-grade distributed task executor."""
    executor = DistributedTaskExecutor(max_workers=max_workers)
    executor.load_balancer.algorithm = load_balancing
    executor.auto_scaler.strategy = scaling_strategy
    
    return executor

def create_sample_cluster(num_nodes: int = 5) -> List[WorkerNode]:
    """Create sample worker nodes for testing."""
    nodes = []
    
    for i in range(num_nodes):
        node = WorkerNode(
            node_id=f"worker_{i:03d}",
            hostname=f"worker-{i:03d}.cluster.local",
            cpu_cores=8 + (i % 4) * 2,  # 8-14 cores
            memory_gb=32 + (i % 3) * 16,  # 32-64 GB
            gpu_count=1 if i % 2 == 0 else 0,  # Every other node has GPU
            current_load=0.1 + (i * 0.1) % 0.8,  # 10-80% load
            active_tasks=i % 5,
            health_status="healthy",
            last_heartbeat=time.time(),
            capabilities=["cpu_compute", "gpu_compute" if i % 2 == 0 else "cpu_only"][0:1 + (i % 2)],
            performance_score=0.7 + (i * 0.05) % 0.3  # 0.7-1.0
        )
        nodes.append(node)
    
    return nodes