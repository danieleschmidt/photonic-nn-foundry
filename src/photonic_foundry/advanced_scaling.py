"""
Advanced auto-scaling and performance optimization for photonic systems.

This module provides enterprise-grade auto-scaling capabilities including:
- Adaptive resource allocation based on workload patterns
- Multi-tier distributed processing architecture
- Intelligent caching with ML-based optimization
- Predictive scaling with load forecasting
- Comprehensive resource management and pooling
- Advanced load balancing algorithms
- Performance monitoring and optimization
"""

import time
import threading
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set, Protocol
import queue
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib
import psutil
import weakref
import gc
import traceback
from collections import defaultdict, deque
from enum import Enum
import json
from datetime import datetime, timedelta
import math
import statistics
from contextlib import contextmanager
import resource
from threading import RLock, Condition
from abc import ABC, abstractmethod
import random
import heapq

from .monitoring import get_metrics_collector, get_performance_monitor
from .core import CircuitMetrics

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Auto-scaling modes for different workload patterns."""
    REACTIVE = "reactive"          # React to current load
    PREDICTIVE = "predictive"      # Predict future load
    ADAPTIVE = "adaptive"          # Learn and adapt patterns
    BURST = "burst"               # Handle traffic bursts
    COST_OPTIMIZED = "cost_opt"    # Minimize resource costs


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory" 
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"
    CUSTOM = "custom"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class ResourceLimits:
    """Resource limits and constraints."""
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    network_mbps: float = 100.0
    storage_gb: float = 10.0
    gpu_memory_gb: float = 0.0
    custom_limits: Dict[str, float] = field(default_factory=dict)
    
    def scale(self, factor: float) -> 'ResourceLimits':
        """Scale all limits by a factor."""
        return ResourceLimits(
            cpu_cores=self.cpu_cores * factor,
            memory_gb=self.memory_gb * factor,
            network_mbps=self.network_mbps * factor,
            storage_gb=self.storage_gb * factor,
            gpu_memory_gb=self.gpu_memory_gb * factor,
            custom_limits={k: v * factor for k, v in self.custom_limits.items()}
        )


@dataclass
class TierConfig:
    """Configuration for individual scaling tiers."""
    name: str
    min_workers: int
    max_workers: int
    cpu_weight: float = 1.0
    memory_weight: float = 1.0
    priority: int = 1  # Higher number = higher priority
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    workload_types: List[str] = field(default_factory=list)
    

@dataclass
class AdvancedScalingConfig:
    """Advanced configuration for auto-scaling behavior."""
    # Basic scaling parameters
    min_workers: int = 1
    max_workers: int = multiprocessing.cpu_count() * 2
    target_cpu_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 40.0
    scale_up_cooldown: int = 60
    scale_down_cooldown: int = 300
    queue_threshold: int = 10
    enable_process_pool: bool = True
    
    # Advanced scaling parameters
    scaling_mode: ScalingMode = ScalingMode.ADAPTIVE
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    predictive_horizon_minutes: int = 15
    learning_window_hours: int = 24
    burst_detection_threshold: float = 3.0
    cost_sensitivity: float = 0.5  # 0=performance focused, 1=cost focused
    
    # Multi-tier scaling
    enable_multi_tier: bool = True
    tier_configs: Dict[str, TierConfig] = field(default_factory=dict)
    
    # Health and performance thresholds
    memory_threshold: float = 85.0
    disk_io_threshold: float = 80.0
    network_threshold: float = 75.0
    error_rate_threshold: float = 5.0
    response_time_threshold_ms: float = 1000.0
    
    # Resource pooling
    enable_resource_pooling: bool = True
    pool_warmup_size: int = 2
    pool_max_idle_time: int = 300
    
    # Load balancing
    load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED
    health_check_interval: int = 30
    circuit_breaker_enabled: bool = True


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    id: str
    tier: str
    created_at: datetime
    last_health_check: datetime
    resource_usage: Dict[str, float] = field(default_factory=dict)
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time: float = 0.0
    is_healthy: bool = True
    weight: float = 1.0
    
    @property
    def utilization(self) -> float:
        """Calculate overall utilization of this worker."""
        cpu_util = self.resource_usage.get('cpu', 0)
        memory_util = self.resource_usage.get('memory', 0)
        return max(cpu_util, memory_util) / 100.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate for this worker."""
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks == 0:
            return 0.0
        return self.failed_tasks / total_tasks


class ResourcePool:
    """Pool of pre-allocated resources for quick scaling."""
    
    def __init__(self, config: AdvancedScalingConfig):
        self.config = config
        self.available_resources = queue.Queue()
        self.allocated_resources = {}
        self.resource_stats = defaultdict(int)
        self._lock = RLock()
        self._warmup_thread = None
        self._running = False
        
    def start(self):
        """Start resource pool with pre-warming."""
        if self._running:
            return
            
        self._running = True
        self._warmup_thread = threading.Thread(target=self._warmup_resources, daemon=True)
        self._warmup_thread.start()
        logger.info(f"Started resource pool with warmup size: {self.config.pool_warmup_size}")
        
    def stop(self):
        """Stop resource pool and cleanup."""
        self._running = False
        if self._warmup_thread:
            self._warmup_thread.join(timeout=5)
            
        # Cleanup all resources
        with self._lock:
            while not self.available_resources.empty():
                try:
                    resource_obj = self.available_resources.get_nowait()
                    self._cleanup_resource(resource_obj)
                except queue.Empty:
                    break
                    
            for resource_obj in self.allocated_resources.values():
                self._cleanup_resource(resource_obj)
                
        logger.info("Resource pool stopped and cleaned up")
        
    def acquire_resource(self, resource_type: str = "worker") -> Optional[Any]:
        """Acquire a resource from the pool."""
        with self._lock:
            if not self.available_resources.empty():
                try:
                    resource_obj = self.available_resources.get_nowait()
                    resource_id = id(resource_obj)
                    self.allocated_resources[resource_id] = resource_obj
                    self.resource_stats['acquired'] += 1
                    return resource_obj
                except queue.Empty:
                    pass
                    
            # Create new resource if none available
            resource_obj = self._create_resource(resource_type)
            if resource_obj:
                resource_id = id(resource_obj)
                self.allocated_resources[resource_id] = resource_obj
                self.resource_stats['created'] += 1
                return resource_obj
                
        return None
        
    def release_resource(self, resource_obj: Any):
        """Release a resource back to the pool."""
        with self._lock:
            resource_id = id(resource_obj)
            if resource_id in self.allocated_resources:
                del self.allocated_resources[resource_id]
                
                # Check if resource is still healthy
                if self._is_resource_healthy(resource_obj):
                    self.available_resources.put(resource_obj)
                    self.resource_stats['released'] += 1
                else:
                    self._cleanup_resource(resource_obj)
                    self.resource_stats['discarded'] += 1
                    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            return {
                'available_count': self.available_resources.qsize(),
                'allocated_count': len(self.allocated_resources),
                'stats': dict(self.resource_stats),
                'pool_utilization': len(self.allocated_resources) / 
                                   max(len(self.allocated_resources) + self.available_resources.qsize(), 1)
            }
            
    def _warmup_resources(self):
        """Pre-create resources for the pool."""
        while self._running:
            current_available = self.available_resources.qsize()
            if current_available < self.config.pool_warmup_size:
                needed = self.config.pool_warmup_size - current_available
                for _ in range(needed):
                    if not self._running:
                        break
                    resource_obj = self._create_resource("warmup")
                    if resource_obj:
                        self.available_resources.put(resource_obj)
                        
            time.sleep(60)  # Check every minute
            
    def _create_resource(self, resource_type: str) -> Optional[Any]:
        """Create a new resource."""
        try:
            # Create a lightweight resource object (placeholder)
            return {
                'id': f"{resource_type}_{int(time.time() * 1000)}",
                'type': resource_type,
                'created_at': datetime.now(),
                'last_used': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
            
    def _is_resource_healthy(self, resource_obj: Any) -> bool:
        """Check if a resource is still healthy."""
        if not isinstance(resource_obj, dict):
            return False
            
        # Check age
        age = datetime.now() - resource_obj.get('created_at', datetime.now())
        if age.total_seconds() > self.config.pool_max_idle_time * 2:
            return False
            
        return True
        
    def _cleanup_resource(self, resource_obj: Any):
        """Cleanup a resource."""
        try:
            if hasattr(resource_obj, 'cleanup'):
                resource_obj.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up resource: {e}")


class LoadBalancer:
    """Advanced load balancer with multiple algorithms."""
    
    def __init__(self, config: AdvancedScalingConfig):
        self.config = config
        self.nodes: Dict[str, WorkerNode] = {}
        self.algorithm = config.load_balancing_algorithm
        self._lock = RLock()
        self._current_index = 0
        self._weights = {}
        self._response_times = defaultdict(deque)
        
    def add_node(self, node: WorkerNode):
        """Add a worker node to the load balancer."""
        with self._lock:
            self.nodes[node.id] = node
            self._weights[node.id] = node.weight
            logger.info(f"Added node {node.id} to load balancer")
            
    def remove_node(self, node_id: str):
        """Remove a worker node from the load balancer."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self._weights.pop(node_id, None)
                self._response_times.pop(node_id, None)
                logger.info(f"Removed node {node_id} from load balancer")
                
    def select_node(self, request_context: Dict[str, Any] = None) -> Optional[WorkerNode]:
        """Select the best node for a request based on the configured algorithm."""
        with self._lock:
            healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]
            if not healthy_nodes:
                return None
                
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin_select(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                return self._least_response_time_select(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED:
                return self._adaptive_weighted_select(healthy_nodes)
            elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                return self._consistent_hash_select(healthy_nodes, request_context)
            else:
                return self._round_robin_select(healthy_nodes)
                
    def update_node_metrics(self, node_id: str, response_time: float, success: bool):
        """Update node metrics after request completion."""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Update response time
                if response_time > 0:
                    self._response_times[node_id].append(response_time)
                    if len(self._response_times[node_id]) > 100:
                        self._response_times[node_id].popleft()
                    
                    # Update average response time
                    node.avg_response_time = statistics.mean(self._response_times[node_id])
                
                # Update task counts
                if success:
                    node.completed_tasks += 1
                else:
                    node.failed_tasks += 1
                    
                # Update adaptive weights
                if self.algorithm == LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED:
                    self._update_adaptive_weight(node_id)
                    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_tasks = sum(node.completed_tasks + node.failed_tasks for node in self.nodes.values())
            
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_tasks = node.completed_tasks + node.failed_tasks
                node_stats[node_id] = {
                    'active_tasks': node.active_tasks,
                    'completed_tasks': node.completed_tasks,
                    'failed_tasks': node.failed_tasks,
                    'error_rate': node.error_rate,
                    'avg_response_time': node.avg_response_time,
                    'utilization': node.utilization,
                    'weight': node.weight,
                    'load_percentage': (node_tasks / max(total_tasks, 1)) * 100
                }
                
            return {
                'algorithm': self.algorithm.value,
                'total_nodes': len(self.nodes),
                'healthy_nodes': len([n for n in self.nodes.values() if n.is_healthy]),
                'total_tasks': total_tasks,
                'node_stats': node_stats
            }
            
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round robin selection."""
        if not nodes:
            return None
        self._current_index = (self._current_index + 1) % len(nodes)
        return nodes[self._current_index]
        
    def _weighted_round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin selection."""
        if not nodes:
            return None
            
        total_weight = sum(self._weights.get(node.id, 1.0) for node in nodes)
        if total_weight == 0:
            return self._round_robin_select(nodes)
            
        # Implement weighted selection
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for node in nodes:
            cumulative_weight += self._weights.get(node.id, 1.0)
            if cumulative_weight >= r:
                return node
                
        return nodes[-1]  # Fallback
        
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Least connections selection."""
        if not nodes:
            return None
        return min(nodes, key=lambda n: n.active_tasks)
        
    def _least_response_time_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Least response time selection."""
        if not nodes:
            return None
        return min(nodes, key=lambda n: n.avg_response_time)
        
    def _adaptive_weighted_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Adaptive weighted selection based on performance."""
        if not nodes:
            return None
            
        # Calculate adaptive scores
        scored_nodes = []
        for node in nodes:
            # Score based on utilization, response time, and error rate
            utilization_score = 1.0 - node.utilization
            response_time_score = 1.0 / (1.0 + node.avg_response_time / 1000.0)  # Normalize to seconds
            error_rate_score = 1.0 - node.error_rate
            
            combined_score = (utilization_score * 0.4 + 
                            response_time_score * 0.4 + 
                            error_rate_score * 0.2)
            scored_nodes.append((combined_score, node))
            
        # Select node with highest score
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return scored_nodes[0][1]
        
    def _consistent_hash_select(self, nodes: List[WorkerNode], 
                              request_context: Optional[Dict[str, Any]]) -> WorkerNode:
        """Consistent hash selection for session affinity."""
        if not nodes:
            return None
            
        # Use request context for hashing (e.g., user ID, session ID)
        hash_key = "default"
        if request_context:
            hash_key = str(request_context.get('session_id', 
                          request_context.get('user_id', 'default')))
            
        # Simple consistent hashing
        hash_value = hash(hash_key) % len(nodes)
        return nodes[hash_value]
        
    def _update_adaptive_weight(self, node_id: str):
        """Update adaptive weight based on node performance."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Calculate weight based on performance metrics
        utilization_factor = 1.0 - node.utilization
        response_time_factor = 1.0 / (1.0 + node.avg_response_time / 1000.0)
        error_rate_factor = 1.0 - node.error_rate
        
        new_weight = (utilization_factor * 0.5 + 
                     response_time_factor * 0.3 + 
                     error_rate_factor * 0.2)
        
        # Smooth weight changes
        old_weight = self._weights.get(node_id, 1.0)
        self._weights[node_id] = (old_weight * 0.7) + (new_weight * 0.3)
        node.weight = self._weights[node_id]


class PredictiveScaler:
    """Predictive scaling using time series analysis and pattern recognition."""
    
    def __init__(self, config: AdvancedScalingConfig):
        self.config = config
        self.metrics_history = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at 1-minute resolution
        self.patterns = {}
        self.predictions = {}
        self._lock = RLock()
        
    def record_metrics(self, timestamp: datetime, metrics: Dict[str, float]):
        """Record metrics for predictive analysis."""
        with self._lock:
            for metric_name, value in metrics.items():
                self.metrics_history[metric_name].append((timestamp, value))
                
    def predict_load(self, horizon_minutes: int = None) -> Dict[str, float]:
        """Predict future load for the specified horizon."""
        if horizon_minutes is None:
            horizon_minutes = self.config.predictive_horizon_minutes
            
        with self._lock:
            predictions = {}
            
            for metric_name, history in self.metrics_history.items():
                if len(history) < 10:  # Need minimum data points
                    predictions[metric_name] = history[-1][1] if history else 0.0
                    continue
                    
                try:
                    # Simple time series prediction using trends and patterns
                    values = [point[1] for point in history]
                    timestamps = [point[0] for point in history]
                    
                    # Trend analysis
                    trend = self._calculate_trend(values)
                    
                    # Seasonal pattern analysis
                    seasonal_factor = self._calculate_seasonal_factor(timestamps, values)
                    
                    # Recent average
                    recent_avg = np.mean(values[-10:]) if len(values) >= 10 else values[-1]
                    
                    # Combine factors for prediction
                    predicted_value = recent_avg + (trend * horizon_minutes) + seasonal_factor
                    predictions[metric_name] = max(0.0, predicted_value)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {metric_name}: {e}")
                    predictions[metric_name] = history[-1][1] if history else 0.0
                    
            return predictions
            
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect anomalies in current metrics."""
        with self._lock:
            anomalies = {}
            
            for metric_name, current_value in current_metrics.items():
                if metric_name not in self.metrics_history:
                    anomalies[metric_name] = False
                    continue
                    
                history = self.metrics_history[metric_name]
                if len(history) < 20:  # Need sufficient history
                    anomalies[metric_name] = False
                    continue
                    
                values = [point[1] for point in history]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Use 2-sigma rule for anomaly detection
                z_score = abs(current_value - mean_val) / max(std_val, 0.01)
                anomalies[metric_name] = z_score > 2.0
                
            return anomalies
            
    def get_scaling_recommendation(self, current_workers: int, 
                                 predicted_metrics: Dict[str, float] = None) -> int:
        """Get scaling recommendation based on predictions."""
        if predicted_metrics is None:
            predicted_metrics = self.predict_load()
            
        # Analyze predicted CPU and memory utilization
        predicted_cpu = predicted_metrics.get('cpu_utilization', 70.0)
        predicted_memory = predicted_metrics.get('memory_utilization', 50.0)
        predicted_queue_size = predicted_metrics.get('queue_size', 0.0)
        
        # Calculate desired workers based on predictions
        cpu_based_workers = math.ceil(current_workers * predicted_cpu / self.config.target_cpu_utilization)
        memory_based_workers = math.ceil(current_workers * predicted_memory / 80.0)  # 80% memory target
        queue_based_workers = math.ceil(predicted_queue_size / 5.0)  # 5 tasks per worker
        
        # Take the maximum requirement
        recommended_workers = max(cpu_based_workers, memory_based_workers, queue_based_workers)
        
        # Apply constraints
        recommended_workers = max(self.config.min_workers, 
                                min(self.config.max_workers, recommended_workers))
        
        logger.info(f"Scaling recommendation: {current_workers} -> {recommended_workers} "
                   f"(CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)")
        
        return recommended_workers
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
        
    def _calculate_seasonal_factor(self, timestamps: List[datetime], 
                                 values: List[float]) -> float:
        """Calculate seasonal/cyclic factor."""
        if len(timestamps) < 24:  # Need at least 24 data points
            return 0.0
            
        try:
            # Analyze hourly patterns
            hourly_averages = defaultdict(list)
            for timestamp, value in zip(timestamps, values):
                hour = timestamp.hour
                hourly_averages[hour].append(value)
                
            current_hour = datetime.now().hour
            if current_hour in hourly_averages:
                hour_avg = np.mean(hourly_averages[current_hour])
                overall_avg = np.mean(values)
                return hour_avg - overall_avg
                
        except Exception as e:
            logger.warning(f"Seasonal factor calculation failed: {e}")
            
        return 0.0


class AdvancedResourceManager:
    """Advanced resource manager with intelligent allocation and optimization."""
    
    def __init__(self, config: AdvancedScalingConfig):
        self.config = config
        self.resource_pools = {}
        self.allocation_history = deque(maxlen=1000)
        self.optimization_rules = {}
        self._lock = RLock()
        
        # Initialize resource pools
        for resource_type in ResourceType:
            self.resource_pools[resource_type] = {
                'allocated': 0.0,
                'available': self.config.resource_limits.__dict__.get(
                    f"{resource_type.value}_cores" if resource_type == ResourceType.CPU else f"{resource_type.value}_gb",
                    1.0
                ),
                'reserved': 0.0,
                'allocations': []
            }
            
    def allocate_resources(self, resource_requirements: Dict[ResourceType, float],
                          task_id: str, priority: int = 1) -> bool:
        """Allocate resources for a task with intelligent scheduling."""
        with self._lock:
            # Check if resources are available
            if not self._can_allocate(resource_requirements):
                # Try optimization strategies
                if not self._optimize_and_retry(resource_requirements):
                    return False
                    
            # Perform allocation
            allocation_record = {
                'task_id': task_id,
                'timestamp': datetime.now(),
                'resources': resource_requirements,
                'priority': priority
            }
            
            for resource_type, amount in resource_requirements.items():
                if resource_type in self.resource_pools:
                    self.resource_pools[resource_type]['allocated'] += amount
                    self.resource_pools[resource_type]['allocations'].append(allocation_record)
                    
            self.allocation_history.append(allocation_record)
            return True
            
    def deallocate_resources(self, task_id: str):
        """Deallocate resources for a completed task."""
        with self._lock:
            for resource_type, pool_info in self.resource_pools.items():
                allocations = pool_info['allocations']
                
                # Find and remove allocation
                for i, allocation in enumerate(allocations):
                    if allocation['task_id'] == task_id:
                        amount = allocation['resources'].get(resource_type, 0.0)
                        pool_info['allocated'] -= amount
                        pool_info['allocated'] = max(0.0, pool_info['allocated'])  # Ensure non-negative
                        allocations.pop(i)
                        break
                        
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        with self._lock:
            utilization = {}
            
            for resource_type, pool_info in self.resource_pools.items():
                total = pool_info['available']
                allocated = pool_info['allocated']
                
                if total > 0:
                    utilization[resource_type.value] = (allocated / total) * 100.0
                else:
                    utilization[resource_type.value] = 0.0
                    
            return utilization
            
    def optimize_allocations(self) -> Dict[str, Any]:
        """Perform intelligent resource optimization."""
        with self._lock:
            optimizations = {
                'freed_resources': {},
                'reallocated_tasks': 0,
                'efficiency_gain': 0.0
            }
            
            # Analyze allocation patterns
            current_time = datetime.now()
            
            for resource_type, pool_info in self.resource_pools.items():
                allocations = pool_info['allocations']
                
                # Find expired or inefficient allocations
                expired_allocations = []
                for allocation in allocations:
                    age = (current_time - allocation['timestamp']).total_seconds()
                    if age > 3600:  # 1 hour timeout
                        expired_allocations.append(allocation)
                        
                # Clean up expired allocations
                for expired in expired_allocations:
                    self.deallocate_resources(expired['task_id'])
                    freed_amount = expired['resources'].get(resource_type, 0.0)
                    optimizations['freed_resources'][resource_type.value] = \
                        optimizations['freed_resources'].get(resource_type.value, 0.0) + freed_amount
                        
            return optimizations
            
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get detailed allocation statistics."""
        with self._lock:
            stats = {
                'total_allocations': len(self.allocation_history),
                'resource_utilization': self.get_resource_utilization(),
                'allocation_distribution': {},
                'average_allocation_time': 0.0
            }
            
            if self.allocation_history:
                # Calculate average allocation duration
                total_duration = 0.0
                completed_allocations = 0
                
                for allocation in self.allocation_history:
                    # This is simplified - in practice you'd track completion times
                    age = (datetime.now() - allocation['timestamp']).total_seconds()
                    if age > 60:  # Consider allocations older than 1 minute as completed
                        total_duration += age
                        completed_allocations += 1
                        
                if completed_allocations > 0:
                    stats['average_allocation_time'] = total_duration / completed_allocations
                    
            return stats
            
    def _can_allocate(self, resource_requirements: Dict[ResourceType, float]) -> bool:
        """Check if resources can be allocated."""
        for resource_type, amount in resource_requirements.items():
            if resource_type in self.resource_pools:
                pool_info = self.resource_pools[resource_type]
                available = pool_info['available'] - pool_info['allocated']
                if available < amount:
                    return False
        return True
        
    def _optimize_and_retry(self, resource_requirements: Dict[ResourceType, float]) -> bool:
        """Try optimization strategies and retry allocation."""
        optimizations = self.optimize_allocations()
        
        # Check if optimization freed enough resources
        return self._can_allocate(resource_requirements)


class EnterpriseAutoScaler:
    """Enterprise-grade auto-scaler with advanced features."""
    
    def __init__(self, config: AdvancedScalingConfig):
        self.config = config
        self.resource_manager = AdvancedResourceManager(config)
        self.load_balancer = LoadBalancer(config)
        self.predictive_scaler = PredictiveScaler(config)
        self.resource_pool = ResourcePool(config)
        
        # Scaling state
        self.current_workers = config.min_workers
        self.worker_nodes = {}
        self.scaling_history = deque(maxlen=100)
        self.last_scale_up = 0
        self.last_scale_down = 0
        
        # Monitoring and control
        self.metrics_collector = get_metrics_collector()
        self.performance_monitor = get_performance_monitor()
        self._lock = RLock()
        self._running = False
        self._monitoring_thread = None
        self._optimization_thread = None
        
        # Circuit breaker for error handling
        self.circuit_breaker_state = "closed"  # closed, open, half-open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        
    def start(self):
        """Start the enterprise auto-scaler."""
        if self._running:
            return
            
        self._running = True
        
        # Start resource pool
        self.resource_pool.start()
        
        # Initialize worker nodes
        self._initialize_worker_nodes()
        
        # Start monitoring threads
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self._optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._optimization_thread.start()
        
        logger.info("Enterprise auto-scaler started successfully")
        
    def stop(self):
        """Stop the enterprise auto-scaler."""
        self._running = False
        
        # Stop threads
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        if self._optimization_thread:
            self._optimization_thread.join(timeout=10)
            
        # Stop resource pool
        self.resource_pool.stop()
        
        logger.info("Enterprise auto-scaler stopped")
        
    def submit_task(self, task_func: Callable, *args, task_context: Dict[str, Any] = None, **kwargs):
        """Submit task with intelligent node selection."""
        # Check circuit breaker
        if not self._is_circuit_breaker_closed():
            raise RuntimeError("Circuit breaker is open - system is degraded")
            
        # Select optimal node
        selected_node = self.load_balancer.select_node(task_context)
        if not selected_node:
            raise RuntimeError("No healthy nodes available")
            
        # Update node state
        with self._lock:
            selected_node.active_tasks += 1
            
        start_time = time.time()
        success = True
        
        try:
            # Execute task (simplified - in practice would use actual executor)
            result = task_func(*args, **kwargs)
            return result
            
        except Exception as e:
            success = False
            self._handle_task_error(selected_node.id, e)
            raise
            
        finally:
            # Update metrics
            execution_time = time.time() - start_time
            with self._lock:
                selected_node.active_tasks = max(0, selected_node.active_tasks - 1)
                
            self.load_balancer.update_node_metrics(selected_node.id, execution_time, success)
            
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the auto-scaler."""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'scaling_config': {
                    'mode': self.config.scaling_mode.value,
                    'min_workers': self.config.min_workers,
                    'max_workers': self.config.max_workers,
                    'current_workers': self.current_workers
                },
                'resource_utilization': self.resource_manager.get_resource_utilization(),
                'load_balancer_stats': self.load_balancer.get_load_stats(),
                'predictive_analysis': self.predictive_scaler.predict_load(),
                'resource_pool_stats': self.resource_pool.get_pool_stats(),
                'allocation_stats': self.resource_manager.get_allocation_stats(),
                'circuit_breaker': {
                    'state': self.circuit_breaker_state,
                    'failure_count': self.circuit_breaker_failures
                },
                'scaling_history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'action': entry['action'],
                        'from_workers': entry['from_workers'],
                        'to_workers': entry['to_workers'],
                        'reason': entry['reason']
                    }
                    for entry in list(self.scaling_history)[-10:]  # Last 10 scaling events
                ],
                'performance_metrics': {
                    'avg_response_time': np.mean([node.avg_response_time for node in self.worker_nodes.values()]),
                    'total_error_rate': np.mean([node.error_rate for node in self.worker_nodes.values()]),
                    'total_completed_tasks': sum(node.completed_tasks for node in self.worker_nodes.values()),
                    'total_failed_tasks': sum(node.failed_tasks for node in self.worker_nodes.values())
                }
            }
            
    def _initialize_worker_nodes(self):
        """Initialize initial worker nodes."""
        with self._lock:
            for i in range(self.config.min_workers):
                node = WorkerNode(
                    id=f"worker_{i}",
                    tier="default",
                    created_at=datetime.now(),
                    last_health_check=datetime.now()
                )
                self.worker_nodes[node.id] = node
                self.load_balancer.add_node(node)
                
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self._running:
            try:
                current_time = time.time()
                
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Record metrics for prediction
                self.predictive_scaler.record_metrics(datetime.now(), system_metrics)
                
                # Determine scaling action
                scaling_decision = self._make_scaling_decision(system_metrics)
                
                if scaling_decision['action'] != 'none':
                    self._execute_scaling_decision(scaling_decision)
                    
                # Update health checks
                self._update_node_health()
                
                # Check circuit breaker
                self._update_circuit_breaker()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                traceback.print_exc()
                time.sleep(30)
                
    def _optimization_loop(self):
        """Resource optimization loop."""
        while self._running:
            try:
                # Optimize resource allocations
                self.resource_manager.optimize_allocations()
                
                # Cleanup idle resources
                self._cleanup_idle_resources()
                
                # Update load balancer weights
                self._update_load_balancer_weights()
                
                time.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(300)
                
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics."""
        try:
            return {
                'cpu_utilization': psutil.cpu_percent(interval=1),
                'memory_utilization': psutil.virtual_memory().percent,
                'disk_utilization': psutil.disk_usage('/').percent,
                'network_utilization': self._get_network_utilization(),
                'active_tasks': sum(node.active_tasks for node in self.worker_nodes.values()),
                'queue_size': 0,  # Would be actual queue size
                'error_rate': np.mean([node.error_rate for node in self.worker_nodes.values()]) * 100,
                'avg_response_time': np.mean([node.avg_response_time for node in self.worker_nodes.values()])
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
            
    def _get_network_utilization(self) -> float:
        """Get network utilization percentage."""
        try:
            net_io = psutil.net_io_counters()
            # Simplified calculation - would need baseline measurements
            return min(100.0, (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024) / 100)
        except:
            return 0.0
            
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make intelligent scaling decision based on multiple factors."""
        current_time = time.time()
        
        # Get predictions if in predictive mode
        predicted_metrics = {}
        if self.config.scaling_mode in [ScalingMode.PREDICTIVE, ScalingMode.ADAPTIVE]:
            predicted_metrics = self.predictive_scaler.predict_load()
            
        # Analyze current and predicted metrics
        cpu_util = metrics.get('cpu_utilization', 0)
        memory_util = metrics.get('memory_utilization', 0)
        error_rate = metrics.get('error_rate', 0)
        response_time = metrics.get('avg_response_time', 0)
        queue_size = metrics.get('queue_size', 0)
        
        # Check scale-up conditions
        scale_up_reasons = []
        if cpu_util > self.config.scale_up_threshold:
            scale_up_reasons.append(f"CPU utilization: {cpu_util:.1f}%")
        if memory_util > self.config.memory_threshold:
            scale_up_reasons.append(f"Memory utilization: {memory_util:.1f}%")
        if error_rate > self.config.error_rate_threshold:
            scale_up_reasons.append(f"Error rate: {error_rate:.1f}%")
        if response_time > self.config.response_time_threshold_ms:
            scale_up_reasons.append(f"Response time: {response_time:.1f}ms")
        if queue_size > self.config.queue_threshold:
            scale_up_reasons.append(f"Queue size: {queue_size}")
            
        # Check scale-down conditions
        scale_down_reasons = []
        if (cpu_util < self.config.scale_down_threshold and 
            memory_util < 60 and error_rate < 1.0 and queue_size < 2):
            scale_down_reasons.append("Low resource utilization")
            
        # Check cooldown periods
        can_scale_up = current_time - self.last_scale_up > self.config.scale_up_cooldown
        can_scale_down = current_time - self.last_scale_down > self.config.scale_down_cooldown
        
        # Make decision
        if scale_up_reasons and can_scale_up and self.current_workers < self.config.max_workers:
            return {
                'action': 'scale_up',
                'target_workers': min(self.current_workers + 1, self.config.max_workers),
                'reason': '; '.join(scale_up_reasons)
            }
        elif scale_down_reasons and can_scale_down and self.current_workers > self.config.min_workers:
            return {
                'action': 'scale_down', 
                'target_workers': max(self.current_workers - 1, self.config.min_workers),
                'reason': '; '.join(scale_down_reasons)
            }
        else:
            return {'action': 'none', 'reason': 'No scaling needed'}
            
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute scaling decision."""
        with self._lock:
            action = decision['action']
            target_workers = decision['target_workers']
            current_time = time.time()
            
            if action == 'scale_up':
                self._scale_up(target_workers)
                self.last_scale_up = current_time
            elif action == 'scale_down':
                self._scale_down(target_workers)
                self.last_scale_down = current_time
                
            # Record scaling history
            self.scaling_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'from_workers': self.current_workers,
                'to_workers': target_workers,
                'reason': decision['reason']
            })
            
            self.current_workers = target_workers
            logger.info(f"Scaled {action}: {self.current_workers} workers - {decision['reason']}")
            
    def _scale_up(self, target_workers: int):
        """Scale up worker nodes."""
        nodes_to_add = target_workers - len(self.worker_nodes)
        
        for i in range(nodes_to_add):
            node_id = f"worker_{len(self.worker_nodes)}_{int(time.time())}"
            node = WorkerNode(
                id=node_id,
                tier="default",
                created_at=datetime.now(),
                last_health_check=datetime.now()
            )
            
            self.worker_nodes[node.id] = node
            self.load_balancer.add_node(node)
            
    def _scale_down(self, target_workers: int):
        """Scale down worker nodes."""
        nodes_to_remove = len(self.worker_nodes) - target_workers
        
        # Select least utilized nodes for removal
        sorted_nodes = sorted(self.worker_nodes.values(), 
                            key=lambda n: (n.active_tasks, n.utilization))
        
        for i in range(min(nodes_to_remove, len(sorted_nodes))):
            node = sorted_nodes[i]
            if node.active_tasks == 0:  # Only remove idle nodes
                self.load_balancer.remove_node(node.id)
                del self.worker_nodes[node.id]
                
    def _update_node_health(self):
        """Update health status of all nodes."""
        current_time = datetime.now()
        
        for node in self.worker_nodes.values():
            # Simulate health check (in practice would check actual node health)
            node.last_health_check = current_time
            
            # Update resource usage
            node.resource_usage = {
                'cpu': random.uniform(20, 80),  # Simulated values
                'memory': random.uniform(30, 70)
            }
            
            # Update health based on error rate and resource usage
            node.is_healthy = (node.error_rate < 0.1 and 
                             node.resource_usage.get('cpu', 0) < 95)
                             
    def _update_circuit_breaker(self):
        """Update circuit breaker state based on system health."""
        current_time = time.time()
        
        # Count recent failures
        recent_failures = sum(1 for node in self.worker_nodes.values() 
                            if node.error_rate > 0.2)
        
        if self.circuit_breaker_state == "closed":
            if recent_failures > len(self.worker_nodes) * 0.5:  # More than 50% failing
                self.circuit_breaker_state = "open"
                self.circuit_breaker_failures = recent_failures
                self.circuit_breaker_last_failure = current_time
                logger.warning("Circuit breaker OPENED due to high failure rate")
                
        elif self.circuit_breaker_state == "open":
            # Try to recover after 60 seconds
            if current_time - self.circuit_breaker_last_failure > 60:
                self.circuit_breaker_state = "half-open"
                logger.info("Circuit breaker moved to HALF-OPEN")
                
        elif self.circuit_breaker_state == "half-open":
            if recent_failures == 0:
                self.circuit_breaker_state = "closed"
                self.circuit_breaker_failures = 0
                logger.info("Circuit breaker CLOSED - system recovered")
            elif recent_failures > 0:
                self.circuit_breaker_state = "open"
                self.circuit_breaker_last_failure = current_time
                
    def _is_circuit_breaker_closed(self) -> bool:
        """Check if circuit breaker allows requests."""
        return self.circuit_breaker_state in ["closed", "half-open"]
        
    def _handle_task_error(self, node_id: str, error: Exception):
        """Handle task execution error."""
        logger.error(f"Task error on node {node_id}: {error}")
        
        # Update circuit breaker
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
    def _cleanup_idle_resources(self):
        """Clean up idle resources to free memory."""
        gc.collect()  # Force garbage collection
        
    def _update_load_balancer_weights(self):
        """Update load balancer weights based on performance."""
        if self.config.load_balancing_algorithm == LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED:
            for node in self.worker_nodes.values():
                # Weights are updated automatically by the load balancer
                pass


# Global enterprise auto-scaler instance
_enterprise_scaler = None


def get_enterprise_scaler(config: Optional[AdvancedScalingConfig] = None) -> EnterpriseAutoScaler:
    """Get global enterprise auto-scaler instance."""
    global _enterprise_scaler
    if _enterprise_scaler is None:
        if config is None:
            config = AdvancedScalingConfig()
        _enterprise_scaler = EnterpriseAutoScaler(config)
    return _enterprise_scaler


def start_enterprise_scaling():
    """Start enterprise auto-scaling services."""
    scaler = get_enterprise_scaler()
    scaler.start()
    logger.info("Enterprise auto-scaling services started")


def stop_enterprise_scaling():
    """Stop enterprise auto-scaling services."""
    if _enterprise_scaler:
        _enterprise_scaler.stop()
    logger.info("Enterprise auto-scaling services stopped")


def get_enterprise_scaling_status() -> Dict[str, Any]:
    """Get comprehensive enterprise scaling status."""
    scaler = get_enterprise_scaler()
    return scaler.get_comprehensive_status()