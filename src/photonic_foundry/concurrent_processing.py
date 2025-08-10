"""
Advanced concurrent processing and distributed computing for photonic foundry systems.

This module provides enterprise-grade concurrent processing capabilities including:
- Multi-tiered processing pools with dynamic allocation  
- Distributed task execution with fault tolerance
- Advanced work-stealing algorithms
- Resource-aware task scheduling
- Adaptive thread/process pool management
- Streaming data processing pipelines
- Actor-based concurrent processing model
"""

import time
import threading
import multiprocessing
import asyncio
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, 
                               as_completed, Future, Executor)
from dataclasses import dataclass, field
from typing import (Dict, List, Any, Optional, Callable, Union, Tuple, 
                   AsyncGenerator, Coroutine, Protocol, TypeVar, Generic)
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
from contextlib import contextmanager, asynccontextmanager
import resource
from threading import RLock, Condition, Event, Barrier, Semaphore
from abc import ABC, abstractmethod
import uuid
import signal
import os
import socket
from queue import PriorityQueue, Queue, Empty, Full
import heapq
import random
import asyncio.subprocess

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ProcessingMode(Enum):
    """Processing modes for different workload patterns."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MIXED = "mixed"
    STREAMING = "streaming"
    BATCH = "batch"


class DistributionStrategy(Enum):
    """Task distribution strategies."""
    ROUND_ROBIN = "round_robin"
    WORK_STEALING = "work_stealing"
    LOCALITY_AWARE = "locality_aware"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"


@dataclass
class TaskMetadata:
    """Metadata for task execution."""
    task_id: str
    priority: TaskPriority
    created_at: datetime
    estimated_duration: float
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.priority.value < other.priority.value


@dataclass
class ProcessingStats:
    """Statistics for processing performance."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    def update_completion(self, execution_time: float, success: bool):
        """Update stats after task completion."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
            
        self.total_execution_time += execution_time
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.average_execution_time = self.total_execution_time / total_tasks
            self.error_rate = self.tasks_failed / total_tasks
            
        # Calculate throughput (simplified)
        if self.total_execution_time > 0:
            self.throughput_per_second = total_tasks / self.total_execution_time


class TaskResult(Generic[T]):
    """Result container for task execution."""
    
    def __init__(self, task_id: str, success: bool, result: Optional[T] = None, 
                 error: Optional[Exception] = None, execution_time: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.completed_at = datetime.now()


class WorkerProtocol(Protocol):
    """Protocol for worker implementations."""
    
    def execute_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Execute a task and return the result."""
        ...
        
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        ...
        
    def shutdown(self):
        """Shutdown the worker."""
        ...


class AdaptiveWorker:
    """Adaptive worker that adjusts to different workload types."""
    
    def __init__(self, worker_id: str, processing_mode: ProcessingMode = ProcessingMode.MIXED):
        self.worker_id = worker_id
        self.processing_mode = processing_mode
        self.is_busy = False
        self.current_task = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        self.resource_usage = {}
        self.last_activity = datetime.now()
        self._lock = RLock()
        
    def execute_task(self, task_func: Callable, task_metadata: TaskMetadata, 
                    *args, **kwargs) -> TaskResult:
        """Execute a task with comprehensive error handling and monitoring."""
        start_time = time.time()
        
        with self._lock:
            self.is_busy = True
            self.current_task = task_metadata
            self.last_activity = datetime.now()
            
        try:
            # Set resource limits if specified
            self._apply_resource_limits(task_metadata.resource_requirements)
            
            # Execute task with timeout
            result = self._execute_with_timeout(
                task_func, task_metadata.timeout_seconds, *args, **kwargs
            )
            
            execution_time = time.time() - start_time
            
            with self._lock:
                self.completed_tasks += 1
                self.total_execution_time += execution_time
                
            return TaskResult(
                task_id=task_metadata.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={'worker_id': self.worker_id}
            )
            
        except TimeoutError:
            execution_time = time.time() - start_time
            with self._lock:
                self.failed_tasks += 1
                
            return TaskResult(
                task_id=task_metadata.task_id,
                success=False,
                error=TimeoutError(f"Task timed out after {task_metadata.timeout_seconds}s"),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            with self._lock:
                self.failed_tasks += 1
                
            return TaskResult(
                task_id=task_metadata.task_id,
                success=False,
                error=e,
                execution_time=execution_time
            )
            
        finally:
            with self._lock:
                self.is_busy = False
                self.current_task = None
                
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        with self._lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            avg_execution_time = (self.total_execution_time / max(total_tasks, 1))
            
            return {
                'worker_id': self.worker_id,
                'processing_mode': self.processing_mode.value,
                'is_busy': self.is_busy,
                'current_task_id': self.current_task.task_id if self.current_task else None,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'total_tasks': total_tasks,
                'success_rate': self.completed_tasks / max(total_tasks, 1),
                'avg_execution_time': avg_execution_time,
                'total_execution_time': self.total_execution_time,
                'last_activity': self.last_activity.isoformat(),
                'resource_usage': self.resource_usage.copy()
            }
            
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        with self._lock:
            return not self.is_busy
            
    def get_load_score(self) -> float:
        """Get load score for work-stealing algorithms."""
        with self._lock:
            if self.is_busy:
                return 1.0
            
            # Consider recent activity and performance
            time_since_activity = (datetime.now() - self.last_activity).total_seconds()
            activity_factor = min(1.0, time_since_activity / 60.0)  # Normalize to 1 minute
            
            total_tasks = self.completed_tasks + self.failed_tasks
            performance_factor = self.completed_tasks / max(total_tasks, 1)
            
            return activity_factor * (1.0 - performance_factor)
            
    def _apply_resource_limits(self, requirements: Dict[str, float]):
        """Apply resource limits for task execution."""
        try:
            if 'memory_mb' in requirements:
                memory_limit = int(requirements['memory_mb'] * 1024 * 1024)  # Convert to bytes
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
                
            if 'cpu_time' in requirements:
                cpu_limit = int(requirements['cpu_time'])
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to apply resource limits: {e}")
            
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs):
        """Execute function with timeout using signal (Unix only)."""
        if os.name == 'nt':  # Windows
            return func(*args, **kwargs)  # Simplified for Windows
            
        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timed out")
            
        # Set timeout signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)


class WorkStealingScheduler:
    """Work-stealing scheduler for load balancing."""
    
    def __init__(self, workers: List[AdaptiveWorker]):
        self.workers = {worker.worker_id: worker for worker in workers}
        self.task_queues = {worker.worker_id: queue.Queue() for worker in workers}
        self.global_queue = PriorityQueue()
        self._lock = RLock()
        self._steal_attempts = defaultdict(int)
        self._steal_successes = defaultdict(int)
        
    def submit_task(self, task_func: Callable, task_metadata: TaskMetadata, 
                   *args, **kwargs) -> str:
        """Submit task to the scheduler."""
        # Find least loaded worker
        best_worker = self._find_best_worker()
        
        if best_worker:
            task_item = (task_metadata, task_func, args, kwargs)
            self.task_queues[best_worker.worker_id].put(task_item)
            return task_metadata.task_id
        else:
            # Add to global queue if no available workers
            task_item = (task_metadata, task_func, args, kwargs)
            self.global_queue.put((task_metadata.priority.value, task_item))
            return task_metadata.task_id
            
    def get_next_task(self, worker_id: str) -> Optional[Tuple]:
        """Get next task for worker, attempting work stealing if needed."""
        # Try local queue first
        try:
            task_item = self.task_queues[worker_id].get_nowait()
            return task_item
        except queue.Empty:
            pass
            
        # Try global queue
        try:
            _, task_item = self.global_queue.get_nowait()
            return task_item
        except queue.Empty:
            pass
            
        # Attempt work stealing from other workers
        return self._attempt_work_stealing(worker_id)
        
    def _find_best_worker(self) -> Optional[AdaptiveWorker]:
        """Find the best worker for task assignment."""
        available_workers = [w for w in self.workers.values() if w.is_available()]
        
        if not available_workers:
            return None
            
        # Sort by load score (lower is better)
        available_workers.sort(key=lambda w: w.get_load_score())
        return available_workers[0]
        
    def _attempt_work_stealing(self, stealing_worker_id: str) -> Optional[Tuple]:
        """Attempt to steal work from other workers."""
        with self._lock:
            self._steal_attempts[stealing_worker_id] += 1
            
            # Find workers with tasks in their queues
            potential_victims = [
                worker_id for worker_id, queue_obj in self.task_queues.items()
                if worker_id != stealing_worker_id and not queue_obj.empty()
            ]
            
            if not potential_victims:
                return None
                
            # Try to steal from a random victim (to avoid contention)
            victim_id = random.choice(potential_victims)
            
            try:
                task_item = self.task_queues[victim_id].get_nowait()
                self._steal_successes[stealing_worker_id] += 1
                
                logger.debug(f"Worker {stealing_worker_id} stole task from {victim_id}")
                return task_item
                
            except queue.Empty:
                return None
                
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            queue_sizes = {
                worker_id: queue_obj.qsize() 
                for worker_id, queue_obj in self.task_queues.items()
            }
            
            total_steal_attempts = sum(self._steal_attempts.values())
            total_steal_successes = sum(self._steal_successes.values())
            steal_success_rate = (total_steal_successes / max(total_steal_attempts, 1))
            
            return {
                'queue_sizes': queue_sizes,
                'global_queue_size': self.global_queue.qsize(),
                'total_steal_attempts': total_steal_attempts,
                'total_steal_successes': total_steal_successes,
                'steal_success_rate': steal_success_rate,
                'steal_attempts_per_worker': dict(self._steal_attempts),
                'steal_successes_per_worker': dict(self._steal_successes)
            }


class StreamProcessor:
    """High-throughput stream processor for continuous data processing."""
    
    def __init__(self, processing_func: Callable, buffer_size: int = 1000,
                 batch_size: int = 10, max_workers: int = 4):
        self.processing_func = processing_func
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)
        self.workers = []
        self.is_running = False
        self._shutdown_event = Event()
        self.stats = ProcessingStats()
        
    def start(self):
        """Start the stream processor."""
        if self.is_running:
            return
            
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(f"stream_worker_{i}",),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started stream processor with {self.max_workers} workers")
        
    def stop(self, timeout: float = 5.0):
        """Stop the stream processor."""
        if not self.is_running:
            return
            
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            
        self.workers.clear()
        logger.info("Stream processor stopped")
        
    def submit(self, item: Any) -> bool:
        """Submit item for processing."""
        if not self.is_running:
            return False
            
        try:
            self.input_queue.put(item, timeout=1.0)
            self.stats.tasks_submitted += 1
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout: float = None) -> Optional[Any]:
        """Get processed result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_results_batch(self, batch_size: int = None, timeout: float = 1.0) -> List[Any]:
        """Get batch of processed results."""
        if batch_size is None:
            batch_size = self.batch_size
            
        results = []
        end_time = time.time() + timeout
        
        while len(results) < batch_size and time.time() < end_time:
            try:
                remaining_time = max(0.1, end_time - time.time())
                result = self.output_queue.get(timeout=remaining_time)
                results.append(result)
            except queue.Empty:
                break
                
        return results
        
    def _worker_loop(self, worker_id: str):
        """Main worker loop for stream processing."""
        batch = []
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Collect batch of items
                batch.clear()
                
                # Get first item (blocking)
                try:
                    item = self.input_queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    continue
                    
                # Get additional items for batch (non-blocking)
                for _ in range(self.batch_size - 1):
                    try:
                        item = self.input_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        break
                        
                # Process batch
                if batch:
                    self._process_batch(batch, worker_id)
                    
            except Exception as e:
                logger.error(f"Stream worker {worker_id} error: {e}")
                
    def _process_batch(self, batch: List[Any], worker_id: str):
        """Process a batch of items."""
        start_time = time.time()
        
        try:
            # Process items individually or as batch depending on function signature
            results = []
            
            for item in batch:
                try:
                    result = self.processing_func(item)
                    results.append(result)
                    self.stats.update_completion(time.time() - start_time, True)
                except Exception as e:
                    logger.error(f"Processing error in {worker_id}: {e}")
                    self.stats.update_completion(time.time() - start_time, False)
                    results.append(None)  # or error marker
                    
            # Put results in output queue
            for result in results:
                if result is not None:
                    try:
                        self.output_queue.put(result, timeout=1.0)
                    except queue.Full:
                        logger.warning("Output queue full, dropping result")
                        
        except Exception as e:
            logger.error(f"Batch processing error in {worker_id}: {e}")


class DistributedTaskExecutor:
    """Distributed task executor with advanced scheduling and fault tolerance."""
    
    def __init__(self, max_workers: int = None, 
                 distribution_strategy: DistributionStrategy = DistributionStrategy.ADAPTIVE):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.distribution_strategy = distribution_strategy
        
        # Worker management
        self.workers = []
        self.worker_scheduler = None
        self.executor = None
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_futures = {}
        self.task_dependencies = {}
        
        # Performance monitoring
        self.stats = ProcessingStats()
        self.performance_history = deque(maxlen=1000)
        
        # Control
        self._lock = RLock()
        self.is_running = False
        self._monitor_thread = None
        
        # Initialize components
        self._initialize_workers()
        
    def _initialize_workers(self):
        """Initialize adaptive workers."""
        self.workers = [
            AdaptiveWorker(f"worker_{i}") 
            for i in range(self.max_workers)
        ]
        
        if self.distribution_strategy == DistributionStrategy.WORK_STEALING:
            self.worker_scheduler = WorkStealingScheduler(self.workers)
            
        # Create thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def start(self):
        """Start the distributed task executor."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Started distributed task executor with {self.max_workers} workers")
        
    def stop(self, timeout: float = 30.0):
        """Stop the distributed task executor."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop monitoring
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, timeout=timeout)
            
        logger.info("Distributed task executor stopped")
        
    def submit_task(self, task_func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: float = 300.0, dependencies: List[str] = None,
                   resource_requirements: Dict[str, float] = None, **kwargs) -> str:
        """Submit task for distributed execution."""
        task_id = str(uuid.uuid4())
        
        # Create task metadata
        task_metadata = TaskMetadata(
            task_id=task_id,
            priority=priority,
            created_at=datetime.now(),
            estimated_duration=timeout,
            resource_requirements=resource_requirements or {},
            dependencies=dependencies or [],
            timeout_seconds=timeout
        )
        
        with self._lock:
            # Check dependencies
            if not self._are_dependencies_satisfied(dependencies or []):
                # Store task for later execution
                self.task_dependencies[task_id] = task_metadata
                return task_id
                
            # Submit task based on distribution strategy
            if self.distribution_strategy == DistributionStrategy.WORK_STEALING:
                self.worker_scheduler.submit_task(task_func, task_metadata, *args, **kwargs)
                
            # Submit to executor
            future = self.executor.submit(
                self._execute_task_with_worker, 
                task_func, task_metadata, *args, **kwargs
            )
            
            self.active_tasks[task_id] = task_metadata
            self.task_futures[task_id] = future
            self.stats.tasks_submitted += 1
            
        return task_id
        
    def get_result(self, task_id: str, timeout: float = None) -> TaskResult:
        """Get result for a specific task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
            
        if task_id not in self.task_futures:
            raise ValueError(f"Task {task_id} not found")
            
        try:
            future = self.task_futures[task_id]
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=e
            )
            
    def get_results(self, timeout: float = None) -> List[TaskResult]:
        """Get all completed results."""
        results = []
        
        with self._lock:
            # Get already completed results
            results.extend(self.completed_tasks.values())
            
            # Check for newly completed futures
            completed_futures = []
            for task_id, future in self.task_futures.items():
                if future.done():
                    try:
                        result = future.result()
                        results.append(result)
                        self.completed_tasks[task_id] = result
                        completed_futures.append(task_id)
                    except Exception as e:
                        error_result = TaskResult(
                            task_id=task_id,
                            success=False,
                            error=e
                        )
                        results.append(error_result)
                        self.completed_tasks[task_id] = error_result
                        completed_futures.append(task_id)
                        
            # Clean up completed futures
            for task_id in completed_futures:
                del self.task_futures[task_id]
                self.active_tasks.pop(task_id, None)
                
        return results
        
    def wait_for_completion(self, task_ids: List[str] = None, timeout: float = None) -> bool:
        """Wait for specific tasks or all tasks to complete."""
        if task_ids is None:
            # Wait for all active tasks
            futures_to_wait = list(self.task_futures.values())
        else:
            futures_to_wait = [
                self.task_futures[task_id] for task_id in task_ids 
                if task_id in self.task_futures
            ]
            
        try:
            # Wait for completion
            for future in as_completed(futures_to_wait, timeout=timeout):
                pass  # Just wait for completion
            return True
        except Exception:
            return False
            
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        with self._lock:
            if task_id in self.task_futures:
                future = self.task_futures[task_id]
                success = future.cancel()
                
                if success:
                    del self.task_futures[task_id]
                    self.active_tasks.pop(task_id, None)
                    
                return success
                
            if task_id in self.task_dependencies:
                del self.task_dependencies[task_id]
                return True
                
        return False
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        with self._lock:
            # Worker stats
            worker_stats = [worker.get_stats() for worker in self.workers]
            
            # Task stats
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
            pending_count = len(self.task_dependencies)
            
            # Performance metrics
            if self.performance_history:
                recent_performance = list(self.performance_history)[-100:]  # Last 100 tasks
                avg_execution_time = np.mean([p['execution_time'] for p in recent_performance])
                throughput = len(recent_performance) / max(sum(p['execution_time'] for p in recent_performance), 1)
            else:
                avg_execution_time = 0.0
                throughput = 0.0
                
            stats = {
                'overall_stats': {
                    'tasks_submitted': self.stats.tasks_submitted,
                    'tasks_active': active_count,
                    'tasks_completed': completed_count,
                    'tasks_pending': pending_count,
                    'success_rate': (self.stats.tasks_completed / max(self.stats.tasks_submitted, 1)),
                    'avg_execution_time': avg_execution_time,
                    'throughput_per_second': throughput
                },
                'worker_stats': worker_stats,
                'distribution_strategy': self.distribution_strategy.value,
                'resource_utilization': self._calculate_resource_utilization()
            }
            
            # Add scheduler stats if using work stealing
            if self.worker_scheduler:
                stats['scheduler_stats'] = self.worker_scheduler.get_scheduler_stats()
                
            return stats
            
    def _execute_task_with_worker(self, task_func: Callable, task_metadata: TaskMetadata,
                                 *args, **kwargs) -> TaskResult:
        """Execute task using an adaptive worker."""
        # Select best available worker
        available_workers = [w for w in self.workers if w.is_available()]
        
        if not available_workers:
            # All workers busy, use round-robin
            worker = self.workers[hash(task_metadata.task_id) % len(self.workers)]
        else:
            # Select least loaded worker
            worker = min(available_workers, key=lambda w: w.get_load_score())
            
        # Execute task
        result = worker.execute_task(task_func, task_metadata, *args, **kwargs)
        
        # Record performance
        self.performance_history.append({
            'task_id': task_metadata.task_id,
            'worker_id': worker.worker_id,
            'execution_time': result.execution_time,
            'success': result.success,
            'timestamp': datetime.now()
        })
        
        # Update stats
        self.stats.update_completion(result.execution_time, result.success)
        
        return result
        
    def _are_dependencies_satisfied(self, dependencies: List[str]) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_task_id in dependencies:
            if dep_task_id in self.active_tasks or dep_task_id in self.task_dependencies:
                return False  # Dependency still running or pending
            if dep_task_id not in self.completed_tasks:
                return False  # Dependency never submitted
            if not self.completed_tasks[dep_task_id].success:
                return False  # Dependency failed
        return True
        
    def _monitor_loop(self):
        """Monitor task execution and handle dependencies."""
        while self.is_running:
            try:
                # Check for satisfied dependencies
                with self._lock:
                    ready_tasks = []
                    
                    for task_id, task_metadata in list(self.task_dependencies.items()):
                        if self._are_dependencies_satisfied(task_metadata.dependencies):
                            ready_tasks.append((task_id, task_metadata))
                            
                    # Submit ready tasks
                    for task_id, task_metadata in ready_tasks:
                        del self.task_dependencies[task_id]
                        # Re-submit the task (would need original function and args)
                        logger.info(f"Dependencies satisfied for task {task_id}")
                        
                # Clean up completed tasks periodically
                self.get_results()  # This also cleans up completed futures
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate overall resource utilization."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'active_threads': threading.active_count(),
                'worker_utilization': len([w for w in self.workers if w.is_busy]) / len(self.workers)
            }
        except Exception:
            return {}


# Actor-based processing model
class Actor:
    """Base actor class for actor-based concurrent processing."""
    
    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self.mailbox = queue.Queue()
        self.is_running = False
        self._thread = None
        self.message_handlers = {}
        
    def start(self):
        """Start the actor."""
        if self.is_running:
            return
            
        self.is_running = True
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the actor."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.mailbox.put(('__STOP__', None))
        
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def send_message(self, message_type: str, data: Any):
        """Send message to this actor."""
        self.mailbox.put((message_type, data))
        
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler."""
        self.message_handlers[message_type] = handler
        
    def _message_loop(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                message_type, data = self.mailbox.get(timeout=1.0)
                
                if message_type == '__STOP__':
                    break
                    
                if message_type in self.message_handlers:
                    self.message_handlers[message_type](data)
                else:
                    self.handle_message(message_type, data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Actor {self.actor_id} message handling error: {e}")
                
    def handle_message(self, message_type: str, data: Any):
        """Override this method to handle messages."""
        logger.warning(f"Unhandled message type: {message_type}")


class ActorSystem:
    """Actor system for managing multiple actors."""
    
    def __init__(self):
        self.actors = {}
        self._lock = RLock()
        
    def create_actor(self, actor_class: type, actor_id: str, *args, **kwargs) -> Actor:
        """Create and start an actor."""
        with self._lock:
            if actor_id in self.actors:
                raise ValueError(f"Actor {actor_id} already exists")
                
            actor = actor_class(actor_id, *args, **kwargs)
            actor.start()
            
            self.actors[actor_id] = actor
            return actor
            
    def get_actor(self, actor_id: str) -> Optional[Actor]:
        """Get actor by ID."""
        return self.actors.get(actor_id)
        
    def send_message(self, actor_id: str, message_type: str, data: Any):
        """Send message to an actor."""
        actor = self.get_actor(actor_id)
        if actor:
            actor.send_message(message_type, data)
        else:
            raise ValueError(f"Actor {actor_id} not found")
            
    def stop_actor(self, actor_id: str):
        """Stop and remove an actor."""
        with self._lock:
            if actor_id in self.actors:
                actor = self.actors[actor_id]
                actor.stop()
                del self.actors[actor_id]
                
    def stop_all(self):
        """Stop all actors."""
        with self._lock:
            for actor in self.actors.values():
                actor.stop()
            self.actors.clear()
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get actor system statistics."""
        with self._lock:
            return {
                'total_actors': len(self.actors),
                'running_actors': len([a for a in self.actors.values() if a.is_running]),
                'actor_ids': list(self.actors.keys())
            }


# Global instances
_distributed_executor = None
_actor_system = None


def get_distributed_executor(max_workers: int = None, 
                           strategy: DistributionStrategy = DistributionStrategy.ADAPTIVE) -> DistributedTaskExecutor:
    """Get global distributed task executor."""
    global _distributed_executor
    if _distributed_executor is None:
        _distributed_executor = DistributedTaskExecutor(max_workers, strategy)
    return _distributed_executor


def get_actor_system() -> ActorSystem:
    """Get global actor system."""
    global _actor_system
    if _actor_system is None:
        _actor_system = ActorSystem()
    return _actor_system


def start_concurrent_processing():
    """Start all concurrent processing services."""
    executor = get_distributed_executor()
    executor.start()
    logger.info("Concurrent processing services started")


def stop_concurrent_processing():
    """Stop all concurrent processing services."""
    if _distributed_executor:
        _distributed_executor.stop()
    if _actor_system:
        _actor_system.stop_all()
    logger.info("Concurrent processing services stopped")