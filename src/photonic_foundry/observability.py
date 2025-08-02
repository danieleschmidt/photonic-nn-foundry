"""Observability and monitoring utilities for photonic-nn-foundry."""

import logging
import time
import functools
import threading
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os


# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup structured logging for the application."""
    
    logger = logging.getLogger("photonic_foundry")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    if format_type == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, summary


class MetricsCollector:
    """Thread-safe metrics collector for Prometheus-style metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self._lock:
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type="gauge"
            )
            self._metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            metric = MetricPoint(
                name=name,
                value=self._counters[name],
                timestamp=time.time(),
                labels=labels or {},
                metric_type="counter"
            )
            self._metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            self._histograms[name].append(value)
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type="histogram"
            )
            self._metrics[name].append(metric)
    
    def get_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all collected metrics."""
        with self._lock:
            return {name: list(points) for name, points in self._metrics.items()}
    
    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, points in self._metrics.items():
                if not points:
                    continue
                
                latest_point = points[-1]
                
                # Add help and type comments
                lines.append(f"# HELP {name} Photonic foundry metric")
                lines.append(f"# TYPE {name} {latest_point.metric_type}")
                
                # Format labels
                if latest_point.labels:
                    label_str = ",".join([f'{k}="{v}"' for k, v in latest_point.labels.items()])
                    lines.append(f"{name}{{{label_str}}} {latest_point.value}")
                else:
                    lines.append(f"{name} {latest_point.value}")
        
        return "\n".join(lines)


# Global metrics collector
metrics = MetricsCollector()


class PerformanceTracker:
    """Track performance metrics for operations."""
    
    def __init__(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        metrics.increment_counter(
            f"{self.operation_name}_started_total",
            labels=self.labels
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Record duration
        metrics.record_histogram(
            f"{self.operation_name}_duration_seconds",
            duration,
            labels=self.labels
        )
        
        # Record completion
        status = "error" if exc_type else "success"
        completion_labels = {**self.labels, "status": status}
        metrics.increment_counter(
            f"{self.operation_name}_completed_total",
            labels=completion_labels
        )
        
        # Log performance info
        logger = logging.getLogger("photonic_foundry.performance")
        logger.info(
            f"Operation completed",
            extra={
                'extra_fields': {
                    'operation': self.operation_name,
                    'duration_seconds': duration,
                    'status': status,
                    'labels': self.labels
                }
            }
        )


def track_performance(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track performance of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTracker(operation_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, collect_interval: float = 10.0):
        self.collect_interval = collect_interval
        self._running = False
        self._thread = None
    
    def start(self):
        """Start resource monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_resources, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _collect_resources(self):
        """Collect system resource metrics."""
        logger = logging.getLogger("photonic_foundry.resources")
        
        while self._running:
            try:
                self._collect_memory_metrics()
                self._collect_cpu_metrics()
                self._collect_disk_metrics()
                
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
            
            time.sleep(self.collect_interval)
    
    def _collect_memory_metrics(self):
        """Collect memory usage metrics."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            metrics.record_gauge("system_memory_total_bytes", memory.total)
            metrics.record_gauge("system_memory_used_bytes", memory.used)
            metrics.record_gauge("system_memory_available_bytes", memory.available)
            metrics.record_gauge("system_memory_percent", memory.percent)
            
        except ImportError:
            # Fallback for systems without psutil
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    
                # Parse basic memory info
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        total_kb = int(line.split()[1])
                        metrics.record_gauge("system_memory_total_bytes", total_kb * 1024)
                    elif line.startswith('MemAvailable:'):
                        available_kb = int(line.split()[1])
                        metrics.record_gauge("system_memory_available_bytes", available_kb * 1024)
                        
            except (FileNotFoundError, ValueError):
                pass
    
    def _collect_cpu_metrics(self):
        """Collect CPU usage metrics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            metrics.record_gauge("system_cpu_percent", cpu_percent)
            metrics.record_gauge("system_cpu_count", cpu_count)
            
            # Per-CPU metrics
            per_cpu = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu_usage in enumerate(per_cpu):
                metrics.record_gauge(
                    "system_cpu_percent_per_core",
                    cpu_usage,
                    labels={"cpu": str(i)}
                )
                
        except ImportError:
            # Fallback using /proc/loadavg
            try:
                with open('/proc/loadavg', 'r') as f:
                    load_avg = f.read().strip().split()
                    metrics.record_gauge("system_load_average_1m", float(load_avg[0]))
                    metrics.record_gauge("system_load_average_5m", float(load_avg[1]))
                    metrics.record_gauge("system_load_average_15m", float(load_avg[2]))
                    
            except (FileNotFoundError, ValueError, IndexError):
                pass
    
    def _collect_disk_metrics(self):
        """Collect disk usage metrics."""
        try:
            import psutil
            disk_usage = psutil.disk_usage('.')
            
            metrics.record_gauge("system_disk_total_bytes", disk_usage.total)
            metrics.record_gauge("system_disk_used_bytes", disk_usage.used)
            metrics.record_gauge("system_disk_free_bytes", disk_usage.free)
            metrics.record_gauge(
                "system_disk_percent",
                (disk_usage.used / disk_usage.total) * 100
            )
            
        except ImportError:
            # Fallback using shutil
            import shutil
            try:
                total, used, free = shutil.disk_usage('.')
                metrics.record_gauge("system_disk_total_bytes", total)
                metrics.record_gauge("system_disk_used_bytes", used)
                metrics.record_gauge("system_disk_free_bytes", free)
                metrics.record_gauge("system_disk_percent", (used / total) * 100)
                
            except OSError:
                pass


class ApplicationHealthCheck:
    """Application health check utilities."""
    
    def __init__(self):
        self.checks = {}
        self.logger = logging.getLogger("photonic_foundry.health")
    
    def register_check(self, name: str, check_func: Callable[[], bool], critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'status': 'healthy',
            'checks': {},
            'timestamp': time.time()
        }
        
        failed_critical = False
        
        for name, check_config in self.checks.items():
            try:
                start_time = time.time()
                check_result = check_config['func']()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': 'pass' if check_result else 'fail',
                    'duration_seconds': duration,
                    'critical': check_config['critical']
                }
                
                if not check_result and check_config['critical']:
                    failed_critical = True
                
                # Record metrics
                metrics.record_histogram(f"health_check_duration_seconds", duration, {"check": name})
                metrics.increment_counter(
                    "health_check_total",
                    labels={"check": name, "status": "pass" if check_result else "fail"}
                )
                
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check_config['critical']
                }
                
                if check_config['critical']:
                    failed_critical = True
                
                self.logger.error(f"Health check '{name}' failed: {e}")
        
        if failed_critical:
            results['status'] = 'unhealthy'
        elif any(check['status'] != 'pass' for check in results['checks'].values()):
            results['status'] = 'degraded'
        
        return results


# Initialize global components
health_checker = ApplicationHealthCheck()
resource_monitor = ResourceMonitor()


def setup_observability(
    log_level: str = "INFO",
    metrics_enabled: bool = True,
    resource_monitoring: bool = True,
    log_file: Optional[str] = None
):
    """Setup comprehensive observability for the application."""
    
    # Setup logging
    logger = setup_logging(
        level=log_level,
        format_type="structured",
        log_file=log_file
    )
    
    logger.info("Observability system initialized")
    
    # Start resource monitoring
    if resource_monitoring:
        resource_monitor.start()
        logger.info("Resource monitoring started")
    
    # Register basic health checks
    health_checker.register_check("logging", lambda: True, critical=True)
    health_checker.register_check("metrics", lambda: metrics_enabled, critical=False)
    
    return {
        'logger': logger,
        'metrics': metrics,
        'health_checker': health_checker,
        'resource_monitor': resource_monitor
    }


@contextmanager
def trace_operation(operation_name: str, **labels):
    """Context manager for tracing operations with distributed tracing."""
    
    trace_id = os.urandom(8).hex()
    span_id = os.urandom(4).hex()
    
    logger = logging.getLogger("photonic_foundry.tracing")
    
    start_time = time.time()
    
    logger.info(
        f"Starting operation: {operation_name}",
        extra={
            'extra_fields': {
                'trace_id': trace_id,
                'span_id': span_id,
                'operation': operation_name,
                'labels': labels,
                'event': 'operation_start'
            }
        }
    )
    
    try:
        yield {
            'trace_id': trace_id,
            'span_id': span_id,
            'operation': operation_name
        }
        
        duration = time.time() - start_time
        
        logger.info(
            f"Completed operation: {operation_name}",
            extra={
                'extra_fields': {
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'labels': labels,
                    'event': 'operation_complete',
                    'status': 'success'
                }
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            f"Failed operation: {operation_name}",
            extra={
                'extra_fields': {
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'labels': labels,
                    'event': 'operation_error',
                    'status': 'error',
                    'error': str(e)
                }
            }
        )
        raise


# Convenience functions for common operations
def log_model_conversion(model_name: str, input_size: tuple, output_format: str):
    """Log model conversion operation."""
    logger = logging.getLogger("photonic_foundry.conversion")
    logger.info(
        "Model conversion initiated",
        extra={
            'extra_fields': {
                'model_name': model_name,
                'input_size': input_size,
                'output_format': output_format,
                'event': 'model_conversion_start'
            }
        }
    )


def log_simulation_results(model_name: str, energy_per_op: float, latency: float, accuracy: float):
    """Log simulation results."""
    logger = logging.getLogger("photonic_foundry.simulation")
    
    # Log structured data
    logger.info(
        "Simulation completed",
        extra={
            'extra_fields': {
                'model_name': model_name,
                'energy_per_op_pj': energy_per_op,
                'latency_ps': latency,
                'accuracy': accuracy,
                'event': 'simulation_complete'
            }
        }
    )
    
    # Record metrics
    metrics.record_gauge("simulation_energy_per_op_pj", energy_per_op, {"model": model_name})
    metrics.record_gauge("simulation_latency_ps", latency, {"model": model_name})
    metrics.record_gauge("simulation_accuracy", accuracy, {"model": model_name})


# Export key components
__all__ = [
    'setup_observability',
    'track_performance', 
    'PerformanceTracker',
    'trace_operation',
    'metrics',
    'health_checker',
    'resource_monitor',
    'log_model_conversion',
    'log_simulation_results'
]