"""
Comprehensive monitoring and observability for photonic systems.
"""

import time
import psutil
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SystemHealth:
    """System health snapshot."""
    cpu_percent: float
    memory_percent: float  
    disk_usage_percent: float
    active_threads: int
    open_files: int
    network_connections: int
    timestamp: datetime
    status: str = "healthy"  # healthy, degraded, critical


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, retention_hours: int = 24, collection_interval: int = 60):
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        self.metrics = defaultdict(lambda: deque(maxlen=retention_hours * 60))
        self.health_history = deque(maxlen=retention_hours * 60)
        
        self._lock = threading.Lock()
        self._collectors = {}
        self._running = False
        self._thread = None
        
        # Built-in system collectors
        self.register_collector("system_cpu", self._collect_cpu_usage)
        self.register_collector("system_memory", self._collect_memory_usage)  
        self.register_collector("system_disk", self._collect_disk_usage)
        
    def register_collector(self, name: str, collector_func: Callable[[], float]):
        """Register a custom metric collector."""
        self._collectors[name] = collector_func
        logger.info(f"Registered metric collector: {name}")
        
    def start_collection(self):
        """Start background metric collection."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Started metrics collection")
        
    def stop_collection(self):
        """Stop background metric collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped metrics collection")
        
    def record_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record a custom metric value."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
    def get_metric_history(self, name: str, hours: int = 1) -> List[PerformanceMetric]:
        """Get metric history for specified time range."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if name not in self.metrics:
                return []
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
            
    def get_metric_summary(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of metric over time range."""
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {}
            
        values = [m.value for m in history]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        try:
            health = SystemHealth(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage_percent=psutil.disk_usage('/').percent,
                active_threads=threading.active_count(),
                open_files=len(psutil.Process().open_files()),
                network_connections=len(psutil.net_connections()),
                timestamp=datetime.now()
            )
            
            # Determine health status
            if (health.cpu_percent > 90 or health.memory_percent > 90 or 
                health.disk_usage_percent > 95):
                health.status = "critical"
            elif (health.cpu_percent > 70 or health.memory_percent > 80 or 
                  health.disk_usage_percent > 85):
                health.status = "degraded"
            else:
                health.status = "healthy"
                
            return health
            
        except Exception as e:
            logger.error(f"Failed to collect system health: {e}")
            return SystemHealth(
                cpu_percent=0, memory_percent=0, disk_usage_percent=0,
                active_threads=0, open_files=0, network_connections=0,
                timestamp=datetime.now(), status="unknown"
            )
            
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        with self._lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'retention_hours': self.retention_hours,
                'metrics': {}
            }
            
            for name, metric_deque in self.metrics.items():
                export_data['metrics'][name] = [
                    asdict(metric) for metric in metric_deque
                ]
                
            if format == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
    def _collection_loop(self):
        """Background thread for collecting metrics."""
        while self._running:
            try:
                # Collect system health
                health = self.get_system_health()
                with self._lock:
                    self.health_history.append(health)
                    
                # Run registered collectors
                for name, collector_func in self._collectors.items():
                    try:
                        value = collector_func()
                        self.record_metric(name, value)
                    except Exception as e:
                        logger.warning(f"Collector {name} failed: {e}")
                        
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
        
    def _collect_memory_usage(self) -> float:
        """Collect memory usage percentage."""
        return psutil.virtual_memory().percent
        
    def _collect_disk_usage(self) -> float:
        """Collect disk usage percentage."""
        return psutil.disk_usage('/').percent


class PhotonicPerformanceMonitor:
    """Specialized performance monitor for photonic operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        self._lock = threading.Lock()
        
        # Register photonic-specific collectors
        self.metrics_collector.register_collector(
            "photonic_operations_per_second", 
            self._collect_operations_per_second
        )
        
    def record_operation(self, operation_type: str, duration: float, success: bool = True):
        """Record a photonic operation with timing."""
        with self._lock:
            self.operation_counts[operation_type] += 1
            self.operation_times[operation_type].append(duration)
            
            if not success:
                self.error_counts[operation_type] += 1
                
        # Record metrics
        self.metrics_collector.record_metric(
            f"photonic_{operation_type}_duration",
            duration,
            unit="seconds",
            tags={"operation": operation_type, "success": str(success)}
        )
        
    def get_operation_stats(self, operation_type: str = None) -> Dict[str, Any]:
        """Get operation statistics."""
        with self._lock:
            if operation_type:
                if operation_type not in self.operation_counts:
                    return {}
                    
                times = self.operation_times[operation_type]
                return {
                    'operation_type': operation_type,
                    'total_count': self.operation_counts[operation_type],
                    'error_count': self.error_counts[operation_type],
                    'success_rate': (self.operation_counts[operation_type] - 
                                   self.error_counts[operation_type]) / 
                                   max(self.operation_counts[operation_type], 1),
                    'avg_duration': np.mean(times) if times else 0,
                    'p95_duration': np.percentile(times, 95) if times else 0,
                    'p99_duration': np.percentile(times, 99) if times else 0
                }
            else:
                # Return stats for all operations
                stats = {}
                for op_type in self.operation_counts:
                    stats[op_type] = self.get_operation_stats(op_type)
                return stats
                
    def _collect_operations_per_second(self) -> float:
        """Calculate operations per second over last minute."""
        with self._lock:
            total_ops = sum(self.operation_counts.values())
            return total_ops / 60.0  # Rough approximation
            

class AlertManager:
    """Manages alerts and notifications for system issues."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        self._lock = threading.Lock()
        self._check_interval = 30  # seconds
        self._running = False
        self._thread = None
        
        # Default alert rules
        self.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda health: health.cpu_percent > 85,
            message="CPU usage is critically high: {cpu_percent:.1f}%",
            severity="critical"
        )
        
        self.add_alert_rule(
            name="high_memory_usage", 
            condition=lambda health: health.memory_percent > 90,
            message="Memory usage is critically high: {memory_percent:.1f}%",
            severity="critical"
        )
        
    def add_alert_rule(self, name: str, condition: Callable, message: str, severity: str = "warning"):
        """Add a new alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'message': message,
            'severity': severity
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
        
    def start_monitoring(self):
        """Start alert monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Started alert monitoring")
        
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped alert monitoring")
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
            
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time range."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self.alert_history 
                   if alert['timestamp'] >= cutoff_time]
                   
    def _monitoring_loop(self):
        """Background alert monitoring loop."""
        while self._running:
            try:
                health = self.metrics_collector.get_system_health()
                
                for rule in self.alert_rules:
                    try:
                        if rule['condition'](health):
                            self._trigger_alert(rule, health)
                        else:
                            self._resolve_alert(rule['name'])
                    except Exception as e:
                        logger.error(f"Alert rule {rule['name']} failed: {e}")
                        
                time.sleep(self._check_interval)
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                time.sleep(self._check_interval)
                
    def _trigger_alert(self, rule: Dict[str, Any], health: SystemHealth):
        """Trigger an alert."""
        alert_id = rule['name']
        
        with self._lock:
            if alert_id not in self.active_alerts:
                alert = {
                    'id': alert_id,
                    'name': rule['name'],
                    'message': rule['message'].format(**asdict(health)),
                    'severity': rule['severity'],
                    'timestamp': datetime.now(),
                    'health_snapshot': asdict(health)
                }
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert.copy())
                
                logger.warning(f"ALERT TRIGGERED: {alert['message']}")
                
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                
                resolution = {
                    'id': alert_id,
                    'name': alert['name'],
                    'message': f"Alert resolved: {alert['message']}",
                    'severity': 'resolved',
                    'timestamp': datetime.now(),
                    'resolved_alert': alert
                }
                
                self.alert_history.append(resolution)
                logger.info(f"ALERT RESOLVED: {alert['name']}")


# Global monitoring instances
_metrics_collector = None
_performance_monitor = None
_alert_manager = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_performance_monitor() -> PhotonicPerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PhotonicPerformanceMonitor(get_metrics_collector())
    return _performance_monitor


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager


def start_monitoring():
    """Start all monitoring services."""
    collector = get_metrics_collector()
    collector.start_collection()
    
    alert_mgr = get_alert_manager()
    alert_mgr.start_monitoring()
    
    logger.info("All monitoring services started")


def stop_monitoring():
    """Stop all monitoring services."""
    if _metrics_collector:
        _metrics_collector.stop_collection()
    if _alert_manager:
        _alert_manager.stop_monitoring()
        
    logger.info("All monitoring services stopped")


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    collector = get_metrics_collector()
    perf_monitor = get_performance_monitor()
    alert_mgr = get_alert_manager()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'system_health': asdict(collector.get_system_health()),
        'active_alerts': alert_mgr.get_active_alerts(),
        'photonic_operations': perf_monitor.get_operation_stats(),
        'metric_summary': {
            name: collector.get_metric_summary(name, hours=1)
            for name in ['system_cpu', 'system_memory', 'system_disk']
        }
    }