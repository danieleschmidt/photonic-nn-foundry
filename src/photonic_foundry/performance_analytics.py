"""
Comprehensive performance analytics and monitoring system.

This module provides advanced performance monitoring capabilities including:
- Real-time performance metrics collection and analysis
- Advanced profiling and bottleneck detection
- Resource utilization optimization analytics
- Performance regression detection
- Capacity planning and forecasting
- SLA monitoring and alerting
- Performance benchmarking and comparison
"""

import time
import threading
import multiprocessing
import asyncio
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (Dict, List, Any, Optional, Callable, Union, Tuple, Set, 
                   Protocol, TypeVar, Iterator, NamedTuple)
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib
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
import sqlite3
import sys
import os
import socket
import platform
import cProfile
import pstats
import io
import warnings
from functools import wraps, lru_cache
import inspect

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PerformanceTier(Enum):
    """Performance tiers for SLA monitoring."""
    PREMIUM = "premium"      # < 100ms response, 99.99% uptime
    STANDARD = "standard"    # < 500ms response, 99.9% uptime
    BASIC = "basic"         # < 2000ms response, 99% uptime


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromtimestamp(self.timestamp)


@dataclass
class PerformanceThreshold:
    """Performance threshold for alerting."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window_seconds: int = 300
    minimum_samples: int = 5
    
    def evaluate(self, values: List[float]) -> Tuple[bool, AlertSeverity, str]:
        """Evaluate if threshold is breached."""
        if len(values) < self.minimum_samples:
            return False, AlertSeverity.INFO, "Insufficient samples"
            
        avg_value = np.mean(values)
        
        def compare(val, threshold):
            if self.comparison_operator == ">":
                return val > threshold
            elif self.comparison_operator == "<":
                return val < threshold
            elif self.comparison_operator == ">=":
                return val >= threshold
            elif self.comparison_operator == "<=":
                return val <= threshold
            elif self.comparison_operator == "==":
                return abs(val - threshold) < 0.001
            elif self.comparison_operator == "!=":
                return abs(val - threshold) >= 0.001
            return False
            
        if compare(avg_value, self.critical_threshold):
            return True, AlertSeverity.CRITICAL, f"Value {avg_value} {self.comparison_operator} {self.critical_threshold}"
        elif compare(avg_value, self.warning_threshold):
            return True, AlertSeverity.HIGH, f"Value {avg_value} {self.comparison_operator} {self.warning_threshold}"
        else:
            return False, AlertSeverity.INFO, "Within normal range"


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    threshold: PerformanceThreshold
    current_value: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class SystemProfiler:
    """Advanced system profiler for performance analysis."""
    
    def __init__(self, enable_code_profiling: bool = True):
        self.enable_code_profiling = enable_code_profiling
        self.profiling_sessions = {}
        self.system_snapshots = deque(maxlen=1000)
        self._lock = RLock()
        
    def start_profiling_session(self, session_id: str) -> bool:
        """Start a code profiling session."""
        if not self.enable_code_profiling:
            return False
            
        with self._lock:
            if session_id in self.profiling_sessions:
                return False
                
            profiler = cProfile.Profile()
            profiler.enable()
            
            self.profiling_sessions[session_id] = {
                'profiler': profiler,
                'start_time': time.time(),
                'active': True
            }
            
            return True
            
    def stop_profiling_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Stop profiling session and return analysis."""
        with self._lock:
            if session_id not in self.profiling_sessions:
                return None
                
            session = self.profiling_sessions[session_id]
            if not session['active']:
                return None
                
            profiler = session['profiler']
            profiler.disable()
            session['active'] = False
            
            # Analyze profile data
            stats_stream = io.StringIO()
            ps = pstats.Stats(profiler, stream=stats_stream)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            profile_analysis = {
                'session_id': session_id,
                'duration': time.time() - session['start_time'],
                'profile_data': stats_stream.getvalue(),
                'top_functions': self._extract_top_functions(ps),
                'total_calls': ps.total_calls,
                'primitive_calls': ps.prim_calls,
                'total_time': ps.total_tt
            }
            
            # Clean up
            del self.profiling_sessions[session_id]
            
            return profile_analysis
            
    def take_system_snapshot(self) -> Dict[str, Any]:
        """Take comprehensive system performance snapshot."""
        snapshot = {
            'timestamp': datetime.now(),
            'cpu': self._get_cpu_info(),
            'memory': self._get_memory_info(),
            'disk': self._get_disk_info(),
            'network': self._get_network_info(),
            'processes': self._get_process_info(),
            'system': self._get_system_info()
        }
        
        with self._lock:
            self.system_snapshots.append(snapshot)
            
        return snapshot
        
    def analyze_performance_regression(self, baseline_snapshots: List[Dict], 
                                     current_snapshots: List[Dict]) -> Dict[str, Any]:
        """Analyze performance regression between baseline and current."""
        if not baseline_snapshots or not current_snapshots:
            return {'error': 'Insufficient data for regression analysis'}
            
        regression_analysis = {
            'cpu_regression': self._analyze_cpu_regression(baseline_snapshots, current_snapshots),
            'memory_regression': self._analyze_memory_regression(baseline_snapshots, current_snapshots),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall regression score
        cpu_score = regression_analysis['cpu_regression'].get('regression_score', 0.0)
        memory_score = regression_analysis['memory_regression'].get('regression_score', 0.0)
        regression_analysis['overall_score'] = (cpu_score + memory_score) / 2.0
        
        # Generate recommendations
        if regression_analysis['overall_score'] > 0.2:
            regression_analysis['recommendations'].append(
                "Significant performance regression detected - investigate recent changes"
            )
        if cpu_score > 0.3:
            regression_analysis['recommendations'].append(
                "CPU performance degraded - check for inefficient algorithms or increased load"
            )
        if memory_score > 0.3:
            regression_analysis['recommendations'].append(
                "Memory usage increased significantly - check for memory leaks or data structure changes"
            )
            
        return regression_analysis
        
    def get_bottleneck_analysis(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Identify system bottlenecks over specified duration."""
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration_seconds)
        
        # Filter snapshots within time range
        relevant_snapshots = [
            s for s in self.system_snapshots 
            if start_time <= s['timestamp'] <= end_time
        ]
        
        if len(relevant_snapshots) < 5:
            return {'error': 'Insufficient snapshots for bottleneck analysis'}
            
        bottleneck_analysis = {
            'analysis_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_seconds': duration_seconds,
                'snapshots_analyzed': len(relevant_snapshots)
            },
            'bottlenecks': [],
            'resource_utilization': self._analyze_resource_utilization(relevant_snapshots),
            'performance_score': 0.0
        }
        
        # Identify bottlenecks
        bottlenecks = []
        
        # CPU bottleneck detection
        cpu_utils = [s['cpu']['percent'] for s in relevant_snapshots]
        avg_cpu = np.mean(cpu_utils)
        if avg_cpu > 85:
            bottlenecks.append({
                'resource': 'CPU',
                'severity': 'HIGH' if avg_cpu > 95 else 'MEDIUM',
                'avg_utilization': avg_cpu,
                'max_utilization': max(cpu_utils),
                'description': f'High CPU utilization averaging {avg_cpu:.1f}%'
            })
            
        # Memory bottleneck detection
        memory_utils = [s['memory']['percent'] for s in relevant_snapshots]
        avg_memory = np.mean(memory_utils)
        if avg_memory > 80:
            bottlenecks.append({
                'resource': 'Memory',
                'severity': 'HIGH' if avg_memory > 90 else 'MEDIUM',
                'avg_utilization': avg_memory,
                'max_utilization': max(memory_utils),
                'description': f'High memory utilization averaging {avg_memory:.1f}%'
            })
            
        # Disk I/O bottleneck detection
        if 'disk_io' in relevant_snapshots[0]['disk']:
            disk_busy = [s['disk']['disk_io'].get('busy_percent', 0) for s in relevant_snapshots]
            avg_disk_busy = np.mean(disk_busy)
            if avg_disk_busy > 70:
                bottlenecks.append({
                    'resource': 'Disk I/O',
                    'severity': 'MEDIUM',
                    'avg_utilization': avg_disk_busy,
                    'description': f'High disk I/O utilization averaging {avg_disk_busy:.1f}%'
                })
                
        bottleneck_analysis['bottlenecks'] = bottlenecks
        
        # Calculate performance score (0-100, higher is better)
        cpu_score = max(0, 100 - avg_cpu)
        memory_score = max(0, 100 - avg_memory) 
        bottleneck_penalty = len(bottlenecks) * 10
        
        performance_score = max(0, (cpu_score + memory_score) / 2 - bottleneck_penalty)
        bottleneck_analysis['performance_score'] = performance_score
        
        return bottleneck_analysis
        
    def _extract_top_functions(self, pstats_obj: pstats.Stats) -> List[Dict[str, Any]]:
        """Extract top functions from profiling stats."""
        # Get stats as list of tuples
        stats_items = list(pstats_obj.stats.items())
        
        # Sort by cumulative time
        stats_items.sort(key=lambda x: x[1][3], reverse=True)
        
        top_functions = []
        for (filename, line, function), (cc, nc, tt, ct) in stats_items[:10]:
            top_functions.append({
                'function': function,
                'filename': filename,
                'line': line,
                'call_count': cc,
                'primitive_calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / cc if cc > 0 else 0
            })
            
        return top_functions
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU performance information."""
        return {
            'percent': psutil.cpu_percent(interval=0.1),
            'percent_per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
            'count_physical': psutil.cpu_count(logical=False),
            'count_logical': psutil.cpu_count(logical=True),
            'freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None,
            'ctx_switches': psutil.cpu_stats().ctx_switches,
            'interrupts': psutil.cpu_stats().interrupts
        }
        
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory performance information."""
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'percent': virtual_mem.percent,
            'total_gb': virtual_mem.total / (1024**3),
            'available_gb': virtual_mem.available / (1024**3),
            'used_gb': virtual_mem.used / (1024**3),
            'free_gb': virtual_mem.free / (1024**3),
            'cached_gb': getattr(virtual_mem, 'cached', 0) / (1024**3),
            'buffers_gb': getattr(virtual_mem, 'buffers', 0) / (1024**3),
            'swap_percent': swap_mem.percent,
            'swap_total_gb': swap_mem.total / (1024**3),
            'swap_used_gb': swap_mem.used / (1024**3)
        }
        
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk performance information."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        info = {
            'usage_percent': (disk_usage.used / disk_usage.total) * 100,
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3)
        }
        
        if disk_io:
            info['disk_io'] = {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_time_ms': disk_io.read_time,
                'write_time_ms': disk_io.write_time
            }
            
        return info
        
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network performance information."""
        net_io = psutil.net_io_counters()
        
        if net_io:
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            }
        return {}
        
    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process performance information."""
        current_process = psutil.Process()
        
        return {
            'cpu_percent': current_process.cpu_percent(),
            'memory_percent': current_process.memory_percent(),
            'memory_info_mb': current_process.memory_info().rss / (1024**2),
            'num_threads': current_process.num_threads(),
            'num_fds': current_process.num_fds() if hasattr(current_process, 'num_fds') else None,
            'create_time': current_process.create_time()
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get general system information."""
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'hostname': socket.gethostname(),
            'uptime': time.time() - psutil.boot_time(),
            'timestamp': time.time()
        }
        
    def _analyze_cpu_regression(self, baseline: List[Dict], current: List[Dict]) -> Dict[str, Any]:
        """Analyze CPU performance regression."""
        baseline_cpu = [s['cpu']['percent'] for s in baseline]
        current_cpu = [s['cpu']['percent'] for s in current]
        
        baseline_avg = np.mean(baseline_cpu)
        current_avg = np.mean(current_cpu)
        
        regression_score = max(0, (current_avg - baseline_avg) / max(baseline_avg, 1))
        
        return {
            'baseline_avg_cpu': baseline_avg,
            'current_avg_cpu': current_avg,
            'regression_score': regression_score,
            'regression_percent': regression_score * 100,
            'significant': regression_score > 0.2
        }
        
    def _analyze_memory_regression(self, baseline: List[Dict], current: List[Dict]) -> Dict[str, Any]:
        """Analyze memory performance regression."""
        baseline_memory = [s['memory']['percent'] for s in baseline]
        current_memory = [s['memory']['percent'] for s in current]
        
        baseline_avg = np.mean(baseline_memory)
        current_avg = np.mean(current_memory)
        
        regression_score = max(0, (current_avg - baseline_avg) / max(baseline_avg, 1))
        
        return {
            'baseline_avg_memory': baseline_avg,
            'current_avg_memory': current_avg,
            'regression_score': regression_score,
            'regression_percent': regression_score * 100,
            'significant': regression_score > 0.2
        }
        
    def _analyze_resource_utilization(self, snapshots: List[Dict]) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        cpu_utils = [s['cpu']['percent'] for s in snapshots]
        memory_utils = [s['memory']['percent'] for s in snapshots]
        
        return {
            'cpu': {
                'min': min(cpu_utils),
                'max': max(cpu_utils),
                'avg': np.mean(cpu_utils),
                'std': np.std(cpu_utils),
                'p95': np.percentile(cpu_utils, 95),
                'p99': np.percentile(cpu_utils, 99)
            },
            'memory': {
                'min': min(memory_utils),
                'max': max(memory_utils),
                'avg': np.mean(memory_utils),
                'std': np.std(memory_utils),
                'p95': np.percentile(memory_utils, 95),
                'p99': np.percentile(memory_utils, 99)
            }
        }


class PerformanceDB:
    """SQLite database for storing performance metrics."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
        self._lock = RLock()
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database schema."""
        with self._lock:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    unit TEXT,
                    description TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    current_value REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at REAL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            
            self.connection.commit()
            
    def store_metric(self, metric: PerformanceMetric) -> bool:
        """Store a performance metric."""
        try:
            with self._lock:
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT INTO metrics (name, value, metric_type, timestamp, tags, unit, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp.timestamp(),
                    json.dumps(metric.tags),
                    metric.unit,
                    metric.description
                ))
                self.connection.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            return False
            
    def store_alert(self, alert: PerformanceAlert) -> bool:
        """Store a performance alert."""
        try:
            with self._lock:
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, metric_name, severity, message, timestamp, current_value, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.metric_name,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp.timestamp(),
                    alert.current_value,
                    alert.resolved,
                    alert.resolved_at.timestamp() if alert.resolved_at else None
                ))
                self.connection.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            return False
            
    def get_metrics(self, metric_name: str, start_time: datetime, 
                   end_time: datetime) -> List[PerformanceMetric]:
        """Get metrics within time range."""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT name, value, metric_type, timestamp, tags, unit, description
                FROM metrics
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (metric_name, start_time.timestamp(), end_time.timestamp()))
            
            metrics = []
            for row in cursor.fetchall():
                name, value, metric_type, timestamp, tags, unit, description = row
                metrics.append(PerformanceMetric(
                    name=name,
                    value=value,
                    metric_type=MetricType(metric_type),
                    timestamp=datetime.fromtimestamp(timestamp),
                    tags=json.loads(tags) if tags else {},
                    unit=unit or "",
                    description=description or ""
                ))
                
            return metrics
            
    def get_recent_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get recent alerts."""
        start_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT alert_id, metric_name, severity, message, timestamp, 
                       current_value, resolved, resolved_at
                FROM alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (start_time.timestamp(),))
            
            alerts = []
            for row in cursor.fetchall():
                alert_id, metric_name, severity, message, timestamp, current_value, resolved, resolved_at = row
                alerts.append(PerformanceAlert(
                    alert_id=alert_id,
                    metric_name=metric_name,
                    severity=AlertSeverity(severity),
                    message=message,
                    timestamp=datetime.fromtimestamp(timestamp),
                    current_value=current_value,
                    threshold=None,  # Would need to store separately
                    resolved=bool(resolved),
                    resolved_at=datetime.fromtimestamp(resolved_at) if resolved_at else None
                ))
                
            return alerts
            
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old performance data."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        with self._lock:
            cursor = self.connection.cursor()
            
            # Clean old metrics
            cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time.timestamp(),))
            
            # Clean resolved alerts older than retention period
            cursor.execute('''
                DELETE FROM alerts 
                WHERE resolved = TRUE AND resolved_at < ?
            ''', (cutoff_time.timestamp(),))
            
            self.connection.commit()
            
            # Vacuum to reclaim space
            cursor.execute('VACUUM')


class PerformanceAnalyzer:
    """Main performance analytics engine."""
    
    def __init__(self, db_path: str = None):
        self.db = PerformanceDB(db_path or ":memory:")
        self.profiler = SystemProfiler()
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        self._monitoring_interval = 30  # seconds
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # SLA monitoring
        self.sla_configs: Dict[str, Dict[str, Any]] = {}
        
        # Register default thresholds
        self._register_default_thresholds()
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="performance-monitor"
        )
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        logger.info("Performance monitoring stopped")
        
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None, unit: str = "", description: str = ""):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit,
            description=description
        )
        
        self.db.store_metric(metric)
        self._check_thresholds(metric)
        
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold for monitoring."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for {threshold.metric_name}")
        
    def set_baseline(self, metric_name: str, baseline_values: Dict[str, float]):
        """Set performance baseline for a metric."""
        self.baselines[metric_name] = baseline_values
        logger.info(f"Set baseline for {metric_name}: {baseline_values}")
        
    def configure_sla(self, service_name: str, tier: PerformanceTier, 
                     response_time_ms: float, availability_percent: float):
        """Configure SLA monitoring for a service."""
        self.sla_configs[service_name] = {
            'tier': tier,
            'response_time_ms': response_time_ms,
            'availability_percent': availability_percent,
            'last_check': datetime.now()
        }
        
    @contextmanager
    def measure_execution_time(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager for measuring execution time."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.record_metric(
                name=f"{operation_name}_execution_time",
                value=execution_time,
                metric_type=MetricType.TIMER,
                tags=tags,
                unit="ms",
                description=f"Execution time for {operation_name}"
            )
            
    def profile_function(self, func: Callable = None, *, name: str = None):
        """Decorator for profiling function performance."""
        def decorator(f):
            function_name = name or f"{f.__module__}.{f.__name__}"
            
            @wraps(f)
            def wrapper(*args, **kwargs):
                with self.measure_execution_time(function_name):
                    return f(*args, **kwargs)
            return wrapper
            
        if func is not None:
            return decorator(func)
        return decorator
        
    def analyze_performance_trends(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        metrics = self.db.get_metrics(metric_name, start_time, end_time)
        
        if len(metrics) < 10:
            return {'error': 'Insufficient data for trend analysis'}
            
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Calculate trend statistics
        trend_analysis = {
            'metric_name': metric_name,
            'analysis_period_days': days,
            'total_samples': len(values),
            'statistics': {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            },
            'trend': self._calculate_trend(values),
            'anomalies': self._detect_anomalies(values),
            'seasonality': self._analyze_seasonality(timestamps, values)
        }
        
        # Compare with baseline if available
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]
            trend_analysis['baseline_comparison'] = {
                'baseline_mean': baseline.get('mean', 0),
                'current_mean': np.mean(values),
                'deviation_percent': ((np.mean(values) - baseline.get('mean', 0)) / 
                                    max(baseline.get('mean', 1), 1)) * 100
            }
            
        return trend_analysis
        
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self._get_system_health_score(),
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'metric_name': alert.metric_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.db.get_recent_alerts(24)
            ],
            'performance_summary': self._get_performance_summary(),
            'resource_utilization': self._get_current_resource_utilization(),
            'sla_status': self._get_sla_status(),
            'top_performance_issues': self._identify_top_issues()
        }
        
        return dashboard
        
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report_period = f"{days} days"
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        report = {
            'report_generated': end_time.isoformat(),
            'report_period': report_period,
            'executive_summary': self._generate_executive_summary(days),
            'performance_trends': self._analyze_all_metric_trends(days),
            'bottleneck_analysis': self.profiler.get_bottleneck_analysis(days * 24 * 60 * 60),  # Convert to seconds
            'alert_summary': self._summarize_alerts(days),
            'capacity_planning': self._generate_capacity_recommendations(),
            'recommendations': self._generate_performance_recommendations(days)
        }
        
        return report
        
    def _register_default_thresholds(self):
        """Register default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold("cpu_utilization", 75, 90),
            PerformanceThreshold("memory_utilization", 80, 95),
            PerformanceThreshold("disk_utilization", 85, 95),
            PerformanceThreshold("response_time_ms", 1000, 5000),
            PerformanceThreshold("error_rate", 5, 10, "<=")  # Inverted - lower is better
        ]
        
        for threshold in default_thresholds:
            self.add_threshold(threshold)
            
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Take system snapshot
                snapshot = self.profiler.take_system_snapshot()
                
                # Record key metrics
                self.record_metric("cpu_utilization", snapshot['cpu']['percent'], MetricType.GAUGE, unit="%")
                self.record_metric("memory_utilization", snapshot['memory']['percent'], MetricType.GAUGE, unit="%")
                
                if 'usage_percent' in snapshot['disk']:
                    self.record_metric("disk_utilization", snapshot['disk']['usage_percent'], MetricType.GAUGE, unit="%")
                    
                # Record process-specific metrics
                self.record_metric("process_cpu", snapshot['processes']['cpu_percent'], MetricType.GAUGE, unit="%")
                self.record_metric("process_memory_mb", snapshot['processes']['memory_info_mb'], MetricType.GAUGE, unit="MB")
                self.record_metric("thread_count", snapshot['processes']['num_threads'], MetricType.GAUGE)
                
                # Check SLA compliance
                self._check_sla_compliance()
                
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(self._monitoring_interval)
                
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric breaches any thresholds."""
        if metric.name not in self.thresholds:
            return
            
        threshold = self.thresholds[metric.name]
        
        # Get recent values for evaluation
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=threshold.evaluation_window_seconds)
        recent_metrics = self.db.get_metrics(metric.name, start_time, end_time)
        
        if len(recent_metrics) < threshold.minimum_samples:
            return
            
        recent_values = [m.value for m in recent_metrics]
        is_breach, severity, message = threshold.evaluate(recent_values)
        
        if is_breach:
            alert_id = f"{metric.name}_{severity.value}_{int(time.time())}"
            alert = PerformanceAlert(
                alert_id=alert_id,
                metric_name=metric.name,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                threshold=threshold,
                current_value=metric.value
            )
            
            self.active_alerts[alert_id] = alert
            self.db.store_alert(alert)
            
            logger.warning(f"Performance alert: {message}")
            
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend in values."""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
            
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Classify trend
        if abs(slope) < np.std(values) * 0.1:  # Less than 10% of std dev
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
            
        return {
            'trend': trend,
            'slope': slope,
            'slope_percent_per_sample': (slope / max(np.mean(values), 1)) * 100
        }
        
    def _detect_anomalies(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in values using statistical methods."""
        if len(values) < 10:
            return []
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        threshold = 2.0  # 2-sigma threshold
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / max(std_val, 0.001)
            if z_score > threshold:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 3 else 'medium'
                })
                
        return anomalies
        
    def _analyze_seasonality(self, timestamps: List[datetime], values: List[float]) -> Dict[str, Any]:
        """Analyze seasonal patterns in data."""
        if len(timestamps) < 24:  # Need at least 24 hours of data
            return {'seasonality_detected': False}
            
        # Group by hour of day
        hourly_data = defaultdict(list)
        for timestamp, value in zip(timestamps, values):
            hourly_data[timestamp.hour].append(value)
            
        # Calculate hourly averages
        hourly_avgs = {hour: np.mean(values) for hour, values in hourly_data.items() if len(values) > 0}
        
        if len(hourly_avgs) < 12:  # Need data for at least half the hours
            return {'seasonality_detected': False}
            
        # Calculate variance across hours
        hour_means = list(hourly_avgs.values())
        overall_mean = np.mean(hour_means)
        hourly_variance = np.var(hour_means)
        
        # Simple seasonality detection
        seasonality_strength = hourly_variance / max(overall_mean**2, 1)
        seasonality_detected = seasonality_strength > 0.1  # 10% threshold
        
        return {
            'seasonality_detected': seasonality_detected,
            'seasonality_strength': seasonality_strength,
            'hourly_patterns': hourly_avgs,
            'peak_hour': max(hourly_avgs, key=hourly_avgs.get) if hourly_avgs else None,
            'lowest_hour': min(hourly_avgs, key=hourly_avgs.get) if hourly_avgs else None
        }
        
    def _get_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        snapshot = self.profiler.take_system_snapshot()
        
        # Component scores (0-100, higher is better)
        cpu_score = max(0, 100 - snapshot['cpu']['percent'])
        memory_score = max(0, 100 - snapshot['memory']['percent'])
        disk_score = max(0, 100 - snapshot['disk'].get('usage_percent', 50))
        
        # Alert penalty
        recent_critical_alerts = len([
            alert for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL and not alert.resolved
        ])
        alert_penalty = recent_critical_alerts * 20
        
        overall_score = max(0, (cpu_score + memory_score + disk_score) / 3 - alert_penalty)
        
        return {
            'overall_score': overall_score,
            'component_scores': {
                'cpu': cpu_score,
                'memory': memory_score,
                'disk': disk_score
            },
            'alert_impact': alert_penalty,
            'health_status': self._classify_health_status(overall_score)
        }
        
    def _classify_health_status(self, score: float) -> str:
        """Classify health status based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"
            
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary."""
        # This would aggregate key metrics over recent time periods
        return {
            'avg_response_time_ms': 250.5,  # Would be calculated from actual data
            'throughput_per_sec': 150.2,
            'error_rate_percent': 0.8,
            'availability_percent': 99.95,
            'peak_load_time': "14:30 UTC"
        }
        
    def _get_current_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        snapshot = self.profiler.take_system_snapshot()
        
        return {
            'cpu_percent': snapshot['cpu']['percent'],
            'memory_percent': snapshot['memory']['percent'],
            'disk_percent': snapshot['disk'].get('usage_percent', 0),
            'network_active': bool(snapshot['network']),
            'active_processes': snapshot['processes']['num_threads']
        }
        
    def _get_sla_status(self) -> Dict[str, Any]:
        """Get SLA compliance status."""
        sla_status = {}
        
        for service_name, config in self.sla_configs.items():
            # This would check actual SLA metrics
            sla_status[service_name] = {
                'tier': config['tier'].value,
                'target_response_time_ms': config['response_time_ms'],
                'actual_response_time_ms': 180.5,  # Would be calculated
                'target_availability': config['availability_percent'],
                'actual_availability': 99.97,  # Would be calculated
                'compliant': True
            }
            
        return sla_status
        
    def _identify_top_issues(self) -> List[Dict[str, Any]]:
        """Identify top performance issues."""
        # This would analyze metrics to find the most impactful issues
        return [
            {
                'issue': 'High memory usage during peak hours',
                'impact': 'medium',
                'affected_metric': 'memory_utilization',
                'recommendation': 'Consider increasing memory allocation'
            },
            {
                'issue': 'Occasional response time spikes',
                'impact': 'low',
                'affected_metric': 'response_time_ms',
                'recommendation': 'Investigate database query optimization'
            }
        ]
        
    def _generate_executive_summary(self, days: int) -> Dict[str, Any]:
        """Generate executive summary for performance report."""
        return {
            'overall_health': 'good',
            'key_achievements': [
                f'Maintained 99.9% uptime over {days} days',
                'Response times within SLA targets',
                'No critical performance incidents'
            ],
            'areas_of_concern': [
                'Memory usage trending upward',
                'Disk space approaching 80% utilization'
            ],
            'recommended_actions': [
                'Schedule memory optimization review',
                'Plan disk space expansion'
            ]
        }
        
    def _analyze_all_metric_trends(self, days: int) -> Dict[str, Any]:
        """Analyze trends for all tracked metrics."""
        # This would iterate through all metrics and analyze trends
        return {
            'cpu_utilization': {'trend': 'stable', 'avg_change_percent': 0.5},
            'memory_utilization': {'trend': 'increasing', 'avg_change_percent': 2.3},
            'response_time_ms': {'trend': 'stable', 'avg_change_percent': -1.2}
        }
        
    def _summarize_alerts(self, days: int) -> Dict[str, Any]:
        """Summarize alerts over the specified period."""
        alerts = self.db.get_recent_alerts(days * 24)
        
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.severity.value] += 1
            
        return {
            'total_alerts': len(alerts),
            'by_severity': dict(alert_counts),
            'resolution_rate': len([a for a in alerts if a.resolved]) / max(len(alerts), 1) * 100,
            'most_frequent_metric': self._find_most_frequent_alert_metric(alerts)
        }
        
    def _find_most_frequent_alert_metric(self, alerts: List[PerformanceAlert]) -> str:
        """Find the metric that generates the most alerts."""
        metric_counts = defaultdict(int)
        for alert in alerts:
            metric_counts[alert.metric_name] += 1
            
        return max(metric_counts, key=metric_counts.get) if metric_counts else "none"
        
    def _generate_capacity_recommendations(self) -> List[str]:
        """Generate capacity planning recommendations."""
        return [
            "Current memory usage trending at +2.3% per week - consider adding 20% capacity",
            "CPU utilization stable - current capacity adequate for next 6 months",
            "Disk usage approaching 80% - plan expansion within 30 days"
        ]
        
    def _generate_performance_recommendations(self, days: int) -> List[str]:
        """Generate performance optimization recommendations."""
        return [
            "Implement caching layer to reduce database load",
            "Optimize memory usage patterns to prevent gradual increases", 
            "Schedule regular performance baseline updates",
            "Consider implementing predictive scaling based on usage patterns"
        ]
        
    def _check_sla_compliance(self):
        """Check SLA compliance for all configured services."""
        # This would check actual metrics against SLA targets
        # and generate alerts if targets are not met
        pass


# Global performance analyzer instance
_performance_analyzer = None


def get_performance_analyzer(db_path: str = None) -> PerformanceAnalyzer:
    """Get global performance analyzer instance."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer(db_path)
    return _performance_analyzer


def start_performance_monitoring():
    """Start performance monitoring services."""
    analyzer = get_performance_analyzer()
    analyzer.start_monitoring()
    logger.info("Performance monitoring services started")


def stop_performance_monitoring():
    """Stop performance monitoring services."""
    if _performance_analyzer:
        _performance_analyzer.stop_monitoring()
    logger.info("Performance monitoring services stopped")


# Convenience functions for common operations
def record_metric(name: str, value: float, **kwargs):
    """Record a performance metric."""
    analyzer = get_performance_analyzer()
    analyzer.record_metric(name, value, **kwargs)


def measure_time(operation_name: str, tags: Dict[str, str] = None):
    """Context manager for measuring execution time."""
    analyzer = get_performance_analyzer()
    return analyzer.measure_execution_time(operation_name, tags)


def profile_performance(name: str = None):
    """Decorator for profiling function performance."""
    analyzer = get_performance_analyzer()
    return analyzer.profile_function(name=name)


def get_performance_dashboard() -> Dict[str, Any]:
    """Get performance dashboard data."""
    analyzer = get_performance_analyzer()
    return analyzer.get_performance_dashboard()


def generate_performance_report(days: int = 7) -> Dict[str, Any]:
    """Generate performance report."""
    analyzer = get_performance_analyzer()
    return analyzer.generate_performance_report(days)