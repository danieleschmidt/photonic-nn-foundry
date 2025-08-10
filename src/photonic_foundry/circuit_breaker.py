"""
Circuit breaker pattern implementation for fault tolerance in photonic systems.
"""

import time
import threading
import functools
import logging
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics
from collections import deque

from .logging_config import get_logger

logger = get_logger(__name__, component="circuit_breaker")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Preventing calls due to failures
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5                    # Failures before opening
    recovery_timeout: float = 60.0               # Seconds before trying half-open
    success_threshold: int = 3                   # Successes to close from half-open
    timeout: float = 30.0                        # Operation timeout in seconds
    expected_exception_types: tuple = (Exception,)  # Exceptions that count as failures
    monitor_window: int = 100                    # Number of recent calls to monitor
    slow_call_threshold: float = 5.0             # Seconds to consider a call slow
    slow_call_rate_threshold: float = 0.5        # Ratio of slow calls to trigger


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    slow_call_rate: float = 0.0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str, breaker_state: CircuitBreakerState):
        super().__init__(message)
        self.breaker_state = breaker_state


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Exception raised when operation times out."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation with comprehensive monitoring and fault tolerance.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.RLock()
        self._last_failure_time = None
        self._half_open_calls = 0
        self._call_times = deque(maxlen=self.config.monitor_window)
        self._recent_calls = deque(maxlen=self.config.monitor_window)
        
        logger.info(f"Circuit breaker '{name}' initialized", extra={
            'failure_threshold': self.config.failure_threshold,
            'recovery_timeout': self.config.recovery_timeout,
            'timeout': self.config.timeout
        })
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, new_state: CircuitBreakerState):
        """Set circuit breaker state with logging."""
        with self._lock:
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self.metrics.state_changes += 1
                self.metrics.current_state = new_state
                
                logger.warning(f"Circuit breaker '{self.name}' state changed", extra={
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'total_state_changes': self.metrics.state_changes
                })
    
    def _should_attempt_call(self) -> bool:
        """Check if call should be attempted based on current state."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self.config.success_threshold
            else:  # OPEN state
                if self._last_failure_time is None:
                    return True
                
                time_since_failure = time.time() - self._last_failure_time
                if time_since_failure >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
                    
                return False
    
    def _record_success(self, response_time: float):
        """Record successful call."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.utcnow()
            
            self._call_times.append(response_time)
            self._recent_calls.append(True)  # True for success
            
            # Check if call was slow
            if response_time > self.config.slow_call_threshold:
                self.metrics.slow_calls += 1
                logger.warning(f"Slow call detected in '{self.name}'", extra={
                    'response_time': response_time,
                    'threshold': self.config.slow_call_threshold
                })
            
            self._update_metrics()
            
            # Handle state transitions
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
    
    def _record_failure(self, exception: Exception):
        """Record failed call."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self._last_failure_time = time.time()
            
            self._recent_calls.append(False)  # False for failure
            self._update_metrics()
            
            logger.error(f"Call failed in circuit breaker '{self.name}'", extra={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'failure_count': self._get_recent_failure_count()
            }, exc_info=True)
            
            # Handle state transitions
            if self._state == CircuitBreakerState.CLOSED:
                if self._get_recent_failure_count() >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.critical(f"Circuit breaker '{self.name}' opened due to failures")
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened during recovery")
    
    def _get_recent_failure_count(self) -> int:
        """Get count of recent failures."""
        return sum(1 for success in self._recent_calls if not success)
    
    def _update_metrics(self):
        """Update calculated metrics."""
        if self._call_times:
            self.metrics.average_response_time = statistics.mean(self._call_times)
        
        if self._recent_calls:
            failures = sum(1 for success in self._recent_calls if not success)
            self.metrics.failure_rate = failures / len(self._recent_calls)
            
            slow_calls = sum(1 for time in self._call_times if time > self.config.slow_call_threshold)
            self.metrics.slow_call_rate = slow_calls / len(self._call_times) if self._call_times else 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
            CircuitBreakerTimeoutError: If operation times out
        """
        if not self._should_attempt_call():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self._state.value}",
                self._state
            )
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=self.config.timeout)
                    response_time = time.time() - start_time
                    self._record_success(response_time)
                    return result
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    timeout_error = CircuitBreakerTimeoutError(
                        f"Operation in '{self.name}' timed out after {self.config.timeout}s",
                        self._state
                    )
                    self._record_failure(timeout_error)
                    raise timeout_error
                    
        except Exception as e:
            if isinstance(e, self.config.expected_exception_types):
                self._record_failure(e)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator usage of circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return CircuitBreakerMetrics(
                total_calls=self.metrics.total_calls,
                successful_calls=self.metrics.successful_calls,
                failed_calls=self.metrics.failed_calls,
                slow_calls=self.metrics.slow_calls,
                state_changes=self.metrics.state_changes,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                current_state=self._state,
                average_response_time=self.metrics.average_response_time,
                failure_rate=self.metrics.failure_rate,
                slow_call_rate=self.metrics.slow_call_rate
            )
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._last_failure_time = None
            self._half_open_calls = 0
            self._call_times.clear()
            self._recent_calls.clear()
            self.metrics = CircuitBreakerMetrics()
            
            logger.info(f"Circuit breaker '{self.name}' reset")
    
    def force_open(self):
        """Force circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker '{self.name}' forced open")
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        logger.info(f"Circuit breaker '{self.name}' forced closed")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False
    
    def list_all(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_metrics() 
                   for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry
_circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """
    Decorator for adding circuit breaker protection to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: Callable) -> Callable:
        breaker = _circuit_breaker_registry.get_or_create(name, config)
        return breaker(func)
    return decorator


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name."""
    return _circuit_breaker_registry.get(name)


def get_all_circuit_breakers() -> Dict[str, CircuitBreakerMetrics]:
    """Get metrics for all circuit breakers."""
    return _circuit_breaker_registry.list_all()


def reset_circuit_breaker(name: str) -> bool:
    """Reset specific circuit breaker."""
    breaker = _circuit_breaker_registry.get(name)
    if breaker:
        breaker.reset()
        return True
    return False


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    _circuit_breaker_registry.reset_all()


# Predefined configurations for common use cases
DATABASE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    timeout=10.0,
    slow_call_threshold=2.0
)

API_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    timeout=30.0,
    slow_call_threshold=5.0
)

PHOTONIC_SIMULATION_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=120.0,
    timeout=60.0,
    slow_call_threshold=10.0
)