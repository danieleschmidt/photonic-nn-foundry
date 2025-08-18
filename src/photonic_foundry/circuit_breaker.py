"""
Circuit breaker pattern implementation for resilient system design.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Provides automatic failure detection and recovery.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker {self.config.name} is OPEN"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info(f"Circuit breaker {self.config.name} reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker {self.config.name} opened after {self.failure_count} failures"
            )
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitState.HALF_OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitState.CLOSED


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    """
    Decorator to add circuit breaker protection to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
        name: Circuit breaker name (auto-generated if not provided)
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        
        with _registry_lock:
            if breaker_name not in _circuit_breakers:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception,
                    name=breaker_name
                )
                _circuit_breakers[breaker_name] = CircuitBreaker(config)
        
        breaker = _circuit_breakers[breaker_name]
        return breaker(func)
    
    return decorator


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name."""
    with _registry_lock:
        return _circuit_breakers.get(name)


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _registry_lock:
        return _circuit_breakers.copy()


def reset_circuit_breaker(name: str) -> bool:
    """Reset a circuit breaker to closed state."""
    with _registry_lock:
        if name in _circuit_breakers:
            breaker = _circuit_breakers[name]
            with breaker._lock:
                breaker.state = CircuitState.CLOSED
                breaker.failure_count = 0
                breaker.last_failure_time = None
            logger.info(f"Circuit breaker {name} manually reset")
            return True
        return False


def get_circuit_breaker_stats(name: str) -> Optional[Dict[str, Any]]:
    """Get circuit breaker statistics."""
    with _registry_lock:
        if name in _circuit_breakers:
            breaker = _circuit_breakers[name]
            return {
                "name": name,
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure_time": breaker.last_failure_time,
                "failure_threshold": breaker.config.failure_threshold,
                "recovery_timeout": breaker.config.recovery_timeout
            }
        return None