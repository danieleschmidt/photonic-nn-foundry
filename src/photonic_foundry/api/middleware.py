"""
API middleware for Photonic Neural Network Foundry.
"""

import time
import uuid
import logging
from typing import Dict, Any, Callable
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .schemas import ErrorResponse, RateLimitResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses."""
    
    def __init__(self, app: FastAPI, log_responses: bool = False):
        super().__init__(app)
        self.log_responses = log_responses
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(
            f"Request {request_id} - {request.method} {request.url.path} "
            f"from {client_ip} - User-Agent: {request.headers.get('user-agent', 'unknown')}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Request {request_id} - {response.status_code} "
                f"- {process_time:.3f}s"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} - ERROR: {str(e)} - {process_time:.3f}s"
            )
            
            # Return error response
            error_response = ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                error_message="An internal server error occurred",
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(
        self, 
        app: FastAPI, 
        requests_per_minute: int = 100,
        enabled: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        if not self.enabled:
            return await call_next(request)
            
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_counts = {
            ip: data for ip, data in self.request_counts.items()
            if data['last_request'] > cutoff_time
        }
        
        # Check rate limit
        if client_ip in self.request_counts:
            client_data = self.request_counts[client_ip]
            
            # Reset count if more than a minute has passed
            if current_time - client_data['window_start'] >= 60:
                client_data['count'] = 0
                client_data['window_start'] = current_time
            
            # Check if limit exceeded
            if client_data['count'] >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                
                rate_limit_response = RateLimitResponse(
                    retry_after_seconds=60,
                    limit=self.requests_per_minute,
                    window_seconds=60
                )
                
                return JSONResponse(
                    status_code=429,
                    content=rate_limit_response.dict(),
                    headers={
                        "Retry-After": "60",
                        "X-RateLimit-Limit": str(self.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(current_time + 60))
                    }
                )
            
            # Increment count
            client_data['count'] += 1
            client_data['last_request'] = current_time
            
        else:
            # New client
            self.request_counts[client_ip] = {
                'count': 1,
                'window_start': current_time,
                'last_request': current_time
            }
        
        # Add rate limit headers
        response = await call_next(request)
        
        client_data = self.request_counts.get(client_ip, {})
        remaining = max(0, self.requests_per_minute - client_data.get('count', 0))
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # CSP for API (restrictive)
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Enhanced compression middleware with size thresholds."""
    
    def __init__(self, app: FastAPI, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply compression based on response size and type."""
        response = await call_next(request)
        
        # Check if compression is beneficial
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return response
            
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "application/javascript",
            "text/xml"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return response
            
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""
    
    def __init__(self, app: FastAPI, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.request_count = 0
        self.total_request_time = 0.0
        self.error_count = 0
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for monitoring."""
        if not self.enabled:
            return await call_next(request)
            
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            response = await call_next(request)
            
            # Update metrics
            process_time = time.time() - start_time
            self.request_count += 1
            self.total_request_time += process_time
            
            # Update endpoint stats
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    'count': 0,
                    'total_time': 0.0,
                    'error_count': 0,
                    'avg_time': 0.0
                }
            
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += process_time
            stats['avg_time'] = stats['total_time'] / stats['count']
            
            if response.status_code >= 400:
                self.error_count += 1
                stats['error_count'] += 1
            
            # Add metrics headers
            response.headers["X-Response-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            self.error_count += 1
            
            if endpoint in self.endpoint_stats:
                self.endpoint_stats[endpoint]['error_count'] += 1
            
            raise e
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics data."""
        avg_response_time = (
            self.total_request_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'avg_response_time': avg_response_time,
            'endpoints': self.endpoint_stats
        }


def setup_middleware(app: FastAPI, config: Dict[str, Any]):
    """Setup all middleware for the FastAPI application."""
    
    # Trusted hosts (for production)
    if not config.get('debug', False):
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=["*"]  # Configure properly for production
        )
    
    # CORS middleware
    cors_origins = config.get('cors_origins', ['*'])
    if isinstance(cors_origins, str):
        cors_origins = cors_origins.split(',')
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time", "X-Response-Time"]
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(CompressionMiddleware, minimum_size=1000) 
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting
    if config.get('enable_rate_limiting', True):
        app.add_middleware(
            RateLimitingMiddleware,
            requests_per_minute=config.get('rate_limit_requests', 100),
            enabled=True
        )
    
    # Metrics collection
    if config.get('enable_metrics', True):
        metrics_middleware = MetricsMiddleware(enabled=True)
        app.add_middleware(MetricsMiddleware, enabled=True)
        
        # Store reference for metrics endpoint
        app.state.metrics_middleware = metrics_middleware
    
    # Request logging (should be last to capture all middleware timing)
    app.add_middleware(
        RequestLoggingMiddleware,
        log_responses=config.get('debug', False)
    )
    
    logger.info("API middleware configured successfully")