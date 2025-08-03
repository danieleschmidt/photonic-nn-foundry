"""
FastAPI server for photonic neural network foundry.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from ..core import PhotonicAccelerator
from ..database import get_database, close_database
from .endpoints import register_endpoints
from .middleware import setup_middleware
from .schemas import ErrorResponse, HealthCheckResponse

logger = logging.getLogger(__name__)


class PhotonicFoundryAPI:
    """Main API application class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_config()
        self.start_time = time.time()
        self.version = "0.1.0"
        self.accelerator: Optional[PhotonicAccelerator] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'host': os.getenv('API_HOST', '0.0.0.0'),
            'port': int(os.getenv('API_PORT', '8000')),
            'workers': int(os.getenv('API_WORKERS', '1')),
            'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
            'api_prefix': os.getenv('API_PREFIX', '/api/v1'),
            'max_request_size': int(os.getenv('MAX_REQUEST_SIZE', '100')) * 1024 * 1024,  # 100MB
            'rate_limit_requests': int(os.getenv('RATE_LIMIT_REQUESTS', '100')),
            'rate_limit_window': int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            'enable_docs': os.getenv('ENABLE_API_DOCS', 'true').lower() == 'true',
        }
        
    async def startup(self):
        """Application startup handler."""
        logger.info("Starting Photonic Foundry API...")
        
        try:
            # Initialize database connection
            db = get_database()
            logger.info("Database connection established")
            
            # Initialize photonic accelerator
            self.accelerator = PhotonicAccelerator()
            logger.info(f"Photonic accelerator initialized with PDK: {self.accelerator.pdk}")
            
            # Log configuration
            logger.info(f"API configuration: {self.config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize API: {e}")
            raise
            
    async def shutdown(self):
        """Application shutdown handler."""
        logger.info("Shutting down Photonic Foundry API...")
        
        try:
            # Close database connections
            close_database()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    api_instance = app.state.api_instance
    
    # Startup
    await api_instance.startup()
    
    yield
    
    # Shutdown
    await api_instance.shutdown()


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Create API instance
    api_instance = PhotonicFoundryAPI(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Photonic Neural Network Foundry API",
        description="REST API for converting PyTorch models to photonic circuits",
        version=api_instance.version,
        docs_url="/docs" if api_instance.config['enable_docs'] else None,
        redoc_url="/redoc" if api_instance.config['enable_docs'] else None,
        openapi_url="/openapi.json" if api_instance.config['enable_docs'] else None,
        lifespan=lifespan
    )
    
    # Store API instance in app state
    app.state.api_instance = api_instance
    
    # Setup middleware
    setup_middleware(app, api_instance.config)
    
    # Register endpoints
    register_endpoints(app, api_instance.config['api_prefix'])
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title="Photonic Neural Network Foundry API",
            version=api_instance.version,
            description="""
# Photonic Neural Network Foundry API

Convert PyTorch neural networks to energy-efficient photonic circuits.

## Key Features

- **Model Analysis**: Analyze PyTorch models for photonic compatibility
- **Circuit Transpilation**: Convert models to photonic Verilog implementations
- **Performance Benchmarking**: Compare electronic vs photonic performance
- **Circuit Management**: Save, load, and manage photonic circuits
- **Database Integration**: Persistent storage with caching

## Getting Started

1. **Analyze a model**: Use `/analyze` to check compatibility
2. **Transpile model**: Use `/transpile` to generate photonic circuit
3. **Benchmark performance**: Use `/benchmark` to compare implementations
4. **Manage circuits**: Use `/circuits` endpoints for CRUD operations

## Authentication

Currently no authentication required for development. Production deployments
should implement appropriate authentication mechanisms.

## Rate Limiting

API requests are rate limited to prevent abuse. Default limits:
- 100 requests per minute per IP
- Configurable via environment variables

## Error Handling

All endpoints return consistent error responses with:
- `success`: Boolean indicating request success
- `error_code`: Machine-readable error identifier  
- `error_message`: Human-readable error description
- `timestamp`: ISO timestamp of error occurrence
            """,
            routes=app.routes,
        )
        
        # Add custom tags
        openapi_schema["tags"] = [
            {
                "name": "Analysis",
                "description": "Model compatibility analysis endpoints"
            },
            {
                "name": "Transpilation", 
                "description": "Model to circuit conversion endpoints"
            },
            {
                "name": "Benchmarking",
                "description": "Performance comparison endpoints"
            },
            {
                "name": "Circuits",
                "description": "Circuit management endpoints"
            },
            {
                "name": "System",
                "description": "System health and information endpoints"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
        
    app.openapi = custom_openapi
    
    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=f"HTTP_{exc.status_code}",
                error_message=exc.detail,
                timestamp=datetime.utcnow().isoformat()
            ).dict()
        )
        
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                error_message="An internal server error occurred",
                timestamp=datetime.utcnow().isoformat()
            ).dict()
        )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthCheckResponse, tags=["System"])
    async def health_check():
        """Check API health status."""
        try:
            # Check database connection
            db = get_database()
            db_stats = db.get_database_stats()
            database_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            database_connected = False
            
        # Check cache availability
        try:
            from ..database.cache import get_circuit_cache
            cache = get_circuit_cache()
            cache.get_cache_stats()
            cache_available = True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            cache_available = False
            
        uptime = time.time() - api_instance.start_time
        
        return HealthCheckResponse(
            status="healthy" if database_connected and cache_available else "degraded",
            version=api_instance.version,
            uptime_seconds=uptime,
            database_connected=database_connected,
            cache_available=cache_available,
            services={
                "database": "connected" if database_connected else "disconnected",
                "cache": "available" if cache_available else "unavailable",
                "accelerator": "ready" if api_instance.accelerator else "not_initialized"
            },
            timestamp=datetime.utcnow().isoformat()
        )
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root():
        """API root endpoint."""
        return {
            "name": "Photonic Neural Network Foundry API",
            "version": api_instance.version,
            "status": "running",
            "docs_url": "/docs" if api_instance.config['enable_docs'] else None,
            "health_url": "/health",
            "api_prefix": api_instance.config['api_prefix']
        }
    
    return app


def run_server(app: FastAPI, config: Optional[Dict[str, Any]] = None):
    """Run the FastAPI server."""
    import uvicorn
    
    if config is None:
        config = PhotonicFoundryAPI()._load_config()
    
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        workers=config['workers'] if not config['debug'] else 1,
        reload=config['debug'],
        access_log=True,
        log_level="debug" if config['debug'] else "info"
    )


if __name__ == "__main__":
    # Create and run the application
    app = create_app()
    run_server(app)