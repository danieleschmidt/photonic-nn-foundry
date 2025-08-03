"""
API request/response schemas for photonic foundry.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class PDKType(str, Enum):
    """Supported PDK types."""
    SKYWATER130 = "skywater130"
    TSMC65 = "tsmc65"
    GENERIC = "generic"


class TargetArchitecture(str, Enum):
    """Target photonic architectures."""
    PHOTONIC_MAC = "photonic_mac"
    PHOTONIC_CONV = "photonic_conv"
    HYBRID = "hybrid"


class CircuitSchema(BaseModel):
    """Schema for photonic circuit representation."""
    name: str = Field(..., min_length=1, max_length=100)
    layers: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[tuple] = Field(default_factory=list)
    total_components: int = Field(ge=0)
    pdk: PDKType = PDKType.SKYWATER130
    wavelength: float = Field(default=1550.0, ge=1200.0, le=1700.0)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate circuit name."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Circuit name must be alphanumeric with underscores/hyphens')
        return v


class CircuitMetricsSchema(BaseModel):
    """Schema for circuit performance metrics."""
    energy_per_op: float = Field(ge=0, description="Energy per operation in pJ")
    latency: float = Field(ge=0, description="Latency in ps")
    area: float = Field(ge=0, description="Area in mm²")
    power: float = Field(ge=0, description="Power consumption in mW")
    throughput: float = Field(ge=0, description="Throughput in GOPS")
    accuracy: float = Field(ge=0, le=1, description="Accuracy relative to FP32")


class TranspileRequest(BaseModel):
    """Request schema for model transpilation."""
    model_data: str = Field(..., description="Base64 encoded PyTorch model")
    target: TargetArchitecture = TargetArchitecture.PHOTONIC_MAC
    precision: int = Field(default=8, ge=1, le=32)
    optimize: bool = Field(default=True)
    pdk: PDKType = PDKType.SKYWATER130
    wavelength: float = Field(default=1550.0, ge=1200.0, le=1700.0)
    
    @validator('model_data')
    def validate_model_data(cls, v):
        """Validate base64 encoded model data."""
        try:
            import base64
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError('Invalid base64 encoded model data')
        return v


class TranspileResponse(BaseModel):
    """Response schema for model transpilation."""
    success: bool
    circuit: Optional[CircuitSchema] = None
    verilog_code: Optional[str] = None
    metrics: Optional[CircuitMetricsSchema] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = Field(ge=0)


class AnalysisRequest(BaseModel):
    """Request schema for model analysis."""
    model_data: str = Field(..., description="Base64 encoded PyTorch model")
    detailed: bool = Field(default=False, description="Include detailed layer analysis")
    
    @validator('model_data')
    def validate_model_data(cls, v):
        """Validate base64 encoded model data."""
        try:
            import base64
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError('Invalid base64 encoded model data')
        return v


class LayerAnalysis(BaseModel):
    """Schema for individual layer analysis."""
    name: str
    type: str
    input_size: int = Field(ge=0)
    output_size: int = Field(ge=0)
    parameters: int = Field(ge=0)
    photonic_components: int = Field(ge=0)
    supported: bool
    complexity_score: float = Field(ge=0)


class AnalysisResponse(BaseModel):
    """Response schema for model analysis."""
    success: bool
    total_layers: int = Field(ge=0)
    supported_layers: int = Field(ge=0)
    total_parameters: int = Field(ge=0)
    compatibility_score: float = Field(ge=0, le=1)
    complexity_score: float = Field(ge=0)
    layer_details: List[LayerAnalysis] = Field(default_factory=list)
    unsupported_layers: List[tuple] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    estimated_metrics: Optional[CircuitMetricsSchema] = None
    error_message: Optional[str] = None
    execution_time: float = Field(ge=0)


class BenchmarkRequest(BaseModel):
    """Request schema for performance benchmarking."""
    model_data: str = Field(..., description="Base64 encoded PyTorch model")
    input_shape: List[int] = Field(..., min_items=1)
    iterations: int = Field(default=100, ge=1, le=10000)
    pdk: PDKType = PDKType.SKYWATER130
    include_electronic_baseline: bool = Field(default=True)
    precision: int = Field(default=8, ge=1, le=32)
    
    @validator('input_shape')
    def validate_input_shape(cls, v):
        """Validate input shape dimensions."""
        if any(dim <= 0 for dim in v):
            raise ValueError('All input shape dimensions must be positive')
        return v


class BenchmarkMetrics(BaseModel):
    """Schema for benchmark performance metrics."""
    inference_time_ms: float = Field(ge=0)
    throughput_ops: float = Field(ge=0)
    energy_per_inference_pj: float = Field(ge=0)
    power_mw: float = Field(ge=0)
    memory_usage_mb: float = Field(ge=0)


class BenchmarkResponse(BaseModel):
    """Response schema for performance benchmarking."""
    success: bool
    electronic_baseline: Optional[BenchmarkMetrics] = None
    photonic_implementation: Optional[BenchmarkMetrics] = None
    speedup_factor: Optional[float] = Field(ge=0)
    energy_efficiency_factor: Optional[float] = Field(ge=0)
    accuracy_comparison: Optional[float] = Field(ge=0, le=1)
    iterations_completed: int = Field(ge=0)
    total_execution_time: float = Field(ge=0)
    error_message: Optional[str] = None


class CircuitListRequest(BaseModel):
    """Request schema for listing circuits."""
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    name_pattern: Optional[str] = Field(max_length=100)
    pdk_filter: Optional[PDKType] = None
    has_verilog: Optional[bool] = None
    has_metrics: Optional[bool] = None


class CircuitInfo(BaseModel):
    """Schema for circuit information in lists."""
    name: str
    model_hash: str
    layer_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    created_at: str
    updated_at: str
    has_verilog: bool
    has_metrics: bool
    pdk: Optional[str] = None
    metrics_summary: Optional[Dict[str, float]] = None


class CircuitListResponse(BaseModel):
    """Response schema for circuit listing."""
    success: bool
    circuits: List[CircuitInfo] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    has_more: bool
    error_message: Optional[str] = None


class DatabaseStatsResponse(BaseModel):
    """Response schema for database statistics."""
    success: bool
    database_stats: Dict[str, Any] = Field(default_factory=dict)
    cache_stats: Dict[str, Any] = Field(default_factory=dict)
    circuit_stats: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(default="healthy")
    version: str
    uptime_seconds: float = Field(ge=0)
    database_connected: bool
    cache_available: bool
    services: Dict[str, str] = Field(default_factory=dict)
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None


class ValidationErrorDetail(BaseModel):
    """Schema for validation error details."""
    field: str
    message: str
    invalid_value: Any


class ValidationErrorResponse(BaseModel):
    """Response schema for validation errors."""
    success: bool = False
    error_code: str = "VALIDATION_ERROR"
    error_message: str = "Request validation failed"
    validation_errors: List[ValidationErrorDetail] = Field(default_factory=list)
    timestamp: str


# Response schemas for common HTTP status codes
class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class NotFoundResponse(BaseModel):
    """404 Not Found response."""
    success: bool = False
    error_code: str = "NOT_FOUND"
    error_message: str
    resource_type: str
    resource_id: str


class ConflictResponse(BaseModel):
    """409 Conflict response."""
    success: bool = False
    error_code: str = "CONFLICT"
    error_message: str
    conflicting_resource: Optional[str] = None


class RateLimitResponse(BaseModel):
    """429 Rate Limit response."""
    success: bool = False
    error_code: str = "RATE_LIMIT_EXCEEDED"
    error_message: str = "Rate limit exceeded"
    retry_after_seconds: int = Field(ge=0)
    limit: int = Field(ge=0)
    window_seconds: int = Field(ge=0)