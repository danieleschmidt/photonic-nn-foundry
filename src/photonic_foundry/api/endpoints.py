"""
API endpoints for Photonic Neural Network Foundry.
"""

import time
import base64
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
import io
import torch
import numpy as np

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import PlainTextResponse, JSONResponse

from ..core import PhotonicAccelerator, PhotonicCircuit
from ..transpiler import torch2verilog, analyze_model_compatibility
from ..utils.advanced_optimizers import MZIMeshOptimizer, RingResonatorOptimizer
from .schemas import (
    TranspileRequest, TranspileResponse, CircuitSchema, CircuitMetricsSchema,
    AnalysisRequest, AnalysisResponse, LayerAnalysis,
    BenchmarkRequest, BenchmarkResponse, BenchmarkMetrics,
    CircuitListRequest, CircuitListResponse, CircuitInfo,
    DatabaseStatsResponse, SuccessResponse, ErrorResponse,
    NotFoundResponse, ValidationErrorResponse
)

logger = logging.getLogger(__name__)

# Global accelerator instance (will be injected by dependency)
_accelerator_instance: Optional[PhotonicAccelerator] = None


def get_accelerator() -> PhotonicAccelerator:
    """Dependency to get PhotonicAccelerator instance."""
    global _accelerator_instance
    if _accelerator_instance is None:
        _accelerator_instance = PhotonicAccelerator()
    return _accelerator_instance


def register_endpoints(app, api_prefix: str = "/api/v1"):
    """Register all API endpoints with the FastAPI app."""
    
    # Analysis endpoints
    analysis_router = APIRouter(prefix=f"{api_prefix}/analysis", tags=["Analysis"])
    
    @analysis_router.post("/model", response_model=AnalysisResponse)
    async def analyze_model(
        request: AnalysisRequest,
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Analyze PyTorch model for photonic compatibility."""
        start_time = time.time()
        
        try:
            # Decode model data
            model_bytes = base64.b64decode(request.model_data)
            model = torch.load(io.BytesIO(model_bytes), map_location='cpu')
            
            # Analyze model compatibility
            analysis = analyze_model_compatibility(model)
            
            # Create layer details if detailed analysis requested
            layer_details = []
            if request.detailed:
                for layer_info in analysis.get('layer_details', []):
                    layer_details.append(LayerAnalysis(
                        name=layer_info['name'],
                        type=layer_info['type'],
                        input_size=layer_info.get('in_features', layer_info.get('in_channels', 0)),
                        output_size=layer_info.get('out_features', layer_info.get('out_channels', 0)),
                        parameters=layer_info['parameters'],
                        photonic_components=layer_info['photonic_components'],
                        supported=True,
                        complexity_score=layer_info['parameters'] / 1000.0
                    ))
            
            # Estimate metrics for compatible model
            estimated_metrics = None
            if analysis['compatibility_score'] > 0.5:
                try:
                    circuit = accelerator.convert_pytorch_model(model)
                    metrics = accelerator.compile_and_profile(circuit)
                    estimated_metrics = CircuitMetricsSchema(
                        energy_per_op=metrics.energy_per_op,
                        latency=metrics.latency,
                        area=metrics.area,
                        power=metrics.power,
                        throughput=metrics.throughput,
                        accuracy=metrics.accuracy
                    )
                except Exception as e:
                    logger.warning(f"Could not estimate metrics: {e}")
            
            execution_time = time.time() - start_time
            
            return AnalysisResponse(
                success=True,
                total_layers=analysis['total_layers'],
                supported_layers=analysis['supported_layers'],
                total_parameters=analysis['total_parameters'],
                compatibility_score=analysis['compatibility_score'],
                complexity_score=analysis['complexity_score'],
                layer_details=layer_details,
                unsupported_layers=analysis['unsupported_layers'],
                recommendations=analysis['recommendations'],
                estimated_metrics=estimated_metrics,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model analysis failed: {e}")
            
            return AnalysisResponse(
                success=False,
                total_layers=0,
                supported_layers=0,
                total_parameters=0,
                compatibility_score=0.0,
                complexity_score=0.0,
                error_message=str(e),
                execution_time=execution_time
            )
    
    # Transpilation endpoints
    transpile_router = APIRouter(prefix=f"{api_prefix}/transpile", tags=["Transpilation"])
    
    @transpile_router.post("/model", response_model=TranspileResponse)
    async def transpile_model(
        request: TranspileRequest,
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Transpile PyTorch model to photonic Verilog implementation."""
        start_time = time.time()
        
        try:
            # Decode model data
            model_bytes = base64.b64decode(request.model_data)
            model = torch.load(io.BytesIO(model_bytes), map_location='cpu')
            
            # Set accelerator configuration
            accelerator.pdk = request.pdk.value
            accelerator.wavelength = request.wavelength
            
            # Convert to photonic circuit
            circuit = accelerator.convert_pytorch_model(model)
            
            # Generate Verilog
            verilog_code = torch2verilog(
                model,
                target=request.target.value,
                precision=request.precision,
                optimize=request.optimize
            )
            
            # Analyze performance
            metrics = accelerator.compile_and_profile(circuit)
            
            # Get analysis for warnings
            analysis = analyze_model_compatibility(model)
            warnings = []
            if analysis['compatibility_score'] < 1.0:
                warnings.append(f"Model compatibility: {analysis['compatibility_score']:.1%}")
            if analysis['unsupported_layers']:
                warnings.append(f"Unsupported layers found: {len(analysis['unsupported_layers'])}")
            
            # Save circuit to database
            circuit_id = accelerator.save_circuit(circuit, verilog_code, metrics)
            
            execution_time = time.time() - start_time
            
            return TranspileResponse(
                success=True,
                circuit=CircuitSchema(
                    name=circuit.name,
                    layers=[],  # Simplified for response
                    connections=circuit.connections,
                    total_components=circuit.total_components,
                    pdk=request.pdk,
                    wavelength=request.wavelength
                ),
                verilog_code=verilog_code,
                metrics=CircuitMetricsSchema(
                    energy_per_op=metrics.energy_per_op,
                    latency=metrics.latency,
                    area=metrics.area,
                    power=metrics.power,
                    throughput=metrics.throughput,
                    accuracy=metrics.accuracy
                ),
                analysis={'circuit_id': circuit_id, 'compatibility_score': analysis['compatibility_score']},
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model transpilation failed: {e}")
            logger.error(traceback.format_exc())
            
            return TranspileResponse(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    @transpile_router.get("/verilog/{circuit_name}", response_class=PlainTextResponse)
    async def get_verilog_code(
        circuit_name: str = Path(..., description="Name of the circuit"),
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Get Verilog code for a specific circuit."""
        try:
            circuit = accelerator.load_circuit(circuit_name)
            if not circuit:
                raise HTTPException(status_code=404, detail=f"Circuit '{circuit_name}' not found")
            
            # Try to get stored Verilog, otherwise generate
            circuit_data = accelerator.circuit_repo.find_by_name(circuit_name)
            if circuit_data and circuit_data.get('verilog_code'):
                return circuit_data['verilog_code']
            else:
                verilog_code = circuit.generate_verilog()
                return verilog_code
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get Verilog code: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Benchmarking endpoints
    benchmark_router = APIRouter(prefix=f"{api_prefix}/benchmark", tags=["Benchmarking"])
    
    @benchmark_router.post("/performance", response_model=BenchmarkResponse)
    async def benchmark_performance(
        request: BenchmarkRequest,
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Benchmark photonic vs electronic performance."""
        start_time = time.time()
        
        try:
            # Decode model data
            model_bytes = base64.b64decode(request.model_data)
            model = torch.load(io.BytesIO(model_bytes), map_location='cpu')
            
            # Set accelerator configuration
            accelerator.pdk = request.pdk.value
            
            # Convert to photonic circuit
            circuit = accelerator.convert_pytorch_model(model)
            
            # Create test input
            test_input = np.random.randn(*request.input_shape).astype(np.float32)
            
            # Electronic baseline (if requested)
            electronic_metrics = None
            if request.include_electronic_baseline:
                electronic_start = time.time()
                
                # Simulate electronic inference
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input)
                    for _ in range(request.iterations):
                        _ = model(torch_input)
                
                electronic_time = (time.time() - electronic_start) * 1000  # Convert to ms
                electronic_inference_time = electronic_time / request.iterations
                
                electronic_metrics = BenchmarkMetrics(
                    inference_time_ms=electronic_inference_time,
                    throughput_ops=1000.0 / electronic_inference_time,
                    energy_per_inference_pj=150.0,  # Estimated GPU energy
                    power_mw=50000.0,  # Estimated GPU power
                    memory_usage_mb=100.0  # Estimated memory
                )
            
            # Photonic implementation
            photonic_start = time.time()
            
            # Simulate photonic inference
            for _ in range(request.iterations):
                _, inference_time = accelerator.simulate_inference(circuit, test_input)
            
            photonic_time = (time.time() - photonic_start) * 1000  # Convert to ms
            photonic_inference_time = photonic_time / request.iterations
            
            # Get circuit metrics
            circuit_metrics = accelerator.compile_and_profile(circuit)
            
            photonic_metrics = BenchmarkMetrics(
                inference_time_ms=photonic_inference_time,
                throughput_ops=1000.0 / photonic_inference_time,
                energy_per_inference_pj=circuit_metrics.energy_per_op,
                power_mw=circuit_metrics.power,
                memory_usage_mb=10.0  # Lower memory usage for photonic
            )
            
            # Calculate comparison metrics
            speedup_factor = None
            energy_efficiency_factor = None
            
            if electronic_metrics:
                speedup_factor = electronic_metrics.inference_time_ms / photonic_metrics.inference_time_ms
                energy_efficiency_factor = electronic_metrics.energy_per_inference_pj / photonic_metrics.energy_per_inference_pj
            
            total_time = time.time() - start_time
            
            return BenchmarkResponse(
                success=True,
                electronic_baseline=electronic_metrics,
                photonic_implementation=photonic_metrics,
                speedup_factor=speedup_factor,
                energy_efficiency_factor=energy_efficiency_factor,
                accuracy_comparison=circuit_metrics.accuracy,
                iterations_completed=request.iterations,
                total_execution_time=total_time
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Benchmarking failed: {e}")
            
            return BenchmarkResponse(
                success=False,
                iterations_completed=0,
                total_execution_time=total_time,
                error_message=str(e)
            )
    
    # Circuit management endpoints
    circuits_router = APIRouter(prefix=f"{api_prefix}/circuits", tags=["Circuits"])
    
    @circuits_router.get("/list", response_model=CircuitListResponse)
    async def list_circuits(
        limit: int = Query(50, ge=1, le=1000, description="Maximum circuits to return"),
        offset: int = Query(0, ge=0, description="Number of circuits to skip"),
        name_pattern: Optional[str] = Query(None, max_length=100, description="Filter by name pattern"),
        has_verilog: Optional[bool] = Query(None, description="Filter by Verilog availability"),
        has_metrics: Optional[bool] = Query(None, description="Filter by metrics availability"),
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """List saved photonic circuits with filtering options."""
        try:
            # Get circuits from database
            circuits_data = accelerator.list_saved_circuits(limit=limit + offset)
            
            # Apply filters
            filtered_circuits = []
            for circuit_data in circuits_data:
                # Name pattern filter
                if name_pattern and name_pattern.lower() not in circuit_data['name'].lower():
                    continue
                
                # Verilog filter
                if has_verilog is not None and circuit_data['has_verilog'] != has_verilog:
                    continue
                
                # Metrics filter
                if has_metrics is not None and circuit_data['has_metrics'] != has_metrics:
                    continue
                
                filtered_circuits.append(circuit_data)
            
            # Apply offset and limit
            total_count = len(filtered_circuits)
            paginated_circuits = filtered_circuits[offset:offset + limit]
            
            # Convert to response format
            circuit_infos = []
            for circuit_data in paginated_circuits:
                metrics_summary = None
                if circuit_data.get('energy_per_op'):
                    metrics_summary = {
                        'energy_per_op': circuit_data['energy_per_op'],
                        'latency': circuit_data['latency'],
                        'area': circuit_data['area']
                    }
                
                circuit_infos.append(CircuitInfo(
                    name=circuit_data['name'],
                    model_hash=circuit_data['model_hash'],
                    layer_count=circuit_data['layer_count'],
                    component_count=circuit_data['component_count'],
                    created_at=circuit_data['created_at'],
                    updated_at=circuit_data['updated_at'],
                    has_verilog=circuit_data['has_verilog'],
                    has_metrics=circuit_data['has_metrics'],
                    metrics_summary=metrics_summary
                ))
            
            return CircuitListResponse(
                success=True,
                circuits=circuit_infos,
                total_count=total_count,
                returned_count=len(circuit_infos),
                has_more=offset + len(circuit_infos) < total_count
            )
            
        except Exception as e:
            logger.error(f"Failed to list circuits: {e}")
            return CircuitListResponse(
                success=False,
                circuits=[],
                total_count=0,
                returned_count=0,
                has_more=False,
                error_message=str(e)
            )
    
    @circuits_router.get("/{circuit_name}", response_model=Dict[str, Any])
    async def get_circuit(
        circuit_name: str = Path(..., description="Name of the circuit"),
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Get detailed information about a specific circuit."""
        try:
            circuit = accelerator.load_circuit(circuit_name)
            if not circuit:
                raise HTTPException(status_code=404, detail=f"Circuit '{circuit_name}' not found")
            
            # Get additional data from database
            circuit_data = accelerator.circuit_repo.find_by_name(circuit_name)
            
            response_data = {
                'name': circuit.name,
                'layers': len(circuit.layers),
                'connections': circuit.connections,
                'total_components': circuit.total_components,
                'created_at': circuit_data.get('created_at') if circuit_data else None,
                'has_verilog': bool(circuit_data and circuit_data.get('verilog_code')),
                'has_metrics': bool(circuit_data and circuit_data.get('metrics'))
            }
            
            # Add metrics if available
            if circuit_data and circuit_data.get('metrics'):
                metrics = circuit_data['metrics']
                response_data['metrics'] = {
                    'energy_per_op': metrics.get('energy_per_op'),
                    'latency': metrics.get('latency'),
                    'area': metrics.get('area'),
                    'power': metrics.get('power'),
                    'throughput': metrics.get('throughput'),
                    'accuracy': metrics.get('accuracy')
                }
            
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get circuit: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @circuits_router.delete("/{circuit_name}", response_model=SuccessResponse)
    async def delete_circuit(
        circuit_name: str = Path(..., description="Name of the circuit"),
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Delete a specific circuit."""
        try:
            # Check if circuit exists
            circuit_data = accelerator.circuit_repo.find_by_name(circuit_name)
            if not circuit_data:
                raise HTTPException(status_code=404, detail=f"Circuit '{circuit_name}' not found")
            
            # Delete from database
            accelerator.circuit_repo.delete_by_name(circuit_name)
            
            return SuccessResponse(
                message=f"Circuit '{circuit_name}' deleted successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete circuit: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # System endpoints
    system_router = APIRouter(prefix=f"{api_prefix}/system", tags=["System"])
    
    @system_router.get("/stats", response_model=DatabaseStatsResponse)
    async def get_system_stats(
        accelerator: PhotonicAccelerator = Depends(get_accelerator)
    ):
        """Get system statistics including database and cache metrics."""
        try:
            stats = accelerator.get_database_stats()
            
            return DatabaseStatsResponse(
                success=True,
                database_stats=stats['database'],
                cache_stats=stats['cache'],
                circuit_stats=stats['circuit_stats']
            )
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return DatabaseStatsResponse(
                success=False,
                database_stats={},
                cache_stats={},
                circuit_stats={},
                error_message=str(e)
            )
    
    # Optimization endpoints  
    optimize_router = APIRouter(prefix=f"{api_prefix}/optimize", tags=["Optimization"])
    
    @optimize_router.post("/mzi-mesh", response_model=Dict[str, Any])
    async def optimize_mzi_mesh(
        weight_matrix: list,
        precision: int = Query(8, ge=1, le=32, description="Phase shifter precision"),
        loss_budget_db: float = Query(3.0, ge=0.1, le=10.0, description="Loss budget in dB")
    ):
        """Optimize MZI mesh configuration for given weight matrix."""
        try:
            # Convert to numpy array
            weights = np.array(weight_matrix)
            
            # Create optimizer
            optimizer = MZIMeshOptimizer(precision=precision, loss_budget_db=loss_budget_db)
            
            # Optimize mesh
            result = optimizer.optimize_mesh_topology(weights)
            
            return {
                'success': True,
                'optimization_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MZI mesh optimization failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    @optimize_router.post("/ring-resonator", response_model=Dict[str, Any])
    async def optimize_ring_resonator(
        weights: list,
        q_factor: float = Query(10000, ge=1000, le=100000, description="Quality factor"),
        fsr_ghz: float = Query(100, ge=10, le=1000, description="Free spectral range in GHz")
    ):
        """Optimize ring resonator bank configuration."""
        try:
            # Convert to numpy array
            weight_array = np.array(weights)
            
            # Create optimizer
            optimizer = RingResonatorOptimizer(q_factor=q_factor, fsr_ghz=fsr_ghz)
            
            # Optimize resonator bank
            result = optimizer.design_resonator_bank(weight_array)
            
            return {
                'success': True,
                'optimization_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ring resonator optimization failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Register all routers
    app.include_router(analysis_router)
    app.include_router(transpile_router)
    app.include_router(benchmark_router)
    app.include_router(circuits_router)
    app.include_router(system_router)
    app.include_router(optimize_router)
    
    logger.info("API endpoints registered successfully")