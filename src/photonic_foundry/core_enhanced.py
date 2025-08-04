"""
Enhanced core functionality with comprehensive error handling and validation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import hashlib
from pathlib import Path
import threading
from contextlib import contextmanager

from .core import PhotonicAccelerator as BaseAccelerator, CircuitMetrics, PhotonicCircuit
from .utils.validators import ModelValidator, PhotonicConfig, ConfigValidator
from .database import get_database, CircuitRepository, get_circuit_cache

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Fail on any validation error
    MODERATE = "moderate"  # Warn on minor issues, fail on major
    PERMISSIVE = "permissive"  # Log warnings but continue


@dataclass
class ProcessingResult:
    """Enhanced result wrapper with comprehensive metadata."""
    success: bool
    data: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    validation_report: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class EnhancedPhotonicAccelerator(BaseAccelerator):
    """Production-ready photonic accelerator with comprehensive error handling."""
    
    def __init__(self, 
                 pdk: str = "skywater130", 
                 wavelength: float = 1550.0,
                 validation_level: ValidationLevel = ValidationLevel.MODERATE,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced accelerator with validation and monitoring.
        
        Args:
            pdk: Process design kit name
            wavelength: Operating wavelength in nm  
            validation_level: How strict validation should be
            config: Additional configuration parameters
        """
        
        # Validate configuration
        if config is None:
            config = {}
            
        self.photonic_config, config_errors = ConfigValidator.validate_config({
            'pdk': pdk,
            'wavelength': wavelength,
            **config
        })
        
        if config_errors and validation_level == ValidationLevel.STRICT:
            raise ValueError(f"Configuration validation failed: {config_errors}")
        elif config_errors:
            logger.warning(f"Configuration issues: {config_errors}")
            
        # Initialize base class
        super().__init__(pdk, wavelength)
        
        self.validation_level = validation_level
        self.model_validator = ModelValidator(self.photonic_config)
        self.processing_stats = {
            'models_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self._processing_lock = threading.Lock()
        
        logger.info(f"Enhanced PhotonicAccelerator initialized: PDK={pdk}, λ={wavelength}nm, validation_level={validation_level.value}")
        
    @contextmanager
    def _processing_context(self, operation_name: str):
        """Context manager for processing operations with timing and error handling."""
        start_time = time.time()
        with self._processing_lock:
            self.processing_stats['models_processed'] += 1
            
        try:
            logger.info(f"Starting {operation_name}")
            yield
            
            processing_time = time.time() - start_time
            with self._processing_lock:
                self.processing_stats['successful_conversions'] += 1
                self.processing_stats['total_processing_time'] += processing_time
                
            logger.info(f"Completed {operation_name} in {processing_time:.3f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._processing_lock:
                self.processing_stats['failed_conversions'] += 1
                self.processing_stats['total_processing_time'] += processing_time
                
            logger.error(f"Failed {operation_name} after {processing_time:.3f}s: {e}")
            raise
            
    def convert_pytorch_model_safe(self, model: nn.Module, 
                                  validate_input: bool = True) -> ProcessingResult:
        """
        Safely convert PyTorch model with comprehensive validation and error handling.
        
        Args:
            model: PyTorch neural network model
            validate_input: Whether to validate the input model
            
        Returns:
            ProcessingResult with conversion details
        """
        result = ProcessingResult(success=False)
        
        try:
            with self._processing_context("model_conversion"):
                start_time = time.time()
                
                # Input validation
                if validate_input:
                    validation_report = self.model_validator.validate_model(model)
                    result.validation_report = validation_report
                    
                    if not validation_report['is_valid']:
                        if self.validation_level == ValidationLevel.STRICT:
                            result.errors = validation_report['errors']
                            result.warnings = validation_report['warnings']
                            return result
                        else:
                            result.warnings.extend(validation_report['warnings'])
                            logger.warning(f"Model validation issues: {validation_report['errors']}")
                
                # Check cache first
                model_hash = self._calculate_model_hash(model)
                cached_circuit = self._check_cache(model_hash)
                
                if cached_circuit:
                    with self._processing_lock:
                        self.processing_stats['cache_hits'] += 1
                    result.data = cached_circuit
                    result.metadata['cache_hit'] = True
                    logger.info("Retrieved circuit from cache")
                else:
                    with self._processing_lock:
                        self.processing_stats['cache_misses'] += 1
                    
                    # Perform conversion
                    circuit = self.convert_pytorch_model(model)
                    
                    # Cache the result
                    self._cache_circuit(model_hash, circuit)
                    
                    result.data = circuit
                    result.metadata['cache_hit'] = False
                
                result.processing_time = time.time() - start_time
                result.success = True
                result.metadata.update({
                    'model_class': model.__class__.__name__,
                    'model_hash': model_hash,
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'supported_layers': len([m for m in model.modules() 
                                           if type(m).__name__ in ['Linear', 'Conv2d']]),
                })
                
        except Exception as e:
            result.errors.append(f"Model conversion failed: {str(e)}")
            result.metadata['exception_type'] = type(e).__name__
            logger.error(f"Model conversion error: {e}", exc_info=True)
            
        return result
        
    def compile_and_profile_safe(self, circuit: PhotonicCircuit,
                                include_advanced_analysis: bool = False) -> ProcessingResult:
        """
        Safely compile and profile circuit with enhanced error handling.
        
        Args:
            circuit: PhotonicCircuit to analyze
            include_advanced_analysis: Whether to include detailed analysis
            
        Returns:
            ProcessingResult with profiling details
        """
        result = ProcessingResult(success=False)
        
        try:
            with self._processing_context("circuit_profiling"):
                start_time = time.time()
                
                # Basic profiling
                metrics = self.compile_and_profile(circuit)
                result.data = metrics
                
                # Advanced analysis
                if include_advanced_analysis:
                    advanced_metrics = {
                        'energy_breakdown': circuit.calculate_advanced_energy(),
                        'thermal_analysis': circuit.analyze_thermal_requirements(),
                        'component_utilization': self._analyze_component_utilization(circuit),
                        'scaling_analysis': self._analyze_scaling_characteristics(circuit)
                    }
                    result.metadata['advanced_analysis'] = advanced_metrics
                
                # Validate metrics for sanity
                if self._validate_metrics(metrics):
                    result.success = True
                else:
                    result.warnings.append("Generated metrics appear unrealistic")
                    if self.validation_level == ValidationLevel.STRICT:
                        result.errors.append("Metrics validation failed")
                        result.success = False
                    else:
                        result.success = True
                
                result.processing_time = time.time() - start_time
                result.metadata.update({
                    'circuit_layers': len(circuit.layers),
                    'total_components': circuit.total_components,
                    'analysis_level': 'advanced' if include_advanced_analysis else 'basic'
                })
                
        except Exception as e:
            result.errors.append(f"Circuit profiling failed: {str(e)}")
            result.metadata['exception_type'] = type(e).__name__
            logger.error(f"Circuit profiling error: {e}", exc_info=True)
            
        return result
        
    def simulate_inference_safe(self, circuit: PhotonicCircuit, 
                               input_data: np.ndarray,
                               monte_carlo_runs: int = 1) -> ProcessingResult:
        """
        Safely simulate inference with noise modeling and statistical analysis.
        
        Args:
            circuit: PhotonicCircuit to simulate
            input_data: Input data array
            monte_carlo_runs: Number of Monte Carlo simulation runs
            
        Returns:
            ProcessingResult with simulation details
        """
        result = ProcessingResult(success=False)
        
        try:
            with self._processing_context("inference_simulation"):
                start_time = time.time()
                
                # Validate input data
                if not self._validate_input_data(input_data):
                    result.errors.append("Invalid input data format or values")
                    return result
                
                # Run simulations
                simulation_results = []
                inference_times = []
                
                for run in range(monte_carlo_runs):
                    output, inference_time = self.simulate_inference(circuit, input_data)
                    simulation_results.append(output)
                    inference_times.append(inference_time)
                
                # Statistical analysis
                if monte_carlo_runs > 1:
                    outputs_array = np.stack(simulation_results)
                    result.data = {
                        'mean_output': np.mean(outputs_array, axis=0),
                        'std_output': np.std(outputs_array, axis=0),
                        'all_outputs': simulation_results,
                        'mean_inference_time': np.mean(inference_times),
                        'std_inference_time': np.std(inference_times)
                    }
                else:
                    result.data = {
                        'output': simulation_results[0],
                        'inference_time': inference_times[0]
                    }
                
                result.processing_time = time.time() - start_time
                result.success = True
                result.metadata.update({
                    'monte_carlo_runs': monte_carlo_runs,
                    'input_shape': input_data.shape,
                    'output_shape': simulation_results[0].shape,
                    'statistical_analysis': monte_carlo_runs > 1
                })
                
        except Exception as e:
            result.errors.append(f"Inference simulation failed: {str(e)}")
            result.metadata['exception_type'] = type(e).__name__
            logger.error(f"Inference simulation error: {e}", exc_info=True)
            
        return result
        
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        with self._processing_lock:
            stats = self.processing_stats.copy()
            
        # Calculate derived statistics
        if stats['models_processed'] > 0:
            stats['success_rate'] = stats['successful_conversions'] / stats['models_processed']
            stats['average_processing_time'] = stats['total_processing_time'] / stats['models_processed']
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
            
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0.0
            
        # Add system information
        stats.update({
            'validation_level': self.validation_level.value,
            'pdk': self.pdk,
            'wavelength': self.wavelength,
            'database_stats': self.get_database_stats()
        })
        
        return stats
        
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate hash of model for caching."""
        model_string = str(model)
        param_shapes = [str(p.shape) for p in model.parameters()]
        hash_input = model_string + "".join(param_shapes)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _check_cache(self, model_hash: str) -> Optional[PhotonicCircuit]:
        """Check if circuit is cached."""
        try:
            cache = get_circuit_cache()
            cached_data = cache.get_by_key(model_hash)
            if cached_data and 'circuit_data' in cached_data:
                # Reconstruct circuit from cached data
                # This is simplified - in production would need full deserialization
                return cached_data['circuit_data']
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        return None
        
    def _cache_circuit(self, model_hash: str, circuit: PhotonicCircuit):
        """Cache circuit for future use."""
        try:
            cache = get_circuit_cache()
            circuit_data = {
                'name': circuit.name,
                'layers': len(circuit.layers),
                'components': circuit.total_components
            }
            cache.put_circuit(circuit_data)
        except Exception as e:
            logger.warning(f"Circuit caching failed: {e}")
            
    def _validate_metrics(self, metrics: CircuitMetrics) -> bool:
        """Validate that metrics are realistic."""
        # Basic sanity checks
        if metrics.energy_per_op <= 0 or metrics.energy_per_op > 1000:  # pJ
            return False
        if metrics.latency <= 0 or metrics.latency > 1000000:  # ps
            return False
        if metrics.area <= 0 or metrics.area > 1000:  # mm²
            return False
        if metrics.accuracy < 0 or metrics.accuracy > 1:
            return False
        return True
        
    def _validate_input_data(self, input_data: np.ndarray) -> bool:
        """Validate input data for simulation."""
        if not isinstance(input_data, np.ndarray):
            return False
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            return False
        if input_data.size == 0:
            return False
        return True
        
    def _analyze_component_utilization(self, circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Analyze component utilization efficiency."""
        component_types = {}
        for layer in circuit.layers:
            for component in layer.components:
                comp_type = component['type'].value
                component_types[comp_type] = component_types.get(comp_type, 0) + 1
                
        return {
            'component_distribution': component_types,
            'utilization_efficiency': min(1.0, circuit.total_components / (len(circuit.layers) * 10)),
            'component_density': circuit.total_components / len(circuit.layers) if circuit.layers else 0
        }
        
    def _analyze_scaling_characteristics(self, circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Analyze how circuit characteristics scale."""
        total_params = sum(getattr(layer, 'weights', np.array([])).size for layer in circuit.layers)
        
        return {
            'parameter_to_component_ratio': total_params / max(circuit.total_components, 1),
            'layers_to_components_ratio': len(circuit.layers) / max(circuit.total_components, 1),
            'scaling_complexity': 'linear' if len(circuit.layers) < 10 else 'polynomial',
            'estimated_max_frequency_ghz': min(2.0, 10.0 / len(circuit.layers)) if circuit.layers else 2.0
        }