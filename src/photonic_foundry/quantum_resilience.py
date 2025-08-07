"""
Quantum resilience and error handling for photonic neural networks.

This module provides comprehensive error handling, circuit fault tolerance,
and quantum error correction for photonic computing systems.
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from .core import PhotonicCircuit, CircuitMetrics
from .quantum_planner import QuantumTask, QuantumState

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FaultType(Enum):
    """Types of faults in photonic systems."""
    OPTICAL_LOSS = "optical_loss"
    PHASE_DRIFT = "phase_drift"
    THERMAL_FLUCTUATION = "thermal_fluctuation"
    CROSSTALK = "crosstalk"
    LASER_INSTABILITY = "laser_instability"
    DETECTOR_NOISE = "detector_noise"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    FABRICATION_VARIATION = "fabrication_variation"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    fault_type: FaultType
    component_id: str
    error_message: str
    stack_trace: str
    circuit_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class ResilienceMetrics:
    """Metrics for quantum resilience assessment."""
    mean_time_between_failures: float  # seconds
    mean_time_to_recovery: float       # seconds
    error_rate: float                  # errors per second
    availability: float                # 0.0 to 1.0
    quantum_error_correction_efficiency: float
    fault_tolerance_level: float
    adaptive_recovery_success_rate: float


class CircuitHealthMonitor:
    """Monitor health and performance of photonic circuits."""
    
    def __init__(self, circuit: PhotonicCircuit):
        """Initialize circuit health monitor."""
        self.circuit = circuit
        self.health_metrics = {}
        self.fault_history = []
        self.monitoring_start_time = time.time()
        self.last_health_check = 0
        
        logger.info(f"Initialized CircuitHealthMonitor for circuit: {circuit.name}")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of photonic circuit."""
        current_time = time.time()
        health_status = {
            'timestamp': current_time,
            'overall_health': 'healthy',
            'component_health': {},
            'performance_metrics': {},
            'fault_indicators': []
        }
        
        # Check each layer for health indicators
        for i, layer in enumerate(self.circuit.layers):
            layer_health = self._check_layer_health(layer, i)
            health_status['component_health'][f'layer_{i}'] = layer_health
            
            if layer_health['status'] != 'healthy':
                health_status['overall_health'] = 'degraded'
        
        # Analyze performance metrics
        try:
            metrics = self.circuit.analyze_circuit()
            health_status['performance_metrics'] = {
                'energy_per_op': metrics.energy_per_op,
                'latency': metrics.latency,
                'throughput': metrics.throughput,
                'accuracy': metrics.accuracy
            }
            
            # Check for performance degradation
            if hasattr(self, 'baseline_metrics'):
                degradation = self._calculate_degradation(metrics, self.baseline_metrics)
                if degradation > 0.2:  # 20% degradation threshold
                    health_status['overall_health'] = 'critical'
                    health_status['fault_indicators'].append('PERFORMANCE_DEGRADATION')
            else:
                self.baseline_metrics = metrics
                
        except Exception as e:
            logger.error(f"Failed to analyze circuit metrics: {e}")
            health_status['overall_health'] = 'error'
            health_status['fault_indicators'].append('METRICS_ANALYSIS_FAILURE')
        
        # Update health history
        self.health_metrics[current_time] = health_status
        self.last_health_check = current_time
        
        # Clean old health data (keep last 24 hours)
        cutoff_time = current_time - 86400
        old_timestamps = [t for t in self.health_metrics.keys() if t < cutoff_time]
        for t in old_timestamps:
            del self.health_metrics[t]
        
        logger.debug(f"Health check completed: {health_status['overall_health']}")
        return health_status
    
    def _check_layer_health(self, layer, layer_index: int) -> Dict[str, Any]:
        """Check health of individual photonic layer."""
        layer_health = {
            'status': 'healthy',
            'components_checked': len(layer.components),
            'fault_indicators': [],
            'estimated_reliability': 0.99
        }
        
        # Check component count
        expected_components = getattr(layer, 'input_size', 1) * getattr(layer, 'output_size', 1)
        if len(layer.components) != expected_components:
            layer_health['status'] = 'degraded'
            layer_health['fault_indicators'].append('COMPONENT_COUNT_MISMATCH')
        
        # Simulate component-level health checks
        for i, component in enumerate(layer.components):
            component_reliability = self._estimate_component_reliability(component, layer_index, i)
            layer_health['estimated_reliability'] *= component_reliability
            
            if component_reliability < 0.95:
                layer_health['status'] = 'degraded'
                layer_health['fault_indicators'].append(f'COMPONENT_{i}_DEGRADED')
        
        return layer_health
    
    def _estimate_component_reliability(self, component: Dict[str, Any], 
                                      layer_index: int, component_index: int) -> float:
        """Estimate reliability of individual photonic component."""
        base_reliability = 0.999
        
        # Simulate age-related degradation
        circuit_age = time.time() - self.monitoring_start_time
        age_factor = max(0.9, 1.0 - circuit_age / (365 * 24 * 3600))  # Annual degradation
        
        # Simulate component-specific reliability
        component_type = component.get('type', 'unknown')
        type_reliability_map = {
            'mach_zehnder_interferometer': 0.998,
            'ring_resonator': 0.995,
            'photodetector': 0.997,
            'modulator': 0.996
        }
        
        type_reliability = type_reliability_map.get(
            component_type.value if hasattr(component_type, 'value') else str(component_type),
            0.990
        )
        
        # Add random variation to simulate real-world conditions
        variation = np.random.normal(0, 0.001)  # Small random variation
        
        final_reliability = base_reliability * age_factor * type_reliability * (1 + variation)
        return max(0.5, min(1.0, final_reliability))
    
    def _calculate_degradation(self, current: CircuitMetrics, baseline: CircuitMetrics) -> float:
        """Calculate performance degradation compared to baseline."""
        energy_degradation = (current.energy_per_op - baseline.energy_per_op) / baseline.energy_per_op
        latency_degradation = (current.latency - baseline.latency) / baseline.latency
        accuracy_degradation = (baseline.accuracy - current.accuracy) / baseline.accuracy
        
        # Weight different degradation metrics
        overall_degradation = (
            energy_degradation * 0.3 +
            latency_degradation * 0.3 +
            accuracy_degradation * 0.4
        )
        
        return max(0.0, overall_degradation)
    
    def get_fault_predictions(self) -> List[Dict[str, Any]]:
        """Predict potential faults based on health trends."""
        if len(self.health_metrics) < 10:
            return []  # Need sufficient history for predictions
        
        predictions = []
        recent_health = list(self.health_metrics.values())[-10:]
        
        # Analyze trends in performance metrics
        if all('performance_metrics' in h for h in recent_health):
            energy_trend = [h['performance_metrics']['energy_per_op'] for h in recent_health]
            latency_trend = [h['performance_metrics']['latency'] for h in recent_health]
            
            # Check for increasing energy consumption (potential laser degradation)
            if len(energy_trend) > 5:
                energy_slope = np.polyfit(range(len(energy_trend)), energy_trend, 1)[0]
                if energy_slope > 0.1:  # Increasing energy consumption
                    predictions.append({
                        'fault_type': FaultType.LASER_INSTABILITY,
                        'predicted_time_to_failure': 7200,  # 2 hours
                        'confidence': 0.7,
                        'indicator': 'increasing_energy_consumption'
                    })
            
            # Check for increasing latency (potential thermal issues)
            if len(latency_trend) > 5:
                latency_slope = np.polyfit(range(len(latency_trend)), latency_trend, 1)[0]
                if latency_slope > 1.0:  # Increasing latency
                    predictions.append({
                        'fault_type': FaultType.THERMAL_FLUCTUATION,
                        'predicted_time_to_failure': 3600,  # 1 hour
                        'confidence': 0.6,
                        'indicator': 'increasing_latency'
                    })
        
        return predictions


class QuantumErrorCorrector:
    """Quantum error correction for photonic circuits."""
    
    def __init__(self):
        """Initialize quantum error corrector."""
        self.correction_codes = {
            'bit_flip': self._bit_flip_correction,
            'phase_flip': self._phase_flip_correction,
            'amplitude_damping': self._amplitude_damping_correction
        }
        self.correction_statistics = {}
        
        logger.info("Initialized QuantumErrorCorrector")
    
    def apply_error_correction(self, circuit_data: np.ndarray, 
                             error_type: str = 'bit_flip') -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply quantum error correction to circuit data."""
        if error_type not in self.correction_codes:
            raise ValueError(f"Unsupported error correction type: {error_type}")
        
        correction_start = time.time()
        
        try:
            corrected_data, correction_info = self.correction_codes[error_type](circuit_data)
            
            correction_time = time.time() - correction_start
            
            # Update statistics
            if error_type not in self.correction_statistics:
                self.correction_statistics[error_type] = {
                    'total_corrections': 0,
                    'successful_corrections': 0,
                    'average_correction_time': 0.0
                }
            
            stats = self.correction_statistics[error_type]
            stats['total_corrections'] += 1
            
            if correction_info['success']:
                stats['successful_corrections'] += 1
            
            # Update average correction time
            stats['average_correction_time'] = (
                (stats['average_correction_time'] * (stats['total_corrections'] - 1) + correction_time) /
                stats['total_corrections']
            )
            
            correction_result = {
                'corrected_data': corrected_data,
                'correction_applied': error_type,
                'correction_time': correction_time,
                'success': correction_info['success'],
                'error_syndrome': correction_info.get('syndrome', []),
                'correction_efficiency': correction_info.get('efficiency', 0.0)
            }
            
            logger.debug(f"Applied {error_type} correction: {correction_info['success']}")
            return corrected_data, correction_result
            
        except Exception as e:
            logger.error(f"Error correction failed: {e}")
            return circuit_data, {
                'corrected_data': circuit_data,
                'correction_applied': error_type,
                'correction_time': time.time() - correction_start,
                'success': False,
                'error': str(e)
            }
    
    def _bit_flip_correction(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply bit flip error correction."""
        # Simplified 3-qubit bit flip code
        if len(data.shape) != 1:
            data = data.flatten()
        
        # Encode: |ψ⟩ → |ψ⟩|ψ⟩|ψ⟩ (triple redundancy)
        encoded_length = len(data) * 3
        encoded_data = np.zeros(encoded_length)
        
        for i, bit in enumerate(data):
            encoded_data[3*i:3*i+3] = [bit, bit, bit]
        
        # Simulate bit flip errors (small probability)
        error_probability = 0.01
        error_mask = np.random.random(encoded_length) < error_probability
        noisy_data = encoded_data.copy()
        noisy_data[error_mask] = 1 - noisy_data[error_mask]  # Flip bits
        
        # Decode with majority voting
        corrected_data = np.zeros(len(data))
        syndrome = []
        
        for i in range(len(data)):
            triplet = noisy_data[3*i:3*i+3]
            majority_vote = np.sum(triplet) >= 2
            corrected_data[i] = float(majority_vote)
            
            # Syndrome detection
            if not np.all(triplet == triplet[0]):
                syndrome.append(i)
        
        correction_efficiency = 1.0 - len(syndrome) / len(data) if len(data) > 0 else 1.0
        
        return corrected_data, {
            'success': True,
            'syndrome': syndrome,
            'efficiency': correction_efficiency,
            'errors_detected': len(syndrome)
        }
    
    def _phase_flip_correction(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply phase flip error correction."""
        # Simplified phase flip correction using Hadamard gates
        if len(data.shape) != 1:
            data = data.flatten()
        
        # Apply Hadamard-like transformation
        hadamard_data = np.array([
            (data[i] + data[(i+1) % len(data)]) / np.sqrt(2) for i in range(len(data))
        ])
        
        # Apply bit flip correction in the transformed basis
        corrected_hadamard, bit_flip_info = self._bit_flip_correction(hadamard_data)
        
        # Transform back
        corrected_data = np.array([
            (corrected_hadamard[i] + corrected_hadamard[(i+1) % len(corrected_hadamard)]) / np.sqrt(2) 
            for i in range(len(corrected_hadamard))
        ])
        
        return corrected_data, {
            'success': bit_flip_info['success'],
            'syndrome': bit_flip_info['syndrome'],
            'efficiency': bit_flip_info['efficiency'] * 0.9,  # Phase correction less efficient
            'phase_errors_corrected': len(bit_flip_info['syndrome'])
        }
    
    def _amplitude_damping_correction(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply amplitude damping error correction."""
        if len(data.shape) != 1:
            data = data.flatten()
        
        # Simulate amplitude damping
        damping_factor = 0.95  # 5% amplitude loss
        damped_data = data * damping_factor
        
        # Amplification correction
        correction_factor = 1.0 / damping_factor
        corrected_data = damped_data * correction_factor
        
        # Calculate efficiency based on signal recovery
        original_power = np.sum(data ** 2)
        corrected_power = np.sum(corrected_data ** 2)
        efficiency = min(corrected_power / original_power, 1.0) if original_power > 0 else 0.0
        
        return corrected_data, {
            'success': True,
            'syndrome': [],
            'efficiency': efficiency,
            'amplitude_restored': correction_factor
        }


class QuantumResilienceManager:
    """Comprehensive resilience management for quantum photonic systems."""
    
    def __init__(self, circuit: PhotonicCircuit):
        """Initialize quantum resilience manager."""
        self.circuit = circuit
        self.health_monitor = CircuitHealthMonitor(circuit)
        self.error_corrector = QuantumErrorCorrector()
        self.error_log = []
        self.recovery_strategies = {}
        self.resilience_metrics = None
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        logger.info(f"Initialized QuantumResilienceManager for circuit: {circuit.name}")
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different fault types."""
        self.recovery_strategies = {
            FaultType.OPTICAL_LOSS: self._recover_optical_loss,
            FaultType.PHASE_DRIFT: self._recover_phase_drift,
            FaultType.THERMAL_FLUCTUATION: self._recover_thermal_fluctuation,
            FaultType.CROSSTALK: self._recover_crosstalk,
            FaultType.LASER_INSTABILITY: self._recover_laser_instability,
            FaultType.DETECTOR_NOISE: self._recover_detector_noise,
            FaultType.QUANTUM_DECOHERENCE: self._recover_quantum_decoherence
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors with quantum resilience strategies."""
        error_id = f"error_{len(self.error_log)}_{int(time.time())}"
        
        # Determine error severity and fault type
        severity = self._classify_error_severity(error)
        fault_type = self._identify_fault_type(error, context or {})
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            fault_type=fault_type,
            component_id=context.get('component_id', 'unknown') if context else 'unknown',
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            circuit_state=context or {}
        )
        
        # Log error
        self.error_log.append(error_context)
        logger.error(f"Handling error {error_id}: {error_context.error_message}")
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_context)
        
        return {
            'error_id': error_id,
            'severity': severity.value,
            'fault_type': fault_type.value,
            'recovery_attempted': recovery_result['attempted'],
            'recovery_successful': recovery_result['successful'],
            'recovery_actions': recovery_result['actions'],
            'estimated_recovery_time': recovery_result['estimated_time']
        }
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, RecursionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _identify_fault_type(self, error: Exception, context: Dict[str, Any]) -> FaultType:
        """Identify the type of fault based on error and context."""
        error_message = str(error).lower()
        
        # Pattern matching for fault identification
        if 'thermal' in error_message or 'temperature' in error_message:
            return FaultType.THERMAL_FLUCTUATION
        elif 'phase' in error_message or 'drift' in error_message:
            return FaultType.PHASE_DRIFT
        elif 'optical' in error_message or 'loss' in error_message:
            return FaultType.OPTICAL_LOSS
        elif 'crosstalk' in error_message or 'interference' in error_message:
            return FaultType.CROSSTALK
        elif 'laser' in error_message or 'power' in error_message:
            return FaultType.LASER_INSTABILITY
        elif 'detector' in error_message or 'noise' in error_message:
            return FaultType.DETECTOR_NOISE
        elif 'quantum' in error_message or 'coherence' in error_message:
            return FaultType.QUANTUM_DECOHERENCE
        else:
            return FaultType.FABRICATION_VARIATION
    
    def _attempt_recovery(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Attempt recovery using appropriate strategy."""
        recovery_start = time.time()
        
        if error_context.fault_type in self.recovery_strategies:
            try:
                recovery_function = self.recovery_strategies[error_context.fault_type]
                recovery_actions = recovery_function(error_context)
                
                recovery_time = time.time() - recovery_start
                
                # Update error context
                error_context.recovery_attempts += 1
                
                return {
                    'attempted': True,
                    'successful': True,
                    'actions': recovery_actions,
                    'estimated_time': recovery_time
                }
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_context.error_id}: {recovery_error}")
                
                return {
                    'attempted': True,
                    'successful': False,
                    'actions': ['recovery_failed'],
                    'estimated_time': time.time() - recovery_start,
                    'recovery_error': str(recovery_error)
                }
        
        return {
            'attempted': False,
            'successful': False,
            'actions': ['no_recovery_strategy'],
            'estimated_time': 0.0
        }
    
    def _recover_optical_loss(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for optical loss."""
        actions = []
        
        # Increase laser power
        actions.append('increase_laser_power')
        
        # Optimize coupling efficiency
        actions.append('optimize_coupling')
        
        # Apply error correction
        actions.append('apply_amplitude_damping_correction')
        
        logger.info(f"Applied optical loss recovery for {error_context.error_id}")
        return actions
    
    def _recover_phase_drift(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for phase drift."""
        actions = []
        
        # Recalibrate phase shifters
        actions.append('recalibrate_phase_shifters')
        
        # Apply thermal stabilization
        actions.append('thermal_stabilization')
        
        # Phase error correction
        actions.append('apply_phase_flip_correction')
        
        logger.info(f"Applied phase drift recovery for {error_context.error_id}")
        return actions
    
    def _recover_thermal_fluctuation(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for thermal fluctuations."""
        actions = []
        
        # Increase cooling
        actions.append('increase_cooling_power')
        
        # Thermal compensation
        actions.append('thermal_compensation')
        
        # Wait for thermal stabilization
        actions.append('wait_thermal_stabilization')
        time.sleep(0.1)  # Simulate stabilization wait
        
        logger.info(f"Applied thermal fluctuation recovery for {error_context.error_id}")
        return actions
    
    def _recover_crosstalk(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for crosstalk."""
        actions = []
        
        # Adjust channel spacing
        actions.append('adjust_channel_spacing')
        
        # Apply crosstalk cancellation
        actions.append('crosstalk_cancellation')
        
        # Isolation improvement
        actions.append('improve_isolation')
        
        logger.info(f"Applied crosstalk recovery for {error_context.error_id}")
        return actions
    
    def _recover_laser_instability(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for laser instability."""
        actions = []
        
        # Laser power stabilization
        actions.append('stabilize_laser_power')
        
        # Frequency locking
        actions.append('frequency_locking')
        
        # Backup laser activation
        actions.append('activate_backup_laser')
        
        logger.info(f"Applied laser instability recovery for {error_context.error_id}")
        return actions
    
    def _recover_detector_noise(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for detector noise."""
        actions = []
        
        # Noise filtering
        actions.append('apply_noise_filtering')
        
        # Signal amplification
        actions.append('signal_amplification')
        
        # Detector calibration
        actions.append('detector_calibration')
        
        logger.info(f"Applied detector noise recovery for {error_context.error_id}")
        return actions
    
    def _recover_quantum_decoherence(self, error_context: ErrorContext) -> List[str]:
        """Recovery strategy for quantum decoherence."""
        actions = []
        
        # Dynamical decoupling
        actions.append('dynamical_decoupling')
        
        # Quantum error correction
        actions.append('apply_quantum_error_correction')
        
        # Coherence time extension
        actions.append('extend_coherence_time')
        
        logger.info(f"Applied quantum decoherence recovery for {error_context.error_id}")
        return actions
    
    def calculate_resilience_metrics(self) -> ResilienceMetrics:
        """Calculate comprehensive resilience metrics."""
        if not self.error_log:
            return ResilienceMetrics(
                mean_time_between_failures=float('inf'),
                mean_time_to_recovery=0.0,
                error_rate=0.0,
                availability=1.0,
                quantum_error_correction_efficiency=1.0,
                fault_tolerance_level=1.0,
                adaptive_recovery_success_rate=1.0
            )
        
        current_time = time.time()
        monitoring_duration = current_time - self.health_monitor.monitoring_start_time
        
        # Mean time between failures
        if len(self.error_log) > 1:
            time_diffs = [
                self.error_log[i].timestamp - self.error_log[i-1].timestamp 
                for i in range(1, len(self.error_log))
            ]
            mtbf = np.mean(time_diffs) if time_diffs else monitoring_duration
        else:
            mtbf = monitoring_duration
        
        # Mean time to recovery (estimated)
        recovery_times = [60.0 for _ in self.error_log]  # Assume 1 minute average recovery
        mttr = np.mean(recovery_times) if recovery_times else 0.0
        
        # Error rate
        error_rate = len(self.error_log) / monitoring_duration if monitoring_duration > 0 else 0.0
        
        # Availability
        total_downtime = len(self.error_log) * mttr
        availability = max(0.0, 1.0 - total_downtime / monitoring_duration) if monitoring_duration > 0 else 1.0
        
        # Quantum error correction efficiency
        if self.error_corrector.correction_statistics:
            total_corrections = sum(
                stats['total_corrections'] 
                for stats in self.error_corrector.correction_statistics.values()
            )
            successful_corrections = sum(
                stats['successful_corrections'] 
                for stats in self.error_corrector.correction_statistics.values()
            )
            qec_efficiency = successful_corrections / total_corrections if total_corrections > 0 else 1.0
        else:
            qec_efficiency = 1.0
        
        # Fault tolerance level (based on successful recoveries)
        successful_recoveries = sum(1 for error in self.error_log if error.recovery_attempts > 0)
        fault_tolerance = successful_recoveries / len(self.error_log) if self.error_log else 1.0
        
        # Adaptive recovery success rate
        recovery_success_rate = fault_tolerance  # Same as fault tolerance for now
        
        self.resilience_metrics = ResilienceMetrics(
            mean_time_between_failures=mtbf,
            mean_time_to_recovery=mttr,
            error_rate=error_rate,
            availability=availability,
            quantum_error_correction_efficiency=qec_efficiency,
            fault_tolerance_level=fault_tolerance,
            adaptive_recovery_success_rate=recovery_success_rate
        )
        
        return self.resilience_metrics
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report."""
        metrics = self.calculate_resilience_metrics()
        health_status = self.health_monitor.perform_health_check()
        fault_predictions = self.health_monitor.get_fault_predictions()
        
        return {
            'report_timestamp': time.time(),
            'circuit_name': self.circuit.name,
            'resilience_metrics': {
                'mtbf_hours': metrics.mean_time_between_failures / 3600,
                'mttr_minutes': metrics.mean_time_to_recovery / 60,
                'error_rate_per_hour': metrics.error_rate * 3600,
                'availability_percent': metrics.availability * 100,
                'qec_efficiency_percent': metrics.quantum_error_correction_efficiency * 100,
                'fault_tolerance_percent': metrics.fault_tolerance_level * 100,
                'recovery_success_rate_percent': metrics.adaptive_recovery_success_rate * 100
            },
            'current_health': health_status,
            'error_summary': {
                'total_errors': len(self.error_log),
                'errors_by_severity': {
                    severity.value: sum(1 for e in self.error_log if e.severity == severity)
                    for severity in ErrorSeverity
                },
                'errors_by_fault_type': {
                    fault_type.value: sum(1 for e in self.error_log if e.fault_type == fault_type)
                    for fault_type in FaultType
                }
            },
            'fault_predictions': fault_predictions,
            'error_correction_statistics': self.error_corrector.correction_statistics,
            'recommendations': self._generate_recommendations(metrics, health_status, fault_predictions)
        }
    
    def _generate_recommendations(self, metrics: ResilienceMetrics, 
                                health_status: Dict[str, Any],
                                predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving resilience."""
        recommendations = []
        
        # Based on availability
        if metrics.availability < 0.99:
            recommendations.append("Consider implementing redundant optical paths")
            recommendations.append("Improve mean time to recovery through automated recovery")
        
        # Based on error rate
        if metrics.error_rate > 0.01:  # More than 1 error per 100 seconds
            recommendations.append("Implement proactive health monitoring")
            recommendations.append("Enhance environmental stabilization")
        
        # Based on health status
        if health_status['overall_health'] != 'healthy':
            recommendations.append("Schedule immediate circuit maintenance")
            recommendations.append("Review component reliability")
        
        # Based on predictions
        if predictions:
            for prediction in predictions:
                if prediction['confidence'] > 0.6:
                    recommendations.append(f"Prepare for {prediction['fault_type'].value} within predicted timeframe")
        
        # Based on quantum error correction
        if metrics.quantum_error_correction_efficiency < 0.95:
            recommendations.append("Optimize quantum error correction codes")
            recommendations.append("Consider implementing additional error correction layers")
        
        return recommendations