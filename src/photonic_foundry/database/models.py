"""
Data models for photonic circuits and components.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum


class ComponentType(Enum):
    """Enumeration of photonic component types."""
    MZI = "mach_zehnder_interferometer"
    RING = "ring_resonator"
    WAVEGUIDE = "waveguide"
    PHOTODETECTOR = "photodetector"
    MODULATOR = "electro_optic_modulator"
    SPLITTER = "optical_splitter"
    COUPLER = "directional_coupler"
    PHASE_SHIFTER = "phase_shifter"


@dataclass
class CircuitMetrics:
    """Performance metrics for photonic circuits."""
    energy_per_op: float  # pJ per operation
    latency: float        # ps
    area: float          # mm²
    power: float         # mW
    throughput: float    # GOPS
    accuracy: float      # relative to FP32
    loss: float          # dB total loss
    crosstalk: float     # dB crosstalk isolation
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CircuitMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ComponentSpec:
    """Specification for a photonic component."""
    name: str
    component_type: ComponentType
    pdk: str
    parameters: Dict[str, Any]
    verilog_template: Optional[str] = None
    performance_model: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['component_type'] = self.component_type.value
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentSpec':
        """Create from dictionary."""
        data = data.copy()
        data['component_type'] = ComponentType(data['component_type'])
        return cls(**data)


class CircuitModel:
    """Data model for photonic circuits."""
    
    def __init__(self, name: str, circuit_data: Dict[str, Any], 
                 model_hash: Optional[str] = None):
        self.name = name
        self.circuit_data = circuit_data
        self.model_hash = model_hash or self._compute_hash()
        self.verilog_code: Optional[str] = None
        self.metrics: Optional[CircuitMetrics] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.version = 1
        
    def _compute_hash(self) -> str:
        """Compute hash of circuit data for caching."""
        circuit_str = json.dumps(self.circuit_data, sort_keys=True)
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]
        
    def update_data(self, new_data: Dict[str, Any]):
        """Update circuit data and recompute hash."""
        self.circuit_data = new_data
        self.model_hash = self._compute_hash()
        self.updated_at = datetime.now()
        self.version += 1
        
    def set_verilog(self, verilog_code: str):
        """Set generated Verilog code."""
        self.verilog_code = verilog_code
        self.updated_at = datetime.now()
        
    def set_metrics(self, metrics: CircuitMetrics):
        """Set performance metrics."""
        self.metrics = metrics
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'name': self.name,
            'model_hash': self.model_hash,
            'circuit_data': json.dumps(self.circuit_data),
            'verilog_code': self.verilog_code,
            'metrics': json.dumps(self.metrics.to_dict()) if self.metrics else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitModel':
        """Create from dictionary."""
        circuit = cls(
            name=data['name'],
            circuit_data=json.loads(data['circuit_data']),
            model_hash=data['model_hash']
        )
        
        circuit.verilog_code = data.get('verilog_code')
        circuit.version = data.get('version', 1)
        
        if data.get('metrics'):
            circuit.metrics = CircuitMetrics.from_dict(json.loads(data['metrics']))
            
        if data.get('created_at'):
            circuit.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            circuit.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return circuit
        
    def get_layer_count(self) -> int:
        """Get number of layers in circuit."""
        return len(self.circuit_data.get('layers', []))
        
    def get_component_count(self) -> int:
        """Get total component count."""
        return self.circuit_data.get('total_components', 0)
        
    def get_estimated_cost(self) -> float:
        """Get estimated fabrication cost."""
        area_mm2 = self.metrics.area if self.metrics else 1.0
        cost_per_mm2 = 1000.0  # $1000 per mm² (rough estimate)
        return area_mm2 * cost_per_mm2


class ComponentModel:
    """Data model for photonic components."""
    
    def __init__(self, spec: ComponentSpec):
        self.spec = spec
        self.created_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'name': self.spec.name,
            'type': self.spec.component_type.value,
            'pdk': self.spec.pdk,
            'component_data': json.dumps(self.spec.to_dict()),
            'verilog_template': self.spec.verilog_template,
            'parameters': json.dumps(self.spec.parameters),
            'created_at': self.created_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentModel':
        """Create from dictionary."""
        spec_data = json.loads(data['component_data'])
        spec = ComponentSpec.from_dict(spec_data)
        
        component = cls(spec)
        if data.get('created_at'):
            component.created_at = datetime.fromisoformat(data['created_at'])
            
        return component


@dataclass
class SimulationResult:
    """Results from circuit simulation."""
    circuit_id: int
    input_data: np.ndarray
    output_data: np.ndarray
    simulation_config: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'circuit_id': self.circuit_id,
            'input_data': json.dumps(self.input_data.tolist()),
            'output_data': json.dumps(self.output_data.tolist()),
            'simulation_config': json.dumps(self.simulation_config),
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResult':
        """Create from dictionary."""
        return cls(
            circuit_id=data['circuit_id'],
            input_data=np.array(json.loads(data['input_data'])),
            output_data=np.array(json.loads(data['output_data'])),
            simulation_config=json.loads(data['simulation_config']),
            execution_time=data['execution_time'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        
    def get_accuracy_vs_expected(self, expected_output: np.ndarray) -> float:
        """Calculate accuracy compared to expected output."""
        if expected_output.shape != self.output_data.shape:
            return 0.0
            
        # Calculate normalized error
        error = np.abs(self.output_data - expected_output)
        max_val = np.max(np.abs(expected_output))
        
        if max_val == 0:
            return 1.0 if np.allclose(error, 0) else 0.0
            
        normalized_error = error / max_val
        return float(1.0 - np.mean(normalized_error))
        
    def get_snr_db(self) -> float:
        """Calculate signal-to-noise ratio in dB."""
        signal_power = np.mean(self.output_data ** 2)
        
        # Estimate noise from high-frequency components
        if len(self.output_data) > 1:
            diff = np.diff(self.output_data)
            noise_power = np.var(diff) / 2  # Rough noise estimate
        else:
            noise_power = signal_power * 0.01  # Assume 1% noise
            
        if noise_power == 0:
            return float('inf')
            
        snr = signal_power / noise_power
        return 10 * np.log10(snr) if snr > 0 else -float('inf')


@dataclass
class PDKModel:
    """Process Design Kit model."""
    name: str
    version: str
    description: str
    config_data: Dict[str, Any]
    component_library: Dict[str, ComponentSpec]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        # Convert component library to serializable format
        component_lib_data = {
            name: spec.to_dict() 
            for name, spec in self.component_library.items()
        }
        
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'config_data': json.dumps(self.config_data),
            'component_library': json.dumps(component_lib_data)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PDKModel':
        """Create from dictionary."""
        config_data = json.loads(data['config_data'])
        component_lib_data = json.loads(data['component_library'])
        
        # Convert component library back to ComponentSpec objects
        component_library = {
            name: ComponentSpec.from_dict(spec_data)
            for name, spec_data in component_lib_data.items()
        }
        
        return cls(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            config_data=config_data,
            component_library=component_library
        )
        
    def get_component(self, name: str) -> Optional[ComponentSpec]:
        """Get component specification by name."""
        return self.component_library.get(name)
        
    def list_components(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """List available components, optionally filtered by type."""
        if component_type is None:
            return list(self.component_library.keys())
        else:
            return [
                name for name, spec in self.component_library.items()
                if spec.component_type == component_type
            ]


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    circuit_id: int
    metric_name: str
    metric_value: float
    unit: str
    measurement_config: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'circuit_id': self.circuit_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'unit': self.unit,
            'measurement_config': json.dumps(self.measurement_config),
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from dictionary."""
        return cls(
            circuit_id=data['circuit_id'],
            metric_name=data['metric_name'],
            metric_value=data['metric_value'],
            unit=data['unit'],
            measurement_config=json.loads(data['measurement_config']),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )