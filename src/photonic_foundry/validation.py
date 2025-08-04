"""
Input validation and error handling utilities for photonic circuits.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class CircuitValidator:
    """Comprehensive validation for photonic circuits."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.supported_components = {
            'mach_zehnder_interferometer',
            'ring_resonator',
            'waveguide',
            'photodetector',
            'electro_optic_modulator'
        }
        self.supported_layer_types = {
            'linear', 'activation', 'conv2d', 'pooling'
        }
        
    def validate_circuit(self, circuit_data: Dict[str, Any]) -> ValidationResult:
        """Validate complete circuit structure."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate required fields
        self._validate_required_fields(circuit_data, result)
        
        # Validate circuit metadata
        self._validate_circuit_metadata(circuit_data, result)
        
        # Validate layers
        self._validate_layers(circuit_data.get('layers', []), result)
        
        # Validate connections
        self._validate_connections(circuit_data, result)
        
        # Validate component counts
        self._validate_component_counts(circuit_data, result)
        
        # Physical constraints validation
        self._validate_physical_constraints(circuit_data, result)
        
        # Create sanitized version
        if result.is_valid or self.validation_level == ValidationLevel.RELAXED:
            result.sanitized_data = self._sanitize_circuit_data(circuit_data)
            
        return result
        
    def _validate_required_fields(self, circuit_data: Dict[str, Any], result: ValidationResult):
        """Validate required circuit fields."""
        required_fields = ['name', 'layers', 'total_components']
        
        for field in required_fields:
            if field not in circuit_data:
                result.add_error(f"Missing required field: {field}")
            elif circuit_data[field] is None:
                result.add_error(f"Field '{field}' cannot be None")
                
    def _validate_circuit_metadata(self, circuit_data: Dict[str, Any], result: ValidationResult):
        """Validate circuit metadata."""
        # Name validation
        name = circuit_data.get('name', '')
        if not isinstance(name, str) or len(name) == 0:
            result.add_error("Circuit name must be a non-empty string")
        elif len(name) > 64:
            result.add_warning("Circuit name is very long (>64 chars)")
            
        # PDK validation
        pdk = circuit_data.get('pdk', 'unknown')
        supported_pdks = {'skywater130', 'gf180mcu', 'tsmc28', 'generic'}
        if pdk not in supported_pdks:
            result.add_warning(f"PDK '{pdk}' not in supported list: {supported_pdks}")
            
        # Wavelength validation
        wavelength = circuit_data.get('wavelength', 1550)
        if not isinstance(wavelength, (int, float)):
            result.add_error("Wavelength must be a number")
        elif wavelength < 1260 or wavelength > 1625:
            result.add_warning(f"Wavelength {wavelength}nm outside typical range (1260-1625nm)")
            
    def _validate_layers(self, layers: List[Dict[str, Any]], result: ValidationResult):
        """Validate circuit layers."""
        if not isinstance(layers, list):
            result.add_error("Layers must be a list")
            return
            
        if len(layers) == 0:
            result.add_error("Circuit must have at least one layer")
            return
            
        for i, layer in enumerate(layers):
            self._validate_single_layer(layer, i, result)
            
        # Validate layer connectivity
        self._validate_layer_connectivity(layers, result)
        
    def _validate_single_layer(self, layer: Dict[str, Any], layer_idx: int, result: ValidationResult):
        """Validate a single layer."""
        layer_prefix = f"Layer {layer_idx}"
        
        # Required fields
        required_fields = ['type', 'input_size', 'output_size']
        for field in required_fields:
            if field not in layer:
                result.add_error(f"{layer_prefix}: Missing required field '{field}'")
                
        # Layer type validation
        layer_type = layer.get('type', '')
        if layer_type not in self.supported_layer_types:
            result.add_error(f"{layer_prefix}: Unsupported layer type '{layer_type}'")
            
        # Size validation
        input_size = layer.get('input_size', 0)
        output_size = layer.get('output_size', 0)
        
        if not isinstance(input_size, int) or input_size <= 0:
            result.add_error(f"{layer_prefix}: input_size must be positive integer")
        if not isinstance(output_size, int) or output_size <= 0:
            result.add_error(f"{layer_prefix}: output_size must be positive integer")
            
        # Size limits
        if input_size > 2048 or output_size > 2048:
            result.add_warning(f"{layer_prefix}: Very large layer size may cause performance issues")
            
        # Component validation
        self._validate_layer_components(layer, layer_idx, result)
        
    def _validate_layer_components(self, layer: Dict[str, Any], layer_idx: int, result: ValidationResult):
        """Validate layer components."""
        layer_prefix = f"Layer {layer_idx}"
        components = layer.get('components', [])
        
        if not isinstance(components, list):
            result.add_error(f"{layer_prefix}: Components must be a list")
            return
            
        for comp_idx, component in enumerate(components):
            comp_type = component.get('type', '')
            if comp_type not in self.supported_components:
                result.add_error(f"{layer_prefix}.{comp_idx}: Unsupported component type '{comp_type}'")
                
            # Validate component parameters
            params = component.get('params', {})
            if not isinstance(params, dict):
                result.add_error(f"{layer_prefix}.{comp_idx}: Component params must be a dictionary")
                
    def _validate_layer_connectivity(self, layers: List[Dict[str, Any]], result: ValidationResult):
        """Validate that layer sizes are compatible."""
        for i in range(len(layers) - 1):
            current_output = layers[i].get('output_size', 0)
            next_input = layers[i + 1].get('input_size', 0)
            
            if current_output != next_input:
                result.add_error(f"Size mismatch: Layer {i} output ({current_output}) != Layer {i+1} input ({next_input})")
                
    def _validate_connections(self, circuit_data: Dict[str, Any], result: ValidationResult):
        """Validate circuit connections."""
        connections = circuit_data.get('connections', [])
        layers = circuit_data.get('layers', [])
        num_layers = len(layers)
        
        if not isinstance(connections, list):
            result.add_error("Connections must be a list")
            return
            
        for conn_idx, connection in enumerate(connections):
            if not isinstance(connection, (list, tuple)) or len(connection) != 2:
                result.add_error(f"Connection {conn_idx}: Must be [from_layer, to_layer]")
                continue
                
            from_layer, to_layer = connection
            
            if not isinstance(from_layer, int) or not isinstance(to_layer, int):
                result.add_error(f"Connection {conn_idx}: Layer indices must be integers")
                continue
                
            if from_layer < 0 or from_layer >= num_layers:
                result.add_error(f"Connection {conn_idx}: from_layer {from_layer} out of range")
                
            if to_layer < 0 or to_layer >= num_layers:
                result.add_error(f"Connection {conn_idx}: to_layer {to_layer} out of range")
                
            if from_layer >= to_layer:
                result.add_error(f"Connection {conn_idx}: from_layer must be < to_layer (no loops)")
                
    def _validate_component_counts(self, circuit_data: Dict[str, Any], result: ValidationResult):
        """Validate component counts are consistent."""
        layers = circuit_data.get('layers', [])
        declared_total = circuit_data.get('total_components', 0)
        
        # Calculate actual component count
        actual_total = 0
        for layer in layers:
            layer_count = layer.get('component_count', len(layer.get('components', [])))
            actual_total += layer_count
            
        if actual_total != declared_total:
            result.add_error(f"Component count mismatch: declared {declared_total}, actual {actual_total}")
            
    def _validate_physical_constraints(self, circuit_data: Dict[str, Any], result: ValidationResult):
        """Validate physical constraints."""
        layers = circuit_data.get('layers', [])
        
        # Estimate total power consumption
        total_power = 0
        total_area = 0
        
        for layer in layers:
            layer_type = layer.get('type', '')
            component_count = layer.get('component_count', 0)
            
            if layer_type == 'linear':
                # MZI power estimation
                total_power += component_count * 0.5  # 0.5 mW per MZI
                total_area += component_count * 0.001  # 0.001 mm² per MZI
            elif layer_type == 'activation':
                # Modulator power estimation
                total_power += component_count * 1.0  # 1.0 mW per modulator
                total_area += component_count * 0.0005  # 0.0005 mm² per modulator
                
        # Power constraints
        if total_power > 1000:  # 1W limit
            result.add_error(f"Total power consumption ({total_power:.1f} mW) exceeds 1W limit")
        elif total_power > 500:  # 500mW warning
            result.add_warning(f"High power consumption: {total_power:.1f} mW")
            
        # Area constraints
        if total_area > 100:  # 100 mm² limit
            result.add_error(f"Total area ({total_area:.1f} mm²) exceeds 100 mm² limit")
        elif total_area > 50:  # 50 mm² warning
            result.add_warning(f"Large circuit area: {total_area:.1f} mm²")
            
    def _sanitize_circuit_data(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create sanitized version of circuit data."""
        sanitized = circuit_data.copy()
        
        # Ensure required fields have defaults
        sanitized.setdefault('pdk', 'generic')
        sanitized.setdefault('wavelength', 1550)
        sanitized.setdefault('connections', [])
        
        # Sanitize layers
        for layer in sanitized.get('layers', []):
            layer.setdefault('components', [])
            layer.setdefault('component_count', len(layer.get('components', [])))
            
        # Recalculate total components
        total_components = sum(layer.get('component_count', 0) for layer in sanitized.get('layers', []))
        sanitized['total_components'] = total_components
        
        return sanitized


class DataSanitizer:
    """Sanitize and normalize input data."""
    
    @staticmethod
    def sanitize_string(value: Any, max_length: int = 255) -> str:
        """Sanitize string input."""
        if value is None:
            return ""
        
        # Convert to string
        str_value = str(value).strip()
        
        # Remove control characters
        str_value = ''.join(char for char in str_value if ord(char) >= 32)
        
        # Truncate if too long
        if len(str_value) > max_length:
            str_value = str_value[:max_length]
            
        return str_value
        
    @staticmethod
    def sanitize_number(value: Any, min_val: float = None, max_val: float = None) -> Union[int, float]:
        """Sanitize numeric input."""
        if value is None:
            return 0
            
        try:
            # Try to convert to number
            if isinstance(value, str):
                # Remove non-numeric characters except . and -
                clean_value = ''.join(c for c in value if c.isdigit() or c in '.-')
                if '.' in clean_value:
                    num_value = float(clean_value)
                else:
                    num_value = int(clean_value)
            else:
                num_value = float(value)
                
            # Apply bounds
            if min_val is not None and num_value < min_val:
                num_value = min_val
            if max_val is not None and num_value > max_val:
                num_value = max_val
                
            return num_value
            
        except (ValueError, TypeError):
            return 0
            
    @staticmethod
    def sanitize_list(value: Any, item_type: type = None) -> List[Any]:
        """Sanitize list input."""
        if value is None:
            return []
            
        if not isinstance(value, (list, tuple)):
            return [value]
            
        result = list(value)
        
        # Type conversion if specified
        if item_type is not None:
            sanitized_items = []
            for item in result:
                try:
                    if item_type == str:
                        sanitized_items.append(DataSanitizer.sanitize_string(item))
                    elif item_type in (int, float):
                        sanitized_items.append(DataSanitizer.sanitize_number(item))
                    else:
                        sanitized_items.append(item_type(item))
                except (ValueError, TypeError):
                    # Skip invalid items
                    continue
            result = sanitized_items
            
        return result


class SecurityValidator:
    """Security-focused validation."""
    
    def __init__(self):
        self.dangerous_patterns = [
            'script', 'javascript', 'eval', 'exec', 'import',
            '__', 'os.', 'sys.', 'subprocess', 'open(',
        ]
        
    def validate_safe_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input for security issues."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check for dangerous patterns in string data
        self._check_dangerous_patterns(data, result)
        
        # Check for excessively large data
        self._check_data_size(data, result)
        
        # Check for deeply nested structures
        self._check_nesting_depth(data, result)
        
        return result
        
    def _check_dangerous_patterns(self, data: Any, result: ValidationResult, path: str = ""):
        """Recursively check for dangerous patterns."""
        if isinstance(data, str):
            for pattern in self.dangerous_patterns:
                if pattern.lower() in data.lower():
                    result.add_error(f"Dangerous pattern '{pattern}' found in {path}")
                    
        elif isinstance(data, dict):
            for key, value in data.items():
                self._check_dangerous_patterns(value, result, f"{path}.{key}")
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_dangerous_patterns(item, result, f"{path}[{i}]")
                
    def _check_data_size(self, data: Dict[str, Any], result: ValidationResult):
        """Check for excessively large data structures."""
        try:
            # Rough size estimation
            str_repr = str(data)
            size_mb = len(str_repr) / (1024 * 1024)
            
            if size_mb > 100:  # 100MB limit
                result.add_error(f"Data too large: {size_mb:.1f}MB exceeds 100MB limit")
            elif size_mb > 10:  # 10MB warning
                result.add_warning(f"Large data size: {size_mb:.1f}MB")
                
        except Exception as e:
            result.add_warning(f"Could not estimate data size: {e}")
            
    def _check_nesting_depth(self, data: Any, result: ValidationResult, depth: int = 0):
        """Check for excessive nesting depth."""
        max_depth = 20
        
        if depth > max_depth:
            result.add_error(f"Nesting depth exceeds {max_depth} levels")
            return
            
        if isinstance(data, dict):
            for value in data.values():
                self._check_nesting_depth(value, result, depth + 1)
        elif isinstance(data, list):
            for item in data:
                self._check_nesting_depth(item, result, depth + 1)