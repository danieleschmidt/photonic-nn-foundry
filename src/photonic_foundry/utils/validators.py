"""
Input validation and model compatibility checking.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, validator, Field
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PhotonicConfig(BaseModel):
    """Configuration model for photonic accelerator."""
    
    pdk: str = Field(..., pattern=r'^(skywater130|tsmc65|generic)$')
    wavelength: float = Field(default=1550.0, ge=1200.0, le=1700.0)
    precision: int = Field(default=8, ge=1, le=32)
    max_model_size_mb: int = Field(default=100, ge=1, le=1000)
    max_layers: int = Field(default=1000, ge=1, le=10000)
    allowed_layer_types: List[str] = Field(default=['Linear', 'Conv2d', 'ReLU', 'BatchNorm2d'])
    enable_optimization: bool = True
    
    @validator('wavelength')
    def validate_wavelength(cls, v):
        """Ensure wavelength is in optical communication bands."""
        if not (1200 <= v <= 1700):
            raise ValueError('Wavelength must be between 1200-1700nm (optical communication bands)')
        return v
    
    @validator('precision')  
    def validate_precision(cls, v):
        """Ensure precision is reasonable for photonic implementation."""
        if v > 16:
            logger.warning(f"High precision ({v} bits) may be challenging for photonic implementation")
        return v


class ModelValidator:
    """Validates PyTorch models for photonic compatibility."""
    
    def __init__(self, config: PhotonicConfig):
        self.config = config
        self.security_checks = [
            self._check_model_size,
            self._check_layer_count,
            self._check_layer_types,
            self._check_tensor_shapes,
            self._check_parameter_ranges,
        ]
        
    def validate_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model: PyTorch model to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {},
            'security_checks': {},
        }
        
        try:
            # Basic model information
            report['model_info'] = self._extract_model_info(model)
            
            # Run security checks
            for check_func in self.security_checks:
                check_name = check_func.__name__.replace('_check_', '')
                try:
                    check_result = check_func(model)
                    report['security_checks'][check_name] = check_result
                    
                    if not check_result['passed']:
                        report['errors'].extend(check_result.get('errors', []))
                        report['is_valid'] = False
                        
                    report['warnings'].extend(check_result.get('warnings', []))
                    
                except Exception as e:
                    error_msg = f"Security check {check_name} failed: {str(e)}"
                    report['errors'].append(error_msg)
                    report['is_valid'] = False
                    logger.error(error_msg)
                    
            # Additional compatibility checks
            compat_report = self._check_photonic_compatibility(model)
            report.update(compat_report)
            
        except Exception as e:
            report['is_valid'] = False
            report['errors'].append(f"Model validation failed: {str(e)}")
            logger.error(f"Model validation error: {e}")
            
        return report
        
    def _extract_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract basic model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'num_layers': len(list(model.modules())),
            'device': str(next(model.parameters()).device) if total_params > 0 else 'unknown'
        }
        
    def _check_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Check if model size is within limits."""
        model_info = self._extract_model_info(model)
        size_mb = model_info['model_size_mb']
        
        result = {
            'passed': size_mb <= self.config.max_model_size_mb,
            'size_mb': size_mb,
            'limit_mb': self.config.max_model_size_mb,
            'errors': [],
            'warnings': []
        }
        
        if not result['passed']:
            result['errors'].append(f"Model size ({size_mb:.1f}MB) exceeds limit ({self.config.max_model_size_mb}MB)")
        elif size_mb > self.config.max_model_size_mb * 0.8:
            result['warnings'].append(f"Model size ({size_mb:.1f}MB) is approaching limit ({self.config.max_model_size_mb}MB)")
            
        return result
        
    def _check_layer_count(self, model: nn.Module) -> Dict[str, Any]:
        """Check if layer count is within limits.""" 
        num_layers = len(list(model.modules()))
        
        result = {
            'passed': num_layers <= self.config.max_layers,
            'layer_count': num_layers,
            'limit': self.config.max_layers,
            'errors': [],
            'warnings': []
        }
        
        if not result['passed']:
            result['errors'].append(f"Layer count ({num_layers}) exceeds limit ({self.config.max_layers})")
        elif num_layers > self.config.max_layers * 0.8:
            result['warnings'].append(f"Layer count ({num_layers}) is approaching limit ({self.config.max_layers})")
            
        return result
        
    def _check_layer_types(self, model: nn.Module) -> Dict[str, Any]:
        """Check if all layers are supported."""
        unsupported_layers = []
        layer_counts = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_type = type(module).__name__
                layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
                
                if layer_type not in self.config.allowed_layer_types:
                    unsupported_layers.append((name, layer_type))
                    
        result = {
            'passed': len(unsupported_layers) == 0,
            'unsupported_layers': unsupported_layers,
            'layer_counts': layer_counts,
            'errors': [],
            'warnings': []
        }
        
        if not result['passed']:
            result['errors'].append(f"Unsupported layer types found: {[lt for _, lt in unsupported_layers]}")
            
        return result
        
    def _check_tensor_shapes(self, model: nn.Module) -> Dict[str, Any]:
        """Check tensor shapes for validity."""
        shape_issues = []
        
        for name, param in model.named_parameters():
            if param.dim() > 4:
                shape_issues.append(f"Parameter {name} has {param.dim()} dimensions (max 4 supported)")
            if param.numel() > 10**8:  # 100M parameters in single tensor
                shape_issues.append(f"Parameter {name} is very large ({param.numel():,} elements)")
                
        result = {
            'passed': len(shape_issues) == 0,
            'shape_issues': shape_issues,
            'errors': [],
            'warnings': []
        }
        
        if not result['passed']:
            result['errors'].extend(shape_issues)
            
        return result
        
    def _check_parameter_ranges(self, model: nn.Module) -> Dict[str, Any]:
        """Check parameter value ranges."""
        range_issues = []
        
        for name, param in model.named_parameters():
            param_data = param.detach()
            
            # Check for NaN or Inf values
            if torch.isnan(param_data).any():
                range_issues.append(f"Parameter {name} contains NaN values")
            if torch.isinf(param_data).any():
                range_issues.append(f"Parameter {name} contains infinite values")
                
            # Check for extreme values that might cause issues
            param_abs = torch.abs(param_data)
            if param_abs.max() > 100:
                range_issues.append(f"Parameter {name} has very large values (max: {param_abs.max():.2f})")
                
        result = {
            'passed': len(range_issues) == 0,
            'range_issues': range_issues,
            'errors': [],
            'warnings': []
        }
        
        if not result['passed']:
            result['errors'].extend(range_issues)
            
        return result
        
    def _check_photonic_compatibility(self, model: nn.Module) -> Dict[str, Any]:
        """Check specific photonic implementation compatibility."""
        compat_issues = []
        compat_warnings = []
        
        # Check for complex number operations (not supported)
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if torch.is_complex(module.weight):
                    compat_issues.append(f"Complex weights in {name} not supported")
                    
        # Check for dynamic shapes (problematic for hardware)
        try:
            sample_input = torch.randn(1, 3, 224, 224)  # Common input size
            with torch.no_grad():
                _ = model(sample_input)
        except Exception as e:
            compat_warnings.append(f"Model may have dynamic behavior: {str(e)}")
            
        # Check for recurrent connections (not easily mapped to photonics)
        has_rnn = any(isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)) for m in model.modules())
        if has_rnn:
            compat_warnings.append("Recurrent layers detected - may require special handling")
            
        return {
            'photonic_compatibility': {
                'issues': compat_issues,
                'warnings': compat_warnings,
                'has_complex_weights': any('Complex weights' in issue for issue in compat_issues),
                'has_recurrent_layers': has_rnn,
            }
        }


class ConfigValidator:
    """Validates configuration parameters."""
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any]) -> Tuple[PhotonicConfig, List[str]]:
        """
        Validate configuration dictionary.
        
        Args:
            config_dict: Configuration parameters
            
        Returns:
            Tuple of (validated_config, error_messages)
        """
        errors = []
        
        try:
            config = PhotonicConfig(**config_dict)
            return config, errors
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
            # Return default config on failure
            return PhotonicConfig(), errors
            
    @staticmethod 
    def sanitize_file_path(file_path: str) -> str:
        """
        Sanitize file paths to prevent directory traversal.
        
        Args:
            file_path: Input file path
            
        Returns:
            Sanitized file path
        """
        import os
        from pathlib import Path
        
        # Convert to Path object and resolve
        path = Path(file_path).resolve()
        
        # Ensure path doesn't go outside current directory tree
        current_dir = Path.cwd().resolve()
        try:
            path.relative_to(current_dir)
        except ValueError:
            raise ValueError(f"Path {file_path} is outside allowed directory")
            
        return str(path)