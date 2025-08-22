#!/usr/bin/env python3
"""
Simple functionality test for photonic-foundry core components
without requiring heavy dependencies like PyTorch.
"""

import sys
import os
import json
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add src to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the heavy dependencies
class MockTorch:
    class nn:
        class Module:
            pass
        def Linear(input_features, output_features):
            return MockModule(input_features, output_features)
        def ReLU():
            return MockModule(0, 0)
        def Sequential(*layers):
            return MockSequential(layers)
    
    def tensor(data):
        return MockTensor(data)

class MockModule:
    def __init__(self, in_features=0, out_features=0):
        self.in_features = in_features
        self.out_features = out_features

class MockSequential:
    def __init__(self, layers):
        self.layers = layers

class MockTensor:
    def __init__(self, data):
        self.data = data

class MockNumpy:
    def random(self):
        class Random:
            def randn(*shape):
                return [[0.1 * i * j for j in range(shape[1])] for i in range(shape[0])]
        return Random()
    random = random(None)

# Mock the imports
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MockTorch.nn()  
sys.modules['numpy'] = MockNumpy()
sys.modules['scipy'] = type('MockScipy', (), {})()
sys.modules['matplotlib'] = type('MockMatplotlib', (), {})()
sys.modules['pandas'] = type('MockPandas', (), {})()
sys.modules['pydantic'] = type('MockPydantic', (), {})()

# Now we can test core functionality
def test_core_imports():
    """Test that core photonic foundry components can be imported."""
    try:
        from photonic_foundry.core import PhotonicComponent, CircuitMetrics, MZILayer
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Core import failed: {e}")
        return False

def test_photonic_components():
    """Test photonic component enum functionality."""
    try:
        from photonic_foundry.core import PhotonicComponent
        components = [PhotonicComponent.MZI, PhotonicComponent.RING, PhotonicComponent.WAVEGUIDE]
        print(f"✓ Photonic components defined: {[c.value for c in components]}")
        return True
    except Exception as e:
        print(f"✗ Photonic components test failed: {e}")
        return False

def test_circuit_metrics():
    """Test circuit metrics dataclass functionality."""
    try:
        from photonic_foundry.core import CircuitMetrics
        metrics = CircuitMetrics(
            energy_per_op=1.5,
            latency=200.0,
            area=0.1,
            power=10.0,
            throughput=1000.0,
            accuracy=0.99
        )
        metrics_dict = metrics.to_dict()
        print(f"✓ Circuit metrics created and serialized: {metrics_dict['energy_per_op']} pJ/op")
        return True
    except Exception as e:
        print(f"✗ Circuit metrics test failed: {e}")
        return False

def test_mzi_layer_creation():
    """Test MZI layer creation and basic functionality."""
    try:
        from photonic_foundry.core import MZILayer
        layer = MZILayer(input_size=4, output_size=2, precision=8)
        verilog = layer.generate_verilog()
        print(f"✓ MZI layer created: {layer.input_size}→{layer.output_size}, components: {len(layer.components)}")
        print(f"✓ Verilog generation works: {len(verilog)} characters")
        return True
    except Exception as e:
        print(f"✗ MZI layer test failed: {e}")
        return False

def test_quantum_planner_imports():
    """Test quantum planner components can be imported."""
    try:
        from photonic_foundry.quantum_planner import ResourceConstraint
        constraint = ResourceConstraint(max_energy=100.0, max_latency=500.0, thermal_limit=75.0)
        print("✓ Quantum planner imports and ResourceConstraint creation successful")
        return True
    except Exception as e:
        print(f"✗ Quantum planner import failed: {e}")
        return False

def run_all_tests():
    """Run all basic functionality tests."""
    print("🚀 GENERATION 1 VALIDATION: Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Photonic Components", test_photonic_components), 
        ("Circuit Metrics", test_circuit_metrics),
        ("MZI Layer Creation", test_mzi_layer_creation),
        ("Quantum Planner", test_quantum_planner_imports)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ GENERATION 1 VALIDATION: PASSED - Core functionality working!")
        return True
    else:
        print("❌ GENERATION 1 VALIDATION: Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)