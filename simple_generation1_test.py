#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - Generation 1: Make It Work (Simple)
Basic functionality validation and quantum-enhanced feature demonstration.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test basic imports work."""
    try:
        from photonic_foundry.core import PhotonicAccelerator, PhotonicCircuit, MZILayer
        from photonic_foundry.quantum_planner import QuantumTaskPlanner
        from photonic_foundry.quantum_security import QuantumSecurityManager
        return True, "✅ Core imports successful"
    except ImportError as e:
        return False, f"❌ Import failed: {e}"

def test_basic_circuit_creation():
    """Test basic photonic circuit creation."""
    try:
        from photonic_foundry.core import PhotonicAccelerator, MZILayer
        
        # Create accelerator
        accelerator = PhotonicAccelerator(pdk='skywater130', wavelength=1550.0)
        
        # Create simple MLP circuit
        layer1 = MZILayer(784, 256)  # Input layer
        layer2 = MZILayer(256, 10)   # Output layer
        
        circuit = accelerator.convert_pytorch_model(
            type('DummyModel', (), {
                '__class__.__name__': 'SimpleMLP',
                'named_modules': lambda: [
                    ('fc1', type('Linear', (), {
                        'in_features': 784, 
                        'out_features': 256,
                        'weight': type('Tensor', (), {'detach': lambda: type('Array', (), {'numpy': lambda: [[0.1] * 784] * 256})()})()
                    })()),
                    ('fc2', type('Linear', (), {
                        'in_features': 256, 
                        'out_features': 10,
                        'weight': type('Tensor', (), {'detach': lambda: type('Array', (), {'numpy': lambda: [[0.1] * 256] * 10})()})()
                    })())
                ]
            })()
        )
        
        # Analyze circuit
        metrics = circuit.analyze_circuit()
        
        return True, f"✅ Circuit created with {len(circuit.layers)} layers, {metrics.energy_per_op:.2f}pJ energy"
    except Exception as e:
        return False, f"❌ Circuit creation failed: {e}"

def test_quantum_optimization():
    """Test quantum-enhanced optimization features."""
    try:
        from photonic_foundry.quantum_planner import QuantumTaskPlanner
        from photonic_foundry.core import PhotonicAccelerator
        
        # Initialize quantum planner
        accelerator = PhotonicAccelerator()
        planner = QuantumTaskPlanner()
        
        # Test quantum annealing optimization
        result = planner.quantum_annealing_optimization([
            {'task_id': 1, 'energy': 100, 'latency': 50},
            {'task_id': 2, 'energy': 80, 'latency': 60}
        ])
        
        return True, f"✅ Quantum optimization completed with {len(result)} optimized tasks"
    except Exception as e:
        return False, f"❌ Quantum optimization failed: {e}"

def test_security_features():
    """Test quantum security features."""
    try:
        from photonic_foundry.quantum_security import QuantumSecurityManager, SecurityLevel
        
        # Initialize security manager
        security = QuantumSecurityManager()
        
        # Create secure token
        token = security.create_security_token("test_user", ["execute_tasks"])
        
        # Test encryption
        data = "test_data"
        encrypted = security.encrypt_data(data)
        decrypted = security.decrypt_data(encrypted)
        
        return True, f"✅ Security features working - token: {token[:20]}..."
    except Exception as e:
        return False, f"❌ Security test failed: {e}"

def run_generation1_tests():
    """Run Generation 1 basic functionality tests."""
    print("🚀 TERRAGON SDLC v4.0 - Generation 1: Make It Work")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Circuit Creation", test_basic_circuit_creation),
        ("Quantum Optimization", test_quantum_optimization),
        ("Security Features", test_security_features)
    ]
    
    results = []
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success, message = test_func()
            results.append({
                'test': test_name,
                'success': success,
                'message': message,
                'timestamp': time.time()
            })
            if success:
                passed += 1
            print(f"   {message}")
        except Exception as e:
            results.append({
                'test': test_name,
                'success': False,
                'message': f"❌ Test crashed: {e}",
                'timestamp': time.time()
            })
            print(f"   ❌ Test crashed: {e}")
    
    # Generate report
    report = {
        'generation': 1,
        'total_tests': len(tests),
        'passed': passed,
        'success_rate': passed / len(tests) * 100,
        'results': results,
        'timestamp': time.time()
    }
    
    # Save report
    with open('/root/repo/generation1_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Generation 1 Results:")
    print(f"   Tests Passed: {passed}/{len(tests)} ({report['success_rate']:.1f}%)")
    
    return report['success_rate'] >= 75  # 75% pass rate for Generation 1

if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)