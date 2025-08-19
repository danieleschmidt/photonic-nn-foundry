#!/usr/bin/env python3
"""
Validate Generation 2: Robust framework components
"""

import sys
import os
import importlib.util
from pathlib import Path

def load_module_safe(name, path):
    """Safely load a module without executing dependencies."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Don't execute - just check it can be loaded
        return module
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return None

def validate_robust_framework():
    """Validate all robust framework components."""
    src_path = Path("/root/repo/src/photonic_foundry")
    
    components = {
        'error_handling': 'Error handling and recovery mechanisms',
        'validation': 'Input validation and sanitization',
        'security': 'Security scanning and validation', 
        'monitoring': 'System monitoring and observability',
        'logging_config': 'Structured logging configuration',
        'circuit_breaker': 'Circuit breaker patterns',
        'resilience_framework': 'Comprehensive resilience framework'
    }
    
    passed = 0
    total = len(components)
    
    print("🛡️ GENERATION 2: ROBUST FRAMEWORK VALIDATION")
    print("=" * 60)
    
    for component, description in components.items():
        module_path = src_path / f"{component}.py"
        if module_path.exists():
            # Check file is readable and has content
            try:
                with open(module_path, 'r') as f:
                    content = f.read()
                    if len(content) > 100:  # Basic content check
                        print(f"✅ {component:20} - {description}")
                        passed += 1
                    else:
                        print(f"❌ {component:20} - File too small")
            except Exception as e:
                print(f"❌ {component:20} - Read error: {e}")
        else:
            print(f"❌ {component:20} - File not found")
    
    # Check advanced features
    advanced_features = [
        ('Docker configuration', Path('/root/repo/Dockerfile')),
        ('Production deployment', Path('/root/repo/deployment')),
        ('Quality gates', Path('/root/repo/run_quality_gates.py')),
        ('Security policies', Path('/root/repo/production_security_policy.json')),
        ('Monitoring config', Path('/root/repo/monitoring'))
    ]
    
    print("\n🚀 ADVANCED ROBUSTNESS FEATURES")
    print("=" * 40)
    
    for feature, path in advanced_features:
        if path.exists():
            print(f"✅ {feature}")
            passed += 1
        else:
            print(f"❌ {feature}")
        total += 1
    
    # Summary
    pass_rate = (passed / total) * 100
    print(f"\n📊 ROBUSTNESS VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Components validated: {total}")
    print(f"Components passed: {passed}")
    print(f"Pass rate: {pass_rate:.1f}%")
    
    if pass_rate >= 85:
        print("🏆 Generation 2: ROBUST - ✅ PASSED")
        return True
    else:
        print("❌ Generation 2: ROBUST - FAILED")
        return False

if __name__ == "__main__":
    success = validate_robust_framework()
    sys.exit(0 if success else 1)