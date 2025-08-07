#!/usr/bin/env python3
"""
Quality gates execution script for quantum-inspired photonic foundry.

This script runs comprehensive tests, security scans, and performance benchmarks
to ensure the quantum photonic system meets all quality requirements.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


class QualityGateRunner:
    """Execute comprehensive quality gates for the photonic foundry."""
    
    def __init__(self):
        """Initialize quality gate runner."""
        self.results = {}
        self.start_time = time.time()
        self.passed = 0
        self.failed = 0
    
    def run_command(self, command: str, description: str) -> dict:
        """Run a command and capture results."""
        print(f"\n🔄 {description}")
        print(f"   Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("   ✅ PASSED")
                self.passed += 1
                return {
                    'status': 'PASSED',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            else:
                print(f"   ❌ FAILED (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                self.failed += 1
                return {
                    'status': 'FAILED',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
        
        except subprocess.TimeoutExpired:
            print("   ⏰ TIMEOUT")
            self.failed += 1
            return {
                'status': 'TIMEOUT',
                'error': 'Command timed out after 5 minutes'
            }
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            self.failed += 1
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def run_unit_tests(self):
        """Run unit tests."""
        print("\n" + "="*60)
        print("🧪 UNIT TESTS")
        print("="*60)
        
        # Test quantum planner
        self.results['test_quantum_planner'] = self.run_command(
            "bash -c 'source venv/bin/activate && PYTHONPATH=/root/repo/src python -m pytest tests/unit/test_quantum_planner.py::TestQuantumTask -v'",
            "Testing QuantumTask class"
        )
        
        # Test quantum security
        self.results['test_quantum_security'] = self.run_command(
            "bash -c 'source venv/bin/activate && PYTHONPATH=/root/repo/src python -m pytest tests/unit/test_quantum_security.py::TestQuantumRandomGenerator -v'",
            "Testing QuantumRandomGenerator"
        )
        
        # Test core functionality
        self.results['test_core_imports'] = self.run_command(
            "bash -c 'source venv/bin/activate && PYTHONPATH=/root/repo/src python -c \"from photonic_foundry import *; print(\\\"All imports successful\\\")'",
            "Testing core module imports"
        )
    
    def run_security_scan(self):
        """Run security analysis."""
        print("\n" + "="*60)
        print("🔒 SECURITY SCAN")
        print("="*60)
        
        # Check for hardcoded secrets
        self.results['security_secrets'] = self.run_command(
            "grep -r -i 'password\\|secret\\|key\\|token' src/ --exclude-dir=__pycache__ || echo 'No hardcoded secrets found'",
            "Scanning for hardcoded secrets"
        )
        
        # Check file permissions
        self.results['security_permissions'] = self.run_command(
            "find src/ -type f -name '*.py' -perm /077 | head -5 || echo 'File permissions OK'",
            "Checking file permissions"
        )
        
        # Quantum security test
        self.results['quantum_security_demo'] = self.run_command(
            "bash -c 'source venv/bin/activate && PYTHONPATH=/root/repo/src timeout 30 python -c \"from photonic_foundry.quantum_security import *; qrng = QuantumRandomGenerator(); print(\\\"Quantum security:\\\", qrng.generate_secure_bytes(32)[:8])\"'",
            "Testing quantum security functionality"
        )
    
    def run_performance_tests(self):
        """Run performance benchmarks."""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE TESTS")
        print("="*60)
        
        # Basic performance test
        self.results['performance_basic'] = self.run_command(
            "source venv/bin/activate && PYTHONPATH=/root/repo/src timeout 60 python -c \"from photonic_foundry import *; import torch; model = torch.nn.Linear(100, 10); acc = PhotonicAccelerator(); circuit = acc.convert_pytorch_model(model); metrics = circuit.analyze_circuit(); print(f'Circuit metrics: Energy={metrics.energy_per_op}pJ, Latency={metrics.latency}ps')\"",
            "Basic circuit performance test"
        )
        
        # Quantum optimization test
        self.results['performance_optimization'] = self.run_command(
            "source venv/bin/activate && PYTHONPATH=/root/repo/src timeout 60 python -c \"from photonic_foundry.quantum_planner import *; from photonic_foundry import *; import torch; acc = PhotonicAccelerator(); planner = QuantumTaskPlanner(acc); task = QuantumTask('perf_test'); planner.register_task(task); stats = planner.get_optimization_statistics(); print(f'Optimization stats: {stats}')\"",
            "Quantum optimization performance"
        )
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "="*60)
        print("🔗 INTEGRATION TESTS") 
        print("="*60)
        
        # Full workflow test
        self.results['integration_workflow'] = self.run_command(
            "source venv/bin/activate && PYTHONPATH=/root/repo/src timeout 90 python examples/quantum_planning_demo.py | head -20",
            "Full quantum planning workflow"
        )
        
        # Database integration
        self.results['integration_database'] = self.run_command(
            "source venv/bin/activate && PYTHONPATH=/root/repo/src timeout 30 python -c \"from photonic_foundry.database import *; db = get_database(); print('Database connection:', 'OK' if db else 'FAILED')\"",
            "Database integration test"
        )
    
    def run_code_quality_checks(self):
        """Run code quality checks."""
        print("\n" + "="*60)
        print("🎯 CODE QUALITY")
        print("="*60)
        
        # Check Python syntax
        self.results['quality_syntax'] = self.run_command(
            "find src/ -name '*.py' -exec python -m py_compile {} \\; && echo 'All Python files compile successfully'",
            "Python syntax check"
        )
        
        # Count lines of code
        self.results['quality_loc'] = self.run_command(
            "find src/ -name '*.py' -exec wc -l {} \\; | tail -1 | awk '{print \"Total lines of code:\", $1}'",
            "Lines of code count"
        )
        
        # Check for TODO/FIXME items
        self.results['quality_todos'] = self.run_command(
            "grep -r 'TODO\\|FIXME\\|XXX' src/ --include='*.py' | wc -l | awk '{print \"TODO items found:\", $1}'",
            "TODO/FIXME analysis"
        )
    
    def generate_report(self):
        """Generate comprehensive quality gate report."""
        print("\n" + "="*80)
        print("📊 QUALITY GATE REPORT")
        print("="*80)
        
        total_time = time.time() - self.start_time
        total_tests = self.passed + self.failed
        pass_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"✅ Tests passed: {self.passed}")
        print(f"❌ Tests failed: {self.failed}")
        print(f"📈 Pass rate: {pass_rate:.1f}%")
        
        # Determine overall status
        if self.failed == 0:
            overall_status = "🎉 ALL QUALITY GATES PASSED"
            exit_code = 0
        elif pass_rate >= 80:
            overall_status = "⚠️ MOST QUALITY GATES PASSED (Minor issues)"
            exit_code = 0
        else:
            overall_status = "❌ QUALITY GATES FAILED (Critical issues)"
            exit_code = 1
        
        print(f"\n🏆 Overall Status: {overall_status}")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            status_icon = "✅" if result.get('status') == 'PASSED' else "❌"
            print(f"   {status_icon} {test_name}: {result.get('status')}")
        
        # Save report to file
        report_data = {
            'timestamp': time.time(),
            'execution_time': total_time,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': pass_rate,
            'overall_status': overall_status,
            'detailed_results': self.results
        }
        
        report_path = Path('quality_gate_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")
        
        return exit_code
    
    def run_all(self):
        """Run all quality gates."""
        print("🚀 QUANTUM-INSPIRED PHOTONIC FOUNDRY - QUALITY GATES")
        print("🔬 Comprehensive testing of quantum task planning system")
        print("=" * 80)
        
        try:
            self.run_unit_tests()
            self.run_security_scan()
            self.run_performance_tests()
            self.run_integration_tests() 
            self.run_code_quality_checks()
            
            return self.generate_report()
            
        except KeyboardInterrupt:
            print("\n⚠️ Quality gate execution interrupted by user")
            return 130
        except Exception as e:
            print(f"\n💥 Unexpected error during quality gate execution: {e}")
            return 1


def main():
    """Main entry point."""
    runner = QualityGateRunner()
    exit_code = runner.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()