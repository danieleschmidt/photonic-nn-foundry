#!/usr/bin/env python3
"""
Autonomous Quality Validation System
Comprehensive testing, security scanning, and quality gates for TERRAGON SDLC.
"""

import sys
import os
import json
import time
import subprocess
import traceback
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    passed: bool
    execution_time: float
    output: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeQualityAnalyzer:
    """Analyze code quality without external dependencies."""
    
    def __init__(self):
        self.src_path = Path(__file__).parent / "src"
        self.test_metrics = {}
    
    def analyze_python_files(self) -> Dict[str, Any]:
        """Analyze Python code quality."""
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'docstring_coverage': 0.0,
            'complexity_issues': [],
            'security_issues': []
        }
        
        python_files = list(self.src_path.rglob("*.py"))
        metrics['total_files'] = len(python_files)
        
        functions_with_docs = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    metrics['total_lines'] += len(lines)
                    
                    # Count functions and classes
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('def '):
                            total_functions += 1
                            # Check for docstring in next few lines
                            line_idx = lines.index(line)
                            if (line_idx + 1 < len(lines) and 
                                ('"""' in lines[line_idx + 1] or "'''" in lines[line_idx + 1])):
                                functions_with_docs += 1
                        elif stripped.startswith('class '):
                            metrics['total_classes'] += 1
                    
                    # Security checks
                    security_patterns = ['eval(', 'exec(', 'os.system', 'subprocess.call']
                    for pattern in security_patterns:
                        if pattern in content:
                            metrics['security_issues'].append({
                                'file': str(py_file.relative_to(self.src_path)),
                                'issue': f"Potentially dangerous pattern: {pattern}",
                                'severity': 'high'
                            })
                    
                    # Complexity checks (simplified)
                    if len(lines) > 1000:
                        metrics['complexity_issues'].append({
                            'file': str(py_file.relative_to(self.src_path)),
                            'issue': f"Large file: {len(lines)} lines",
                            'severity': 'medium'
                        })
                        
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {str(e)}")
        
        metrics['total_functions'] = total_functions
        metrics['docstring_coverage'] = (
            functions_with_docs / total_functions if total_functions > 0 else 0.0
        )
        
        return metrics
    
    def run_basic_tests(self) -> List[TestResult]:
        """Run basic functionality tests."""
        test_results = []
        
        # Test 1: Core imports
        test_results.append(self._test_core_imports())
        
        # Test 2: Core functionality
        test_results.append(self._test_core_functionality())
        
        # Test 3: Robust framework
        test_results.append(self._test_robust_framework())
        
        # Test 4: Scaling engine
        test_results.append(self._test_scaling_engine())
        
        return test_results
    
    def _test_core_imports(self) -> TestResult:
        """Test that core modules can be imported."""
        start_time = time.time()
        try:
            sys.path.append(str(self.src_path))
            
            # Test standalone core
            spec = importlib.util.spec_from_file_location(
                "core_standalone",
                self.src_path / "photonic_foundry" / "core_standalone.py"
            )
            core_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core_module)
            
            # Verify key components exist
            assert hasattr(core_module, 'PhotonicAccelerator')
            assert hasattr(core_module, 'MZILayer')
            assert hasattr(core_module, 'CircuitMetrics')
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="core_imports",
                passed=True,
                execution_time=execution_time,
                output="All core imports successful"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="core_imports",
                passed=False,
                execution_time=execution_time,
                output="",
                error=str(e)
            )
    
    def _test_core_functionality(self) -> TestResult:
        """Test core photonic functionality."""
        start_time = time.time()
        try:
            # Import and test core standalone
            sys.path.append(str(self.src_path))
            spec = importlib.util.spec_from_file_location(
                "core_standalone",
                self.src_path / "photonic_foundry" / "core_standalone.py"
            )
            core_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core_module)
            
            # Test accelerator creation
            accelerator = core_module.PhotonicAccelerator()
            circuit = accelerator.convert_simple_model([4, 8, 2])
            
            # Verify circuit properties
            assert len(circuit.layers) == 2
            assert circuit.total_components > 0
            
            # Test metrics estimation
            metrics = circuit.estimate_metrics()
            assert metrics.energy_per_op > 0
            assert metrics.latency > 0
            
            # Test Verilog generation
            verilog = circuit.generate_full_verilog()
            assert len(verilog) > 100  # Should be substantial
            assert "module" in verilog
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="core_functionality",
                passed=True,
                execution_time=execution_time,
                output=f"Created circuit with {circuit.total_components} components, "
                       f"estimated energy: {metrics.energy_per_op:.2f} pJ/op"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="core_functionality",
                passed=False,
                execution_time=execution_time,
                output="",
                error=str(e)
            )
    
    def _test_robust_framework(self) -> TestResult:
        """Test robust framework components."""
        start_time = time.time()
        try:
            sys.path.append(str(self.src_path))
            spec = importlib.util.spec_from_file_location(
                "robust_framework",
                self.src_path / "photonic_foundry" / "robust_framework.py"
            )
            robust_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(robust_module)
            
            # Test input validation
            validator = robust_module.InputValidator()
            valid_result = validator.validate_layer_sizes([4, 8, 2])
            assert valid_result.passed
            
            invalid_result = validator.validate_layer_sizes([0, -1])
            assert not invalid_result.passed
            
            # Test security manager
            security_manager = robust_module.SecurityManager()
            token = security_manager.generate_secure_token()
            assert len(token) >= 16
            
            # Test performance monitor
            monitor = robust_module.PerformanceMonitor()
            with monitor.monitor_operation("test_op"):
                time.sleep(0.01)  # Simulate work
            
            assert "test_op" in monitor.metrics
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="robust_framework",
                passed=True,
                execution_time=execution_time,
                output="Robust framework validation, security, and monitoring working"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="robust_framework",
                passed=False,
                execution_time=execution_time,
                output="",
                error=str(e)
            )
    
    def _test_scaling_engine(self) -> TestResult:
        """Test scaling engine components."""
        start_time = time.time()
        try:
            sys.path.append(str(self.src_path))
            spec = importlib.util.spec_from_file_location(
                "scaling_engine",
                self.src_path / "photonic_foundry" / "scaling_engine.py"
            )
            scaling_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scaling_module)
            
            # Test intelligent cache
            cache = scaling_module.IntelligentCache(max_size=100)
            cache.put("test_key", {"data": "test_value"})
            result = cache.get("test_key")
            assert result is not None
            assert result["data"] == "test_value"
            
            # Test load balancer
            lb = scaling_module.LoadBalancer()
            node = scaling_module.WorkerNode("test_node", capacity=10)
            lb.add_node(node)
            selected = lb.select_node()
            assert selected.node_id == "test_node"
            
            # Test concurrent executor
            executor = scaling_module.ConcurrentTaskExecutor(max_workers=2)
            
            def test_func(x):
                return x * 2
            
            task_id = executor.submit_task(test_func, 5)
            result = executor.get_result(task_id)
            assert result == 10
            
            executor.shutdown()
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="scaling_engine",
                passed=True,
                execution_time=execution_time,
                output="Scaling engine cache, load balancing, and concurrency working"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="scaling_engine",
                passed=False,
                execution_time=execution_time,
                output="",
                error=str(e)
            )


class SecurityScanner:
    """Basic security scanning without external dependencies."""
    
    def scan_codebase(self, base_path: Path) -> Dict[str, Any]:
        """Scan codebase for security issues."""
        security_results = {
            'high_risk_patterns': [],
            'medium_risk_patterns': [],
            'low_risk_patterns': [],
            'files_scanned': 0,
            'clean_files': 0
        }
        
        # Define security patterns
        high_risk = [
            'eval(', 'exec(', '__import__', 'compile(',
            'os.system', 'subprocess.call', 'subprocess.run',
            'pickle.loads', 'yaml.load', 'marshal.loads'
        ]
        
        medium_risk = [
            'input(', 'raw_input(', 'open(', 'file(',
            'urllib.request', 'requests.get', 'socket.',
            'random.', 'tempfile.'
        ]
        
        low_risk = [
            'print(', 'logging.', 'assert ', 'raise ',
            'try:', 'except:', 'finally:'
        ]
        
        python_files = list(base_path.rglob("*.py"))
        security_results['files_scanned'] = len(python_files)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_has_issues = False
                    
                    # Check for high risk patterns
                    for pattern in high_risk:
                        if pattern in content:
                            security_results['high_risk_patterns'].append({
                                'file': str(py_file.relative_to(base_path)),
                                'pattern': pattern,
                                'risk_level': 'high'
                            })
                            file_has_issues = True
                    
                    # Check for medium risk patterns
                    for pattern in medium_risk:
                        if pattern in content:
                            security_results['medium_risk_patterns'].append({
                                'file': str(py_file.relative_to(base_path)),
                                'pattern': pattern,
                                'risk_level': 'medium'
                            })
                            file_has_issues = True
                    
                    # Check for low risk patterns (informational)
                    for pattern in low_risk:
                        if pattern in content:
                            security_results['low_risk_patterns'].append({
                                'file': str(py_file.relative_to(base_path)),
                                'pattern': pattern,
                                'risk_level': 'low'
                            })
                    
                    if not file_has_issues:
                        security_results['clean_files'] += 1
                        
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {str(e)}")
        
        return security_results


class AutonomousQualityGates:
    """Comprehensive autonomous quality gate system."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.code_analyzer = CodeQualityAnalyzer()
        self.security_scanner = SecurityScanner()
        self.quality_threshold = 0.85  # 85% pass rate required
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        logger.info("Starting autonomous quality gate validation...")
        start_time = time.time()
        
        quality_gates = []
        
        # Quality Gate 1: Code Quality Analysis
        logger.info("Running code quality analysis...")
        quality_gates.append(self._run_code_quality_gate())
        
        # Quality Gate 2: Functional Testing
        logger.info("Running functional testing...")
        quality_gates.append(self._run_functional_testing_gate())
        
        # Quality Gate 3: Security Scanning
        logger.info("Running security scanning...")
        quality_gates.append(self._run_security_scanning_gate())
        
        # Quality Gate 4: Performance Validation
        logger.info("Running performance validation...")
        quality_gates.append(self._run_performance_gate())
        
        # Quality Gate 5: Integration Testing
        logger.info("Running integration testing...")
        quality_gates.append(self._run_integration_gate())
        
        # Calculate overall results
        total_execution_time = time.time() - start_time
        passed_gates = sum(1 for gate in quality_gates if gate.passed)
        overall_score = sum(gate.score for gate in quality_gates) / len(quality_gates)
        overall_passed = overall_score >= self.quality_threshold
        
        # Generate comprehensive report
        report = {
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'quality_threshold': self.quality_threshold,
            'gates_passed': passed_gates,
            'total_gates': len(quality_gates),
            'execution_time': total_execution_time,
            'quality_gates': [gate.to_dict() for gate in quality_gates],
            'recommendations': self._generate_recommendations(quality_gates),
            'timestamp': time.time()
        }
        
        logger.info(f"Quality gate validation completed: "
                   f"{passed_gates}/{len(quality_gates)} passed, "
                   f"overall score: {overall_score:.2%}")
        
        return report
    
    def _run_code_quality_gate(self) -> QualityGateResult:
        """Run code quality analysis gate."""
        try:
            metrics = self.code_analyzer.analyze_python_files()
            
            # Calculate quality score based on metrics
            score_components = []
            
            # Docstring coverage (weight: 0.3)
            docstring_score = metrics['docstring_coverage']
            score_components.append(docstring_score * 0.3)
            
            # Security issues (weight: 0.4) - inversely scored
            security_penalty = min(len(metrics['security_issues']) * 0.2, 1.0)
            security_score = max(0.0, 1.0 - security_penalty)
            score_components.append(security_score * 0.4)
            
            # Complexity issues (weight: 0.3) - inversely scored
            complexity_penalty = min(len(metrics['complexity_issues']) * 0.1, 1.0)
            complexity_score = max(0.0, 1.0 - complexity_penalty)
            score_components.append(complexity_score * 0.3)
            
            total_score = sum(score_components)
            passed = total_score >= 0.7  # 70% threshold for code quality
            
            message = (
                f"Code quality: {total_score:.1%} - "
                f"{metrics['total_files']} files, {metrics['total_lines']} lines, "
                f"{metrics['docstring_coverage']:.1%} docstring coverage, "
                f"{len(metrics['security_issues'])} security issues"
            )
            
            return QualityGateResult(
                gate_name="code_quality",
                passed=passed,
                score=total_score,
                message=message,
                details=metrics,
                timestamp=time.time()
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="code_quality",
                passed=False,
                score=0.0,
                message=f"Code quality analysis failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _run_functional_testing_gate(self) -> QualityGateResult:
        """Run functional testing gate."""
        try:
            test_results = self.code_analyzer.run_basic_tests()
            
            passed_tests = sum(1 for test in test_results if test.passed)
            total_tests = len(test_results)
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            passed = score >= 0.8  # 80% test pass rate required
            
            avg_execution_time = (
                sum(test.execution_time for test in test_results) / total_tests
                if total_tests > 0 else 0.0
            )
            
            message = (
                f"Functional testing: {score:.1%} pass rate - "
                f"{passed_tests}/{total_tests} tests passed, "
                f"avg execution time: {avg_execution_time:.2f}s"
            )
            
            return QualityGateResult(
                gate_name="functional_testing",
                passed=passed,
                score=score,
                message=message,
                details={
                    'test_results': [test.to_dict() for test in test_results],
                    'avg_execution_time': avg_execution_time
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="functional_testing",
                passed=False,
                score=0.0,
                message=f"Functional testing failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _run_security_scanning_gate(self) -> QualityGateResult:
        """Run security scanning gate."""
        try:
            security_results = self.security_scanner.scan_codebase(self.base_path / "src")
            
            # Calculate security score
            high_risk_count = len(security_results['high_risk_patterns'])
            medium_risk_count = len(security_results['medium_risk_patterns'])
            
            # Penalize based on risk levels
            security_penalty = (high_risk_count * 0.3) + (medium_risk_count * 0.1)
            security_score = max(0.0, 1.0 - security_penalty)
            
            # Require no high-risk issues and minimal medium-risk
            passed = high_risk_count == 0 and medium_risk_count <= 5
            
            message = (
                f"Security scan: {security_score:.1%} security score - "
                f"{high_risk_count} high-risk, {medium_risk_count} medium-risk issues, "
                f"{security_results['files_scanned']} files scanned"
            )
            
            return QualityGateResult(
                gate_name="security_scanning",
                passed=passed,
                score=security_score,
                message=message,
                details=security_results,
                timestamp=time.time()
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="security_scanning",
                passed=False,
                score=0.0,
                message=f"Security scanning failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _run_performance_gate(self) -> QualityGateResult:
        """Run performance validation gate."""
        try:
            # Test performance of key operations
            performance_results = []
            
            # Test 1: Simple circuit creation speed
            start_time = time.time()
            sys.path.append(str(self.base_path / "src"))
            
            spec = importlib.util.spec_from_file_location(
                "core_standalone",
                self.base_path / "src" / "photonic_foundry" / "core_standalone.py"
            )
            core_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core_module)
            
            accelerator = core_module.PhotonicAccelerator()
            circuit = accelerator.convert_simple_model([10, 20, 10])
            creation_time = time.time() - start_time
            performance_results.append(('circuit_creation', creation_time))
            
            # Test 2: Verilog generation speed
            start_time = time.time()
            verilog = circuit.generate_full_verilog()
            verilog_time = time.time() - start_time
            performance_results.append(('verilog_generation', verilog_time))
            
            # Test 3: Metrics estimation speed
            start_time = time.time()
            metrics = circuit.estimate_metrics()
            metrics_time = time.time() - start_time
            performance_results.append(('metrics_estimation', metrics_time))
            
            # Calculate performance score
            # Requirements: < 1s for circuit creation, < 0.5s for verilog, < 0.1s for metrics
            thresholds = [1.0, 0.5, 0.1]
            scores = []
            for i, (name, exec_time) in enumerate(performance_results):
                score = max(0.0, 1.0 - (exec_time / thresholds[i]))
                scores.append(score)
            
            overall_performance_score = sum(scores) / len(scores)
            passed = overall_performance_score >= 0.8
            
            message = (
                f"Performance validation: {overall_performance_score:.1%} - "
                f"Circuit creation: {creation_time:.3f}s, "
                f"Verilog gen: {verilog_time:.3f}s, "
                f"Metrics: {metrics_time:.3f}s"
            )
            
            return QualityGateResult(
                gate_name="performance_validation",
                passed=passed,
                score=overall_performance_score,
                message=message,
                details={
                    'performance_results': dict(performance_results),
                    'thresholds': dict(zip(['circuit_creation', 'verilog_generation', 'metrics_estimation'], thresholds))
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_validation",
                passed=False,
                score=0.0,
                message=f"Performance validation failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _run_integration_gate(self) -> QualityGateResult:
        """Run integration testing gate."""
        try:
            # Test integration between different components
            integration_tests = []
            
            sys.path.append(str(self.base_path / "src"))
            
            # Test 1: Core + Robust framework integration
            try:
                spec = importlib.util.spec_from_file_location(
                    "robust_framework",
                    self.base_path / "src" / "photonic_foundry" / "robust_framework.py"
                )
                robust_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(robust_module)
                
                accelerator = robust_module.RobustPhotonicAccelerator()
                result = accelerator.convert_simple_model_robust([4, 8, 2])
                integration_tests.append(('core_robust_integration', True, result))
                
            except Exception as e:
                integration_tests.append(('core_robust_integration', False, str(e)))
            
            # Test 2: Core + Scaling engine integration
            try:
                spec = importlib.util.spec_from_file_location(
                    "scaling_engine",
                    self.base_path / "src" / "photonic_foundry" / "scaling_engine.py"
                )
                scaling_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scaling_module)
                
                scaling_accelerator = scaling_module.ScalingPhotonicAccelerator()
                result = scaling_accelerator.process_circuit_optimized([4, 8, 2])
                scaling_accelerator.shutdown()
                integration_tests.append(('core_scaling_integration', True, result))
                
            except Exception as e:
                integration_tests.append(('core_scaling_integration', False, str(e)))
            
            # Calculate integration score
            passed_integrations = sum(1 for test_name, passed, result in integration_tests if passed)
            total_integrations = len(integration_tests)
            integration_score = passed_integrations / total_integrations if total_integrations > 0 else 0.0
            passed = integration_score >= 0.9  # 90% integration pass rate
            
            message = (
                f"Integration testing: {integration_score:.1%} - "
                f"{passed_integrations}/{total_integrations} integrations passed"
            )
            
            return QualityGateResult(
                gate_name="integration_testing",
                passed=passed,
                score=integration_score,
                message=message,
                details={
                    'integration_tests': [
                        {'name': name, 'passed': passed, 'result': str(result)[:100]}
                        for name, passed, result in integration_tests
                    ]
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="integration_testing",
                passed=False,
                score=0.0,
                message=f"Integration testing failed: {str(e)}",
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _generate_recommendations(self, quality_gates: List[QualityGateResult]) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for gate in quality_gates:
            if not gate.passed:
                if gate.gate_name == "code_quality":
                    recommendations.append("Improve code quality: Add more docstrings and address complexity issues")
                elif gate.gate_name == "functional_testing":
                    recommendations.append("Fix failing functional tests to improve reliability")
                elif gate.gate_name == "security_scanning":
                    recommendations.append("Address security issues: Review and fix high/medium risk patterns")
                elif gate.gate_name == "performance_validation":
                    recommendations.append("Optimize performance: Focus on slow operations identified in testing")
                elif gate.gate_name == "integration_testing":
                    recommendations.append("Fix integration issues: Ensure all components work together correctly")
        
        if not recommendations:
            recommendations.append("All quality gates passed! System meets production readiness criteria.")
        
        return recommendations


def main():
    """Main execution function."""
    print("🛡️ AUTONOMOUS QUALITY VALIDATION SYSTEM")
    print("=" * 60)
    
    # Initialize quality gate system
    base_path = Path(__file__).parent
    quality_gates = AutonomousQualityGates(base_path)
    
    # Run all quality gates
    report = quality_gates.run_all_quality_gates()
    
    # Save detailed report
    os.makedirs('output', exist_ok=True)
    with open('output/autonomous_quality_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n📊 QUALITY GATE RESULTS:")
    print(f"Overall Status: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
    print(f"Overall Score: {report['overall_score']:.1%}")
    print(f"Gates Passed: {report['gates_passed']}/{report['total_gates']}")
    print(f"Execution Time: {report['execution_time']:.2f}s")
    
    print(f"\n🔍 INDIVIDUAL GATES:")
    for gate_result in report['quality_gates']:
        status = "✅" if gate_result['passed'] else "❌"
        print(f"  {status} {gate_result['gate_name']}: {gate_result['score']:.1%} - {gate_result['message']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📁 Detailed report saved to: output/autonomous_quality_report.json")
    
    if report['overall_passed']:
        print("\n🎉 QUALITY GATES PASSED - SYSTEM READY FOR PRODUCTION!")
        return 0
    else:
        print("\n⚠️  QUALITY GATES FAILED - SYSTEM NEEDS IMPROVEMENTS")
        return 1


if __name__ == "__main__":
    sys.exit(main())