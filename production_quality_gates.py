#!/usr/bin/env python3
"""
Production Quality Gates - Enterprise-Grade Validation

This script executes production-ready quality gates using sophisticated analysis
to ensure enterprise-grade quality standards for deployment.
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import our production security analyzer
from production_security_compliance import ProductionSecurityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionQualityGateResult:
    """Result of a production quality gate execution."""
    
    def __init__(self, gate_name: str, success: bool, score: float, details: Dict[str, Any]):
        self.gate_name = gate_name
        self.success = success
        self.score = score  # 0.0 to 1.0
        self.details = details
        self.timestamp = time.time()
    
    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        return f"{status} {self.gate_name}: {self.score:.1%} ({self.details.get('summary', 'No summary')})"


class ProductionSecurityGate:
    """Production-grade security validation with context awareness."""
    
    async def execute(self) -> ProductionQualityGateResult:
        """Execute production security analysis."""
        logger.info("🔒 Executing Production Security Gate")
        
        try:
            # Use our sophisticated security analyzer
            analyzer = ProductionSecurityAnalyzer()
            src_path = Path('src')
            
            analysis_results = analyzer.analyze_codebase(src_path)
            
            # Calculate security score
            total_issues = analysis_results['total_issues']
            high_severity = analysis_results['high_severity_issues']
            medium_severity = analysis_results['medium_severity_issues']
            false_positives = len(analysis_results['false_positives'])
            
            # Production security scoring
            security_score = 1.0
            security_score -= high_severity * 0.3    # 30% penalty per high severity
            security_score -= medium_severity * 0.1  # 10% penalty per medium severity
            
            security_score = max(0.0, security_score)
            
            # Production requirements: NO high severity issues
            production_ready = (high_severity == 0 and security_score >= 0.9)
            
            # Check for security policy files
            security_files = {
                'security_policy': Path('production_security_policy.json').exists(),
                'security_guidelines': Path('.security').exists(),
                'env_template': Path('.env.template').exists()
            }
            
            return ProductionQualityGateResult(
                gate_name="Production Security",
                success=production_ready,
                score=security_score,
                details={
                    'summary': f'{total_issues} issues, {high_severity} high severity, {false_positives} false positives',
                    'analysis_results': analysis_results,
                    'security_files': security_files,
                    'production_criteria': {
                        'high_severity_issues': high_severity,
                        'security_score': security_score,
                        'policy_files_present': all(security_files.values())
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Production security gate failed: {e}")
            return ProductionQualityGateResult(
                gate_name="Production Security",
                success=False,
                score=0.0,
                details={
                    'summary': f'Security analysis failed: {str(e)}',
                    'error': str(e)
                }
            )


class EnhancedCodeQualityGate:
    """Enhanced code quality analysis."""
    
    async def execute(self) -> ProductionQualityGateResult:
        """Execute enhanced code quality checks."""
        logger.info("🔍 Executing Enhanced Code Quality Gate")
        
        try:
            src_path = Path('src')
            if not src_path.exists():
                return ProductionQualityGateResult(
                    gate_name="Enhanced Code Quality",
                    success=True,
                    score=0.8,
                    details={'summary': 'No src directory found, assuming basic quality'}
                )
            
            python_files = list(src_path.rglob('*.py'))
            
            # Analyze code quality metrics
            total_lines = 0
            total_functions = 0
            documented_functions = 0
            complex_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)
                        
                        # Analyze functions
                        for i, line in enumerate(lines):
                            line = line.strip()
                            
                            if line.startswith('def ') or line.startswith('async def '):
                                total_functions += 1
                                
                                # Check for docstring
                                has_docstring = False
                                for j in range(i + 1, min(i + 5, len(lines))):
                                    next_line = lines[j].strip()
                                    if next_line.startswith('"""') or next_line.startswith("'''"):
                                        has_docstring = True
                                        break
                                    elif next_line and not next_line.startswith('#'):
                                        break
                                
                                if has_docstring:
                                    documented_functions += 1
                                
                                # Check complexity (simple heuristic)
                                function_lines = []
                                indent_level = len(line) - len(line.lstrip())
                                for j in range(i + 1, len(lines)):
                                    if (lines[j].strip() and 
                                        len(lines[j]) - len(lines[j].lstrip()) <= indent_level and
                                        not lines[j].strip().startswith(('#', '"""', "'"))):
                                        break
                                    function_lines.append(lines[j])
                                
                                # Count complexity indicators
                                complexity_indicators = sum(1 for line in function_lines 
                                                          if any(keyword in line for keyword in 
                                                               ['if ', 'elif ', 'for ', 'while ', 'try:', 'except']))
                                
                                if complexity_indicators > 10:  # High complexity threshold
                                    complex_functions += 1
                
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate quality metrics
            doc_coverage = documented_functions / total_functions if total_functions > 0 else 0
            complexity_score = max(0.0, 1.0 - (complex_functions / total_functions)) if total_functions > 0 else 1.0
            size_score = max(0.0, 1.0 - (total_lines / 100000))  # Penalize excessive size
            
            # Enhanced scoring with higher standards
            overall_score = (doc_coverage * 0.4 + complexity_score * 0.3 + size_score * 0.2 + 0.1)
            
            success = overall_score >= 0.8  # Higher threshold for production
            
            return ProductionQualityGateResult(
                gate_name="Enhanced Code Quality",
                success=success,
                score=overall_score,
                details={
                    'summary': f'{len(python_files)} files, {total_lines} lines, {doc_coverage:.1%} documented',
                    'metrics': {
                        'total_files': len(python_files),
                        'total_lines': total_lines,
                        'total_functions': total_functions,
                        'documented_functions': documented_functions,
                        'complex_functions': complex_functions,
                        'documentation_coverage': doc_coverage,
                        'complexity_score': complexity_score
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced code quality gate failed: {e}")
            return ProductionQualityGateResult(
                gate_name="Enhanced Code Quality",
                success=False,
                score=0.0,
                details={'summary': f'Analysis failed: {str(e)}', 'error': str(e)}
            )


class ProductionTestCoverageGate:
    """Production-grade test coverage analysis."""
    
    async def execute(self) -> ProductionQualityGateResult:
        """Execute production test coverage analysis."""
        logger.info("🧪 Executing Production Test Coverage Gate")
        
        try:
            # Find test files
            test_patterns = ['test_*.py', '*_test.py', 'tests.py']
            test_files = []
            
            for pattern in test_patterns:
                test_files.extend(Path('.').rglob(pattern))
            
            # Also check tests/ directory
            tests_dir = Path('tests')
            if tests_dir.exists():
                test_files.extend(tests_dir.rglob('*.py'))
            
            # Remove duplicates
            test_files = list(set(test_files))
            
            # Analyze test files
            total_test_functions = 0
            async_tests = 0
            integration_tests = 0
            unit_tests = 0
            performance_tests = 0
            security_tests = 0
            
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if (line.startswith('def test_') or 
                                line.startswith('async def test_')):
                                total_test_functions += 1
                                
                                # Categorize tests
                                line_lower = line.lower()
                                if 'async def' in line:
                                    async_tests += 1
                                if 'integration' in line_lower:
                                    integration_tests += 1
                                elif 'performance' in line_lower or 'benchmark' in line_lower:
                                    performance_tests += 1
                                elif 'security' in line_lower:
                                    security_tests += 1
                                else:
                                    unit_tests += 1
                
                except Exception as e:
                    logger.warning(f"Could not analyze test file {test_file}: {e}")
            
            # Estimate coverage based on source code
            src_files = list(Path('src').rglob('*.py')) if Path('src').exists() else []
            total_src_lines = 0
            
            for src_file in src_files:
                try:
                    with open(src_file, 'r', encoding='utf-8') as f:
                        total_src_lines += len(f.readlines())
                except Exception:
                    pass
            
            # Production coverage estimation
            if total_src_lines > 0:
                # Higher standards: 1 test per 5 lines for good coverage
                expected_tests = total_src_lines / 5
                coverage_estimate = min(1.0, total_test_functions / max(1, expected_tests))
            else:
                coverage_estimate = 0.0
            
            # Production requirements
            min_tests = 50  # Higher minimum
            min_coverage = 0.6  # 60% coverage minimum
            min_test_types = 3  # Need variety
            
            # Calculate test quality score
            test_count_score = min(1.0, total_test_functions / min_tests)
            coverage_score = coverage_estimate
            variety_score = min(1.0, 
                               (unit_tests > 0) + (integration_tests > 0) + 
                               (performance_tests > 0) + (security_tests > 0)) / 4.0
            async_score = min(1.0, async_tests / max(1, total_test_functions * 0.2))  # 20% async target
            
            test_score = (test_count_score * 0.3 + coverage_score * 0.4 + 
                         variety_score * 0.2 + async_score * 0.1)
            
            success = (total_test_functions >= min_tests and 
                      coverage_estimate >= min_coverage and
                      variety_score >= 0.5)  # At least 2 test types
            
            return ProductionQualityGateResult(
                gate_name="Production Test Coverage",
                success=success,
                score=test_score,
                details={
                    'summary': f'{total_test_functions} tests, {coverage_estimate:.1%} estimated coverage',
                    'metrics': {
                        'test_files_found': len(test_files),
                        'total_tests': total_test_functions,
                        'estimated_coverage': coverage_estimate,
                        'unit_tests': unit_tests,
                        'integration_tests': integration_tests,
                        'performance_tests': performance_tests,
                        'security_tests': security_tests,
                        'async_tests': async_tests
                    },
                    'requirements_met': {
                        'test_count': total_test_functions >= min_tests,
                        'coverage': coverage_estimate >= min_coverage,
                        'variety': variety_score >= 0.5
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Production test coverage gate failed: {e}")
            return ProductionQualityGateResult(
                gate_name="Production Test Coverage",
                success=False,
                score=0.0,
                details={'summary': f'Test analysis failed: {str(e)}', 'error': str(e)}
            )


class ProductionQualityGateExecutor:
    """Execute all production quality gates."""
    
    def __init__(self):
        # Import other gates from the original script
        from comprehensive_quality_gates import PerformanceGate, ComplianceGate
        
        self.gates = [
            EnhancedCodeQualityGate(),
            ProductionSecurityGate(),
            PerformanceGate(),
            ComplianceGate(),
            ProductionTestCoverageGate()
        ]
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all production quality gates."""
        logger.info("🎯 Starting Production Quality Gate Execution")
        
        start_time = time.time()
        results = []
        
        # Execute all gates
        for gate in self.gates:
            try:
                result = await gate.execute()
                results.append(result)
                print(f"   {result}")
            except Exception as e:
                logger.error(f"Gate execution failed: {e}")
                results.append(ProductionQualityGateResult(
                    gate_name=type(gate).__name__,
                    success=False,
                    score=0.0,
                    details={'error': str(e)}
                ))
        
        total_time = time.time() - start_time
        
        # Calculate overall metrics with production standards
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.success)
        failed_gates = total_gates - passed_gates
        
        overall_score = sum(r.score for r in results) / total_gates if total_gates > 0 else 0.0
        pass_rate = passed_gates / total_gates if total_gates > 0 else 0.0
        
        # Production readiness requires ALL gates to pass with high scores
        production_ready = (pass_rate == 1.0 and overall_score >= 0.9)
        
        report = {
            'execution_time': total_time,
            'timestamp': time.time(),
            'overall_success': pass_rate >= 0.8,
            'production_ready': production_ready,
            'overall_score': overall_score,
            'pass_rate': pass_rate,
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'gate_results': [
                {
                    'gate_name': r.gate_name,
                    'success': r.success,
                    'score': r.score,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        return report
    
    def generate_production_report(self, report: Dict[str, Any]) -> str:
        """Generate production quality report."""
        
        report_lines = [
            "🎯 PRODUCTION QUALITY GATE REPORT",
            "=" * 70,
            f"Execution Time: {report['execution_time']:.2f} seconds",
            f"Overall Success: {'✅ PASS' if report['overall_success'] else '❌ FAIL'}",
            f"Production Ready: {'🚀 YES' if report['production_ready'] else '⏸️ NO'}",
            f"Overall Score: {report['overall_score']:.1%}",
            f"Pass Rate: {report['pass_rate']:.1%} ({report['passed_gates']}/{report['total_gates']})",
            "",
            "📋 Individual Gate Results:",
            "-" * 50
        ]
        
        for gate_result in report['gate_results']:
            status = "✅ PASS" if gate_result['success'] else "❌ FAIL"
            report_lines.append(
                f"{status} {gate_result['gate_name']}: {gate_result['score']:.1%} - {gate_result['details'].get('summary', 'No summary')}"
            )
        
        report_lines.extend([
            "",
            "🏆 PRODUCTION ASSESSMENT:",
            "-" * 50
        ])
        
        if report['production_ready']:
            report_lines.extend([
                "🎉🎉🎉 PRODUCTION DEPLOYMENT APPROVED! 🎉🎉🎉",
                "All quality gates passed with excellent scores.",
                "Framework meets enterprise-grade standards.",
                "Ready for global-first production deployment."
            ])
        elif report['overall_success']:
            report_lines.extend([
                "✅ QUALITY VALIDATION SUCCESSFUL",
                "Most quality gates passed. Minor improvements recommended.",
                "Framework is ready for staging environment.",
                "Additional improvements needed for production."
            ])
        else:
            report_lines.extend([
                "❌ QUALITY VALIDATION FAILED",
                "Critical quality gates failed. Significant improvements required.",
                "Framework is NOT ready for production deployment."
            ])
        
        report_lines.extend([
            "",
            "=" * 70,
            "Production quality gate execution complete."
        ])
        
        return "\n".join(report_lines)


async def main():
    """Main execution function."""
    print("🎯 Production Quality Gates - Enterprise Readiness Validation")
    print("=" * 80)
    
    try:
        # Execute production quality gates
        executor = ProductionQualityGateExecutor()
        report = await executor.execute_all_gates()
        
        # Generate and display report
        quality_report = executor.generate_production_report(report)
        print("\n" + quality_report)
        
        # Save report
        report_file = Path("production_quality_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report['production_ready']
        
    except Exception as e:
        logger.error(f"Production quality gate execution failed: {e}")
        print(f"\n💥 Production quality gate execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n🚀 PRODUCTION DEPLOYMENT APPROVED")
    else:
        print("\n⏸️ PRODUCTION DEPLOYMENT BLOCKED")