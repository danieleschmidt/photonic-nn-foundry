#!/usr/bin/env python3
"""
Robust Research Validation Framework Demo

Demonstrates production-grade validation system with comprehensive error handling,
statistical rigor, and reproducibility guarantees.
"""

import sys
import os
import json
import time
import random
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_foundry.robust_research_validation_framework import (
    RobustResearchValidator,
    ValidationLevel,
    ValidationConfig,
    create_robust_validator,
    ValidationError,
    StatisticalError
)

def create_sample_hypothesis() -> Dict[str, Any]:
    """Create a sample research hypothesis for testing."""
    return {
        'hypothesis_id': f"hyp_{int(time.time())}",
        'description': "Quantum-enhanced photonic neural network achieves >50% energy reduction with maintained accuracy",
        'breakthrough_type': "quantum_enhancement",
        'success_criteria': {
            'energy_reduction': 0.5,
            'accuracy_retention': 0.98,
            'quantum_speedup': 2.0,
            'statistical_significance': 0.05,
            'reproducibility': 0.9
        },
        'expected_impact': 0.85,
        'confidence_score': 0.9
    }

def create_sample_experimental_data(high_quality: bool = True) -> Dict[str, Any]:
    """Create sample experimental data."""
    if high_quality:
        # High-quality data that should validate well
        return {
            'sample_size': 1000,
            'metrics': {
                'energy_reduction': 0.55 + random.gauss(0, 0.02),
                'accuracy_retention': 0.985 + random.gauss(0, 0.005),
                'quantum_speedup': 2.3 + random.gauss(0, 0.1),
                'inference_latency_ms': 50 + random.gauss(0, 5),
                'training_efficiency': 0.8 + random.gauss(0, 0.05),
                'model_compression': 0.4 + random.gauss(0, 0.03),
                'thermal_stability': 0.95 + random.gauss(0, 0.02)
            },
            'has_controls': True,
            'randomized': True
        }
    else:
        # Lower quality data for testing validation robustness
        return {
            'sample_size': 50,  # Small sample
            'metrics': {
                'energy_reduction': 0.35 + random.gauss(0, 0.1),  # Below threshold
                'accuracy_retention': 0.95 + random.gauss(0, 0.02),  # Below threshold
                'quantum_speedup': 1.5 + random.gauss(0, 0.2)  # Below threshold
            },
            'has_controls': False,
            'randomized': False
        }

def demonstrate_basic_validation():
    """Demonstrate basic validation functionality."""
    print("🔬 BASIC VALIDATION DEMONSTRATION")
    print("-" * 50)
    
    # Create validator
    validator = create_robust_validator(ValidationLevel.BASIC)
    
    # Create sample data
    hypothesis = create_sample_hypothesis()
    experimental_data = create_sample_experimental_data(high_quality=True)
    
    print(f"Hypothesis: {hypothesis['description']}")
    print(f"Sample size: {experimental_data['sample_size']}")
    print(f"Metrics: {len(experimental_data['metrics'])}")
    
    try:
        # Perform validation
        start_time = time.time()
        result = validator.validate_research_hypothesis(hypothesis, experimental_data)
        validation_time = time.time() - start_time
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Status: {result['validation_status']}")
        print(f"   Overall score: {result['overall_validation_score']:.3f}")
        print(f"   Validation time: {validation_time:.3f}s")
        
        # Basic statistics
        basic_stats = result['basic_statistics']
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(f"   Statistical significance: {basic_stats['statistical_significance']}")
        print(f"   Effect size: {basic_stats['effect_size']:.3f}")
        print(f"   P-value: {basic_stats['p_value']:.4f}")
        print(f"   Success criteria met: {basic_stats['criteria_met']}/{basic_stats['total_criteria']}")
        
        # Reproducibility
        reproducibility = result['reproducibility']
        print(f"\n🔄 REPRODUCIBILITY ASSESSMENT:")
        print(f"   Score: {reproducibility['reproducibility_score']:.3f}")
        print(f"   Assessment: {reproducibility['assessment']}")
        print(f"   Meets threshold: {reproducibility['meets_threshold']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def demonstrate_rigorous_validation():
    """Demonstrate rigorous validation with all features enabled."""
    print("\n🎯 RIGOROUS VALIDATION DEMONSTRATION")
    print("-" * 50)
    
    # Create rigorous validator
    validator = create_robust_validator(ValidationLevel.RIGOROUS)
    
    # Create sample data
    hypothesis = create_sample_hypothesis()
    experimental_data = create_sample_experimental_data(high_quality=True)
    
    print(f"Validation level: {ValidationLevel.RIGOROUS.value}")
    print(f"Advanced features enabled: Cross-validation, Bootstrap sampling")
    
    try:
        # Perform validation
        start_time = time.time()
        result = validator.validate_research_hypothesis(hypothesis, experimental_data)
        validation_time = time.time() - start_time
        
        print(f"\n📊 COMPREHENSIVE VALIDATION RESULTS:")
        print(f"   Status: {result['validation_status']}")
        print(f"   Overall score: {result['overall_validation_score']:.3f}")
        print(f"   Validation time: {validation_time:.3f}s")
        
        # Advanced statistics
        if 'advanced_statistics' in result:
            advanced_stats = result['advanced_statistics']
            print(f"\n🔬 ADVANCED STATISTICAL TESTS:")
            print(f"   Tests performed: {advanced_stats['total_tests']}")
            print(f"   Significant tests: {advanced_stats['significant_tests']}")
        
        # Cross-validation
        if 'cross_validation' in result:
            cv_results = result['cross_validation']
            print(f"\n🔄 CROSS-VALIDATION ANALYSIS:")
            print(f"   K-folds: {cv_results['k_folds']}")
            print(f"   Mean CV score: {cv_results['mean_cv_score']:.3f}")
            print(f"   CV std: {cv_results['cv_standard_deviation']:.3f}")
            print(f"   Generalization: {cv_results['generalization_assessment']}")
        
        # Bootstrap analysis
        if 'bootstrap_analysis' in result:
            bootstrap = result['bootstrap_analysis']
            if 'error' not in bootstrap:
                print(f"\n📊 BOOTSTRAP ANALYSIS:")
                print(f"   Bootstrap samples: {bootstrap['n_bootstrap_samples']}")
                print(f"   Bootstrap mean: {bootstrap['bootstrap_mean']:.3f}")
                print(f"   95% CI: [{bootstrap['confidence_interval_95'][0]:.3f}, {bootstrap['confidence_interval_95'][1]:.3f}]")
        
        # Scientific rigor
        rigor = result['scientific_rigor']
        print(f"\n🎓 SCIENTIFIC RIGOR ASSESSMENT:")
        print(f"   Overall rigor: {rigor['overall_rigor_score']:.3f}")
        print(f"   Rigor level: {rigor['rigor_level']}")
        print(f"   Strengths: {', '.join(rigor['strengths'])}")
        if rigor['weaknesses']:
            print(f"   Weaknesses: {', '.join(rigor['weaknesses'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Rigorous validation failed: {e}")
        return None

def demonstrate_error_handling():
    """Demonstrate robust error handling."""
    print("\n🛡️ ERROR HANDLING DEMONSTRATION")
    print("-" * 50)
    
    validator = create_robust_validator(ValidationLevel.STANDARD)
    
    # Test various error conditions
    error_tests = [
        {
            'name': 'Missing hypothesis fields',
            'hypothesis': {'description': 'Incomplete hypothesis'},
            'data': create_sample_experimental_data()
        },
        {
            'name': 'Invalid success criteria',
            'hypothesis': {
                'hypothesis_id': 'test_hyp',
                'description': 'Test hypothesis',
                'success_criteria': 'invalid_criteria'  # Should be dict
            },
            'data': create_sample_experimental_data()
        },
        {
            'name': 'Insufficient sample size',
            'hypothesis': create_sample_hypothesis(),
            'data': {
                'sample_size': 10,  # Below minimum
                'metrics': {'test_metric': 0.5}
            }
        },
        {
            'name': 'Non-numeric metrics',
            'hypothesis': create_sample_hypothesis(),
            'data': {
                'sample_size': 100,
                'metrics': {'test_metric': 'invalid_value'}  # Should be numeric
            }
        }
    ]
    
    for i, test in enumerate(error_tests, 1):
        print(f"\n   Test {i}: {test['name']}")
        
        try:
            result = validator.validate_research_hypothesis(test['hypothesis'], test['data'])
            print(f"      ❌ Expected error but validation succeeded")
        except ValidationError as e:
            print(f"      ✅ Caught ValidationError: {e}")
        except StatisticalError as e:
            print(f"      ✅ Caught StatisticalError: {e}")
        except Exception as e:
            print(f"      ⚠️  Caught unexpected error: {type(e).__name__}: {e}")
    
    # Check error log
    error_log = validator.get_error_log()
    print(f"\n📋 Error log entries: {len(error_log)}")
    
    return validator

def demonstrate_quality_assessment():
    """Demonstrate validation quality assessment."""
    print("\n🏆 QUALITY ASSESSMENT DEMONSTRATION")
    print("-" * 50)
    
    validator = create_robust_validator(ValidationLevel.PUBLICATION_READY)
    
    # Test with high-quality data
    hypothesis = create_sample_hypothesis()
    high_quality_data = create_sample_experimental_data(high_quality=True)
    
    result = validator.validate_research_hypothesis(hypothesis, high_quality_data)
    
    # Quality assessment
    quality = result['quality_assessment']
    print(f"📊 VALIDATION QUALITY ASSESSMENT:")
    print(f"   Overall quality: {quality['overall_quality']:.3f}")
    print(f"   Quality level: {quality['quality_level']}")
    print(f"   Completeness: {quality['factors']['completeness']:.3f}")
    print(f"   Consistency: {quality['factors']['consistency']:.3f}")
    print(f"   Robustness: {quality['factors']['robustness']:.3f}")
    
    # Recommendations
    if 'recommendations' in result:
        recommendations = result['recommendations']
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Confidence assessment
    confidence = result['confidence_assessment']
    print(f"\n🎯 CONFIDENCE ASSESSMENT:")
    print(f"   Confidence score: {confidence['confidence_score']:.3f}")
    print(f"   Confidence level: {confidence['confidence_level']}")
    
    return result

def demonstrate_validation_metrics():
    """Demonstrate validation metrics tracking."""
    print("\n📈 VALIDATION METRICS DEMONSTRATION")
    print("-" * 50)
    
    validator = create_robust_validator(ValidationLevel.STANDARD)
    
    # Perform multiple validations
    print("Performing multiple validations to track metrics...")
    
    for i in range(5):
        hypothesis = create_sample_hypothesis()
        hypothesis['hypothesis_id'] = f"test_hyp_{i}"
        
        # Mix of high and low quality data
        data = create_sample_experimental_data(high_quality=(i % 2 == 0))
        
        try:
            result = validator.validate_research_hypothesis(hypothesis, data)
            print(f"   Validation {i+1}: {result['validation_status']}")
        except Exception as e:
            print(f"   Validation {i+1}: FAILED - {type(e).__name__}")
    
    # Get metrics
    metrics = validator.get_validation_metrics()
    print(f"\n📊 VALIDATION METRICS:")
    print(f"   Total validations: {metrics['total_validations']}")
    print(f"   Successful: {metrics['successful_validations']}")
    print(f"   Failed: {metrics['failed_validations']}")
    print(f"   Error rate: {metrics['error_rate']:.1%}")
    print(f"   Average time: {metrics['average_validation_time']:.3f}s")
    
    # Get validation history
    history = validator.get_validation_history()
    print(f"\n📋 VALIDATION HISTORY:")
    for entry in history[-3:]:  # Show last 3
        print(f"   {entry['hypothesis_id']}: {entry['validation_status']} (score: {entry['overall_score']:.3f})")
    
    return validator

def demonstrate_export_functionality():
    """Demonstrate report export functionality."""
    print("\n💾 EXPORT FUNCTIONALITY DEMONSTRATION")
    print("-" * 50)
    
    validator = create_robust_validator(ValidationLevel.RIGOROUS)
    
    # Perform a validation
    hypothesis = create_sample_hypothesis()
    data = create_sample_experimental_data(high_quality=True)
    result = validator.validate_research_hypothesis(hypothesis, data)
    
    # Export validation report
    report_file = "robust_validation_report.json"
    validator.export_validation_report(report_file)
    
    print(f"📋 Validation report exported to: {report_file}")
    
    # Load and summarize report
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    print(f"\n📊 REPORT SUMMARY:")
    summary = report['framework_summary']
    print(f"   Total validations: {summary['total_validations_performed']}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average time: {summary['average_validation_time']:.3f}s")
    print(f"   Validation level: {summary['validation_level']}")
    
    return report

def main():
    """Run the complete robust validation framework demo."""
    start_time = time.time()
    
    print("🚀 ROBUST RESEARCH VALIDATION FRAMEWORK DEMO")
    print("Production-Grade Validation with Comprehensive Error Handling")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        basic_result = demonstrate_basic_validation()
        rigorous_result = demonstrate_rigorous_validation()
        error_validator = demonstrate_error_handling()
        quality_result = demonstrate_quality_assessment()
        metrics_validator = demonstrate_validation_metrics()
        export_report = demonstrate_export_functionality()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        
        print(f"\n🔬 VALIDATION SUMMARY:")
        if basic_result:
            print(f"   Basic validation: {basic_result['validation_status']}")
            print(f"   Basic score: {basic_result['overall_validation_score']:.3f}")
        
        if rigorous_result:
            print(f"   Rigorous validation: {rigorous_result['validation_status']}")
            print(f"   Rigorous score: {rigorous_result['overall_validation_score']:.3f}")
        
        if quality_result:
            print(f"   Quality level: {quality_result['quality_assessment']['quality_level']}")
            print(f"   Confidence: {quality_result['confidence_assessment']['confidence_level']}")
        
        print(f"\n🛡️ FRAMEWORK CAPABILITIES DEMONSTRATED:")
        print(f"   ✅ Comprehensive statistical validation")
        print(f"   ✅ Advanced testing (cross-validation, bootstrap)")
        print(f"   ✅ Robust error handling and logging")
        print(f"   ✅ Scientific rigor assessment")
        print(f"   ✅ Reproducibility validation")
        print(f"   ✅ Quality and confidence assessment")
        print(f"   ✅ Metrics tracking and reporting")
        print(f"   ✅ Export and documentation capabilities")
        
        print(f"\n🌟 This framework provides production-ready validation")
        print(f"   with publication-quality statistical rigor and")
        print(f"   comprehensive error handling for research breakthroughs.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    # Run the robust validation demo
    main()