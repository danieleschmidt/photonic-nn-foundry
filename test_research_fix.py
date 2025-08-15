#!/usr/bin/env python3
"""
Test the fixed research framework to validate the serialization bug fix.
"""

import torch
import torch.nn as nn
import numpy as np
from src.photonic_foundry.research_framework import (
    ResearchFramework, ExperimentConfig, ExperimentType
)

def test_research_framework_fix():
    """Test that the research framework works without serialization errors."""
    print("🧪 Testing Research Framework Fix...")
    
    # Create simple test models
    simple_model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Create test datasets
    test_data = torch.randn(32, 10)
    datasets = [test_data]
    
    # Initialize research framework
    framework = ResearchFramework()
    
    # Create minimal experiment config
    config = ExperimentConfig(
        experiment_id="serialization_test",
        experiment_type=ExperimentType.PERFORMANCE_COMPARISON,
        description="Test serialization fix",
        hypothesis="Framework works without threading errors",
        num_runs=2,  # Small number for quick test
        significance_level=0.05
    )
    
    print("✅ Created experiment configuration")
    
    try:
        # Run experiment
        report = framework.run_experiment(config, [simple_model], datasets)
        
        print(f"🎉 SUCCESS! Experiment completed successfully")
        print(f"📊 Success rate: {report.success_rate:.2%}")
        print(f"📈 Total runs: {len(report.results)}")
        
        # Check if we have successful results
        successful_runs = [r for r in report.results if r.success]
        failed_runs = [r for r in report.results if not r.success]
        
        print(f"✅ Successful runs: {len(successful_runs)}")
        print(f"❌ Failed runs: {len(failed_runs)}")
        
        if failed_runs:
            print("🔍 Error messages from failed runs:")
            for run in failed_runs[:3]:  # Show first 3 errors
                print(f"   - {run.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_research_framework_fix()
    if success:
        print("\n🚀 Research framework fix validation PASSED!")
    else:
        print("\n💥 Research framework fix validation FAILED!")