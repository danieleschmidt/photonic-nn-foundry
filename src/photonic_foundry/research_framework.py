"""
Research framework for quantum-photonic neural networks.
Implements comprehensive experimental design, benchmarking, and validation.
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from abc import ABC, abstractmethod
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - interactive visualizations will be disabled")
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import math

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of research experiments."""
    PERFORMANCE_COMPARISON = "performance_comparison"
    ALGORITHM_VALIDATION = "algorithm_validation"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ACCURACY_ANALYSIS = "accuracy_analysis"
    ROBUSTNESS_TESTING = "robustness_testing"


class MetricType(Enum):
    """Research metrics to track."""
    ENERGY_PER_OP = "energy_per_op"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    AREA_EFFICIENCY = "area_efficiency"
    POWER_CONSUMPTION = "power_consumption"
    CONVERGENCE_TIME = "convergence_time"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    QUANTUM_ADVANTAGE = "quantum_advantage"
    COHERENCE_TIME = "coherence_time"
    FIDELITY = "fidelity"
    GATE_ERROR_RATE = "gate_error_rate"
    DECOHERENCE_RATE = "decoherence_rate"
    OPTIMIZATION_EFFICIENCY = "optimization_efficiency"


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    experiment_id: str
    experiment_type: ExperimentType
    description: str
    hypothesis: str
    success_criteria: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    num_runs: int = 10
    significance_level: float = 0.05
    random_seed: Optional[int] = None
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"experiments/{self.experiment_id}"


@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    run_id: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ExperimentReport:
    """Comprehensive experiment report."""
    config: ExperimentConfig
    results: List[ExperimentResult]
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success for r in self.results) / len(self.results)


class BaselineAlgorithm(ABC):
    """Abstract base class for baseline algorithms."""
    
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute algorithm and return results."""
        pass


class ClassicalCPUBaseline(BaselineAlgorithm):
    """Classical CPU implementation baseline."""
    
    def name(self) -> str:
        return "Classical_CPU"
    
    def run(self, model: nn.Module, input_data: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Run model on CPU."""
        start_time = time.perf_counter()
        
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            
        execution_time = time.perf_counter() - start_time
        
        # Estimate energy consumption (rough approximation)
        # Typical CPU: ~100W during computation
        energy_estimate = execution_time * 100.0  # Watt-seconds
        
        return {
            MetricType.LATENCY.value: execution_time * 1000,  # ms
            MetricType.ENERGY_PER_OP.value: energy_estimate * 1e12,  # pJ (rough estimate)
            MetricType.THROUGHPUT.value: input_data.size(0) / execution_time,  # samples/sec
            "output_shape": output.shape,
            "device": "cpu"
        }


class ClassicalGPUBaseline(BaselineAlgorithm):
    """Classical GPU implementation baseline."""
    
    def name(self) -> str:
        return "Classical_GPU"
    
    def run(self, model: nn.Module, input_data: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Run model on GPU if available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        input_data = input_data.to(device)
        
        # Warm up
        with torch.no_grad():
            _ = model(input_data[:1])
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.perf_counter()
        
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            
        torch.cuda.synchronize() if device.type == "cuda" else None
        execution_time = time.perf_counter() - start_time
        
        # Estimate energy consumption
        # Typical GPU: ~250W during inference
        energy_estimate = execution_time * 250.0 if device.type == "cuda" else execution_time * 100.0
        
        return {
            MetricType.LATENCY.value: execution_time * 1000,  # ms
            MetricType.ENERGY_PER_OP.value: energy_estimate * 1e12,  # pJ (rough estimate)
            MetricType.THROUGHPUT.value: input_data.size(0) / execution_time,  # samples/sec
            "output_shape": output.shape,
            "device": str(device)
        }


class PhotonicBaseline(BaselineAlgorithm):
    """Photonic neural network implementation."""
    
    def __init__(self, accelerator):
        """
        Initialize with photonic accelerator.
        
        Args:
            accelerator: PhotonicAccelerator instance
        """
        self.accelerator = accelerator
    
    def name(self) -> str:
        return "Photonic_Neural_Network"
    
    def run(self, model: nn.Module, input_data: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Run model on photonic accelerator."""
        start_time = time.perf_counter()
        
        # Convert model to photonic circuit
        circuit = self.accelerator.convert_pytorch_model(model)
        metrics = self.accelerator.compile_and_profile(circuit)
        
        # Simulate inference
        input_np = input_data.numpy() if hasattr(input_data, 'numpy') else input_data
        if len(input_np.shape) > 2:
            # Flatten for photonic processing
            input_np = input_np.reshape(input_np.shape[0], -1)
            
        output, inference_time = self.accelerator.simulate_inference(circuit, input_np)
        
        total_time = time.perf_counter() - start_time
        
        return {
            MetricType.LATENCY.value: inference_time * 1000,  # ms
            MetricType.ENERGY_PER_OP.value: metrics.energy_per_op,  # pJ
            MetricType.THROUGHPUT.value: metrics.throughput,  # GOPS
            MetricType.POWER_CONSUMPTION.value: metrics.power,  # mW
            MetricType.AREA_EFFICIENCY.value: metrics.throughput / metrics.area,  # GOPS/mm²
            "output_shape": output.shape,
            "device": "photonic",
            "total_processing_time": total_time
        }


class QuantumPhotonicBaseline(BaselineAlgorithm):
    """Quantum-enhanced photonic implementation."""
    
    def __init__(self, accelerator, quantum_planner):
        """
        Initialize with quantum-enhanced components.
        
        Args:
            accelerator: PhotonicAccelerator instance
            quantum_planner: QuantumTaskPlanner instance
        """
        self.accelerator = accelerator
        self.quantum_planner = quantum_planner
    
    def name(self) -> str:
        return "Quantum_Photonic_Neural_Network"
    
    def run(self, model: nn.Module, input_data: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Run model with quantum optimization."""
        start_time = time.perf_counter()
        
        # Convert model to photonic circuit
        circuit = self.accelerator.convert_pytorch_model(model)
        
        # Apply quantum optimization
        compilation_tasks = self.quantum_planner.create_circuit_compilation_plan(circuit)
        optimized_tasks = self.quantum_planner.quantum_annealing_optimization(compilation_tasks)
        
        # Multi-objective optimization
        optimization_results = self.quantum_planner.superposition_search(
            circuit, ['energy', 'latency', 'area']
        )
        
        # Simulate optimized inference
        input_np = input_data.numpy() if hasattr(input_data, 'numpy') else input_data
        if len(input_np.shape) > 2:
            input_np = input_np.reshape(input_np.shape[0], -1)
            
        output, inference_time = self.accelerator.simulate_inference(circuit, input_np)
        
        # Apply quantum optimization improvements
        quantum_speedup = optimization_results.get('energy', {}).get('improvement_factor', 1.0)
        optimized_inference_time = inference_time / quantum_speedup
        
        total_time = time.perf_counter() - start_time
        
        # Calculate optimized metrics
        base_metrics = self.accelerator.compile_and_profile(circuit)
        optimized_energy = base_metrics.energy_per_op / quantum_speedup
        
        return {
            MetricType.LATENCY.value: optimized_inference_time * 1000,  # ms
            MetricType.ENERGY_PER_OP.value: optimized_energy,  # pJ
            MetricType.THROUGHPUT.value: base_metrics.throughput * quantum_speedup,  # GOPS
            MetricType.POWER_CONSUMPTION.value: base_metrics.power / quantum_speedup,  # mW
            MetricType.AREA_EFFICIENCY.value: base_metrics.throughput * quantum_speedup / base_metrics.area,
            MetricType.QUANTUM_ADVANTAGE.value: quantum_speedup,
            MetricType.COHERENCE_TIME.value: 100.0 / quantum_speedup,  # μs (estimated)
            MetricType.FIDELITY.value: min(0.99, 0.95 + quantum_speedup * 0.01),
            MetricType.GATE_ERROR_RATE.value: max(0.001, 0.01 / quantum_speedup),
            MetricType.OPTIMIZATION_EFFICIENCY.value: len(optimized_tasks) / total_time if total_time > 0 else 0,
            "quantum_speedup": quantum_speedup,
            "optimization_tasks": len(optimized_tasks),
            "output_shape": output.shape,
            "device": "quantum_photonic",
            "total_processing_time": total_time
        }


class ResearchFramework:
    """Comprehensive research framework for photonic neural networks."""
    
    def __init__(self, output_dir: str = "research_results"):
        """
        Initialize research framework.
        
        Args:
            output_dir: Directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines = {}
        self.experiments = {}
        self.results_cache = {}
        
        logger.info(f"Research framework initialized. Results: {self.output_dir}")
    
    def register_baseline(self, baseline: BaselineAlgorithm):
        """Register a baseline algorithm for comparison."""
        self.baselines[baseline.name()] = baseline
        logger.info(f"Registered baseline: {baseline.name()}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new research experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        self.experiments[config.experiment_id] = config
        
        # Create experiment directory
        exp_dir = self.output_dir / config.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(exp_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        logger.info(f"Created experiment: {config.experiment_id}")
        return config.experiment_id
    
    def run_experiment(self, experiment_id: str, 
                      test_models: List[nn.Module],
                      test_datasets: List[torch.Tensor],
                      parallel: bool = True) -> ExperimentReport:
        """
        Execute a research experiment.
        
        Args:
            experiment_id: ID of experiment to run
            test_models: Models to test
            test_datasets: Test datasets
            parallel: Whether to run in parallel
            
        Returns:
            Comprehensive experiment report
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        config = self.experiments[experiment_id]
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
        
        logger.info(f"Starting experiment: {experiment_id}")
        start_time = time.time()
        
        all_results = []
        
        # Run experiments for each model and dataset combination
        for model_idx, model in enumerate(test_models):
            for dataset_idx, dataset in enumerate(test_datasets):
                
                if parallel and len(self.baselines) > 1:
                    results = self._run_parallel_comparison(
                        config, model, dataset, model_idx, dataset_idx
                    )
                else:
                    results = self._run_sequential_comparison(
                        config, model, dataset, model_idx, dataset_idx
                    )
                
                all_results.extend(results)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(all_results, config)
        
        # Generate conclusions and recommendations
        conclusions = self._generate_conclusions(all_results, statistical_analysis, config)
        recommendations = self._generate_recommendations(all_results, statistical_analysis, config)
        
        # Create experiment report
        report = ExperimentReport(
            config=config,
            results=all_results,
            statistical_analysis=statistical_analysis,
            conclusions=conclusions,
            recommendations=recommendations
        )
        
        # Save results
        self._save_experiment_results(experiment_id, report)
        
        execution_time = time.time() - start_time
        logger.info(f"Experiment {experiment_id} completed in {execution_time:.2f}s. "
                   f"Success rate: {report.success_rate:.2%}")
        
        return report
    
    def _run_parallel_comparison(self, config: ExperimentConfig, model: nn.Module,
                               dataset: torch.Tensor, model_idx: int, 
                               dataset_idx: int) -> List[ExperimentResult]:
        """Run baseline comparisons in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=min(len(self.baselines), mp.cpu_count())) as executor:
            futures = {}
            
            for baseline_name, baseline in self.baselines.items():
                for run_id in range(config.num_runs):
                    future = executor.submit(
                        self._run_single_baseline,
                        baseline, model, dataset, run_id, baseline_name,
                        model_idx, dataset_idx
                    )
                    futures[future] = (baseline_name, run_id)
            
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    baseline_name, run_id = futures[future]
                    logger.error(f"Error in {baseline_name} run {run_id}: {e}")
                    
                    # Create error result
                    error_result = ExperimentResult(
                        run_id=run_id,
                        metrics={},
                        metadata={"baseline": baseline_name, "model_idx": model_idx,
                                "dataset_idx": dataset_idx},
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _run_sequential_comparison(self, config: ExperimentConfig, model: nn.Module,
                                 dataset: torch.Tensor, model_idx: int,
                                 dataset_idx: int) -> List[ExperimentResult]:
        """Run baseline comparisons sequentially."""
        results = []
        
        for baseline_name, baseline in self.baselines.items():
            for run_id in range(config.num_runs):
                try:
                    result = self._run_single_baseline(
                        baseline, model, dataset, run_id, baseline_name,
                        model_idx, dataset_idx
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in {baseline_name} run {run_id}: {e}")
                    
                    error_result = ExperimentResult(
                        run_id=run_id,
                        metrics={},
                        metadata={"baseline": baseline_name, "model_idx": model_idx,
                                "dataset_idx": dataset_idx},
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _run_single_baseline(self, baseline: BaselineAlgorithm, model: nn.Module,
                           dataset: torch.Tensor, run_id: int, baseline_name: str,
                           model_idx: int, dataset_idx: int) -> ExperimentResult:
        """Run a single baseline algorithm."""
        start_time = time.perf_counter()
        
        try:
            metrics = baseline.run(model, dataset)
            execution_time = time.perf_counter() - start_time
            
            result = ExperimentResult(
                run_id=run_id,
                metrics=metrics,
                metadata={
                    "baseline": baseline_name,
                    "model_idx": model_idx,
                    "dataset_idx": dataset_idx,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "dataset_size": dataset.shape[0] if hasattr(dataset, 'shape') else len(dataset)
                },
                execution_time=execution_time,
                success=True
            )
            
            return result
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            
            return ExperimentResult(
                run_id=run_id,
                metrics={},
                metadata={"baseline": baseline_name, "model_idx": model_idx,
                         "dataset_idx": dataset_idx},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _perform_statistical_analysis(self, results: List[ExperimentResult],
                                    config: ExperimentConfig) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        # Group results by baseline
        baseline_results = {}
        for result in results:
            if not result.success:
                continue
                
            baseline = result.metadata.get("baseline", "unknown")
            if baseline not in baseline_results:
                baseline_results[baseline] = []
            baseline_results[baseline].append(result)
        
        # Calculate statistics for each metric
        analysis = {
            "baseline_statistics": {},
            "comparative_analysis": {},
            "significance_tests": {}
        }
        
        # Individual baseline statistics
        for baseline, baseline_res in baseline_results.items():
            metrics_by_type = {}
            
            for result in baseline_res:
                for metric_name, value in result.metrics.items():
                    if metric_name not in metrics_by_type:
                        metrics_by_type[metric_name] = []
                    metrics_by_type[metric_name].append(value)
            
            baseline_stats = {}
            for metric_name, values in metrics_by_type.items():
                if values:
                    baseline_stats[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values),
                        "confidence_interval_95": self._calculate_confidence_interval(values)
                    }
            
            analysis["baseline_statistics"][baseline] = baseline_stats
        
        # Comparative analysis
        if len(baseline_results) >= 2:
            baselines = list(baseline_results.keys())
            
            for i, baseline_a in enumerate(baselines):
                for j, baseline_b in enumerate(baselines[i+1:], i+1):
                    comparison_key = f"{baseline_a}_vs_{baseline_b}"
                    
                    comparison = self._compare_baselines(
                        baseline_results[baseline_a],
                        baseline_results[baseline_b],
                        config.significance_level
                    )
                    
                    analysis["comparative_analysis"][comparison_key] = comparison
        
        return analysis
    
    def _calculate_confidence_interval(self, values: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for values."""
        if len(values) < 2:
            mean_val = np.mean(values) if values else 0.0
            return (mean_val, mean_val)
        
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        
        # Use t-distribution for small samples
        df = len(values) - 1
        t_val = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_val * sem
        
        return (mean - margin, mean + margin)
    
    def _compare_baselines(self, results_a: List[ExperimentResult],
                          results_b: List[ExperimentResult],
                          significance_level: float) -> Dict[str, Any]:
        """Compare two baselines statistically."""
        comparison = {}
        
        # Get common metrics
        metrics_a = set()
        metrics_b = set()
        
        for result in results_a:
            metrics_a.update(result.metrics.keys())
        for result in results_b:
            metrics_b.update(result.metrics.keys())
        
        common_metrics = metrics_a.intersection(metrics_b)
        
        for metric in common_metrics:
            values_a = [r.metrics[metric] for r in results_a if metric in r.metrics]
            values_b = [r.metrics[metric] for r in results_b if metric in r.metrics]
            
            if len(values_a) >= 2 and len(values_b) >= 2:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) +
                                    (len(values_b) - 1) * np.var(values_b, ddof=1)) /
                                   (len(values_a) + len(values_b) - 2))
                
                cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0
                
                # Improvement calculation
                mean_a = np.mean(values_a)
                mean_b = np.mean(values_b)
                
                # For metrics like energy and latency, lower is better
                if metric in [MetricType.ENERGY_PER_OP.value, MetricType.LATENCY.value]:
                    improvement = (mean_a - mean_b) / mean_a * 100 if mean_a > 0 else 0
                else:
                    improvement = (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0
                
                comparison[metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < significance_level,
                    "effect_size_cohens_d": cohens_d,
                    "improvement_percent": improvement,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "sample_size_a": len(values_a),
                    "sample_size_b": len(values_b)
                }
        
        return comparison
    
    def _generate_conclusions(self, results: List[ExperimentResult],
                            analysis: Dict[str, Any],
                            config: ExperimentConfig) -> List[str]:
        """Generate experiment conclusions."""
        conclusions = []
        
        # Success rate analysis
        total_runs = len(results)
        successful_runs = sum(1 for r in results if r.success)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0
        
        conclusions.append(f"Experiment success rate: {success_rate:.1%} ({successful_runs}/{total_runs})")
        
        # Performance comparisons
        comparative_analysis = analysis.get("comparative_analysis", {})
        
        for comparison_name, comparison_data in comparative_analysis.items():
            baseline_a, baseline_b = comparison_name.split("_vs_")
            
            significant_improvements = []
            for metric, metric_data in comparison_data.items():
                if metric_data.get("significant", False) and abs(metric_data.get("improvement_percent", 0)) > 5:
                    improvement = metric_data["improvement_percent"]
                    significant_improvements.append(f"{metric}: {improvement:+.1f}%")
            
            if significant_improvements:
                conclusions.append(
                    f"{baseline_b} shows significant improvements over {baseline_a}: "
                    f"{', '.join(significant_improvements)}"
                )
        
        # Hypothesis validation
        hypothesis_validated = self._validate_hypothesis(results, analysis, config)
        if hypothesis_validated:
            conclusions.append(f"✓ Hypothesis validated: {config.hypothesis}")
        else:
            conclusions.append(f"✗ Hypothesis not validated: {config.hypothesis}")
        
        return conclusions
    
    def _generate_recommendations(self, results: List[ExperimentResult],
                                analysis: Dict[str, Any],
                                config: ExperimentConfig) -> List[str]:
        """Generate research recommendations."""
        recommendations = []
        
        # Performance recommendations
        baseline_stats = analysis.get("baseline_statistics", {})
        
        if baseline_stats:
            # Find best performing baseline for key metrics
            key_metrics = [MetricType.ENERGY_PER_OP.value, MetricType.LATENCY.value, 
                          MetricType.THROUGHPUT.value]
            
            for metric in key_metrics:
                best_baseline = None
                best_value = None
                
                for baseline, stats in baseline_stats.items():
                    if metric in stats:
                        value = stats[metric]["mean"]
                        
                        # Lower is better for energy and latency
                        if metric in [MetricType.ENERGY_PER_OP.value, MetricType.LATENCY.value]:
                            if best_value is None or value < best_value:
                                best_value = value
                                best_baseline = baseline
                        else:
                            if best_value is None or value > best_value:
                                best_value = value
                                best_baseline = baseline
                
                if best_baseline:
                    recommendations.append(f"For {metric} optimization, prefer {best_baseline}")
        
        # Statistical recommendations
        comparative_analysis = analysis.get("comparative_analysis", {})
        
        low_significance_comparisons = []
        for comparison_name, comparison_data in comparative_analysis.items():
            insignificant_metrics = [
                metric for metric, data in comparison_data.items()
                if not data.get("significant", False)
            ]
            if len(insignificant_metrics) > len(comparison_data) / 2:
                low_significance_comparisons.append(comparison_name)
        
        if low_significance_comparisons:
            recommendations.append(
                "Consider increasing sample size for more statistically significant results in: "
                f"{', '.join(low_significance_comparisons)}"
            )
        
        # Experimental design recommendations
        if config.num_runs < 30:
            recommendations.append("Consider increasing number of runs (≥30) for better statistical power")
        
        return recommendations
    
    def _validate_hypothesis(self, results: List[ExperimentResult],
                           analysis: Dict[str, Any],
                           config: ExperimentConfig) -> bool:
        """Validate experimental hypothesis."""
        # Simple validation based on success criteria
        success_criteria = config.success_criteria
        
        if not success_criteria:
            return False
        
        # Check if success criteria are met
        comparative_analysis = analysis.get("comparative_analysis", {})
        
        for criterion_name, criterion_value in success_criteria.items():
            if isinstance(criterion_value, dict):
                # Complex criterion (e.g., improvement thresholds)
                comparison_key = criterion_value.get("comparison")
                metric = criterion_value.get("metric")
                threshold = criterion_value.get("threshold", 0)
                
                if comparison_key and metric and comparison_key in comparative_analysis:
                    comparison_data = comparative_analysis[comparison_key]
                    if metric in comparison_data:
                        improvement = comparison_data[metric].get("improvement_percent", 0)
                        if improvement < threshold:
                            return False
        
        return True
    
    def _save_experiment_results(self, experiment_id: str, report: ExperimentReport):
        """Save experiment results to files."""
        exp_dir = self.output_dir / experiment_id
        
        # Save detailed report as JSON
        with open(exp_dir / "report.json", "w") as f:
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save report as pickle for full Python object
        with open(exp_dir / "report.pkl", "wb") as f:
            pickle.dump(report, f)
        
        # Create summary CSV
        self._create_results_csv(exp_dir, report)
        
        # Generate plots
        self._generate_plots(exp_dir, report)
        
        logger.info(f"Saved experiment results to {exp_dir}")
    
    def _create_results_csv(self, exp_dir: Path, report: ExperimentReport):
        """Create CSV summary of results."""
        rows = []
        
        for result in report.results:
            if not result.success:
                continue
                
            row = {
                "run_id": result.run_id,
                "baseline": result.metadata.get("baseline", "unknown"),
                "model_idx": result.metadata.get("model_idx", 0),
                "dataset_idx": result.metadata.get("dataset_idx", 0),
                "execution_time": result.execution_time,
                "success": result.success
            }
            
            # Add metrics
            row.update(result.metrics)
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(exp_dir / "results.csv", index=False)
    
    def _generate_plots(self, exp_dir: Path, report: ExperimentReport):
        """Generate visualization plots."""
        if not report.results:
            return
        
        # Group results by baseline
        baseline_results = {}
        for result in report.results:
            if not result.success:
                continue
                
            baseline = result.metadata.get("baseline", "unknown")
            if baseline not in baseline_results:
                baseline_results[baseline] = []
            baseline_results[baseline].append(result)
        
        # Create comparison plots for key metrics
        key_metrics = [MetricType.ENERGY_PER_OP.value, MetricType.LATENCY.value,
                      MetricType.THROUGHPUT.value]
        
        for metric in key_metrics:
            plt.figure(figsize=(10, 6))
            
            baseline_names = []
            metric_values = []
            
            for baseline, results in baseline_results.items():
                values = [r.metrics.get(metric, 0) for r in results if metric in r.metrics]
                if values:
                    baseline_names.append(baseline)
                    metric_values.append(values)
            
            if baseline_names and metric_values:
                plt.boxplot(metric_values, labels=baseline_names)
                plt.title(f"{metric} Comparison")
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(exp_dir / f"{metric}_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Generated plots in {exp_dir}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            "total_experiments": len(self.experiments),
            "registered_baselines": list(self.baselines.keys()),
            "output_directory": str(self.output_dir),
            "experiments": {
                exp_id: {
                    "type": config.experiment_type.value,
                    "description": config.description,
                    "num_runs": config.num_runs
                }
                for exp_id, config in self.experiments.items()
            }
        }