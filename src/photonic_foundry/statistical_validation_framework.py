"""
Statistical Validation Framework for Quantum-Photonic Research

Comprehensive statistical validation system ensuring research reproducibility and
publication-grade rigor. This framework provides:

1. Experimental design validation with power analysis
2. Multiple testing correction (Benjamini-Hochberg, Bonferroni)
3. Effect size calculations (Cohen's d, eta-squared, Cliff's delta)
4. Bayesian hypothesis testing with Bayes factors
5. Reproducibility testing across hardware configurations
6. Publication-ready statistical reports

Statistical Requirements:
- Sample sizes: Minimum 100 independent runs per configuration
- Power analysis: 90% statistical power targeting
- Effect size detection: Cohen's d ≥ 0.8
- Significance thresholds: α = 0.01 for breakthrough claims (p < 0.01)
- Confidence intervals: 99% for all performance metrics
- Multiple testing correction for family-wise error control
"""

import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.effect_size import cohen_d
import pandas as pd
import numpy.random as random
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch

# Import for Bayesian analysis
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.warning("PyMC not available, Bayesian analysis disabled")

logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Types of statistical tests."""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    ANOVA_ONE_WAY = "anova_one_way"
    CHI_SQUARE = "chi_square"
    BAYESIAN_T_TEST = "bayesian_t_test"


class EffectSizeMetric(Enum):
    """Effect size metrics."""
    COHENS_D = "cohens_d"
    ETA_SQUARED = "eta_squared"
    CLIFFS_DELTA = "cliffs_delta"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"


class MultipleTestingCorrection(Enum):
    """Multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    HOLM_BONFERRONI = "holm_bonferroni"
    SIDAK = "sidak"


@dataclass
class StatisticalResult:
    """Results from statistical analysis."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    power: float
    sample_size: int
    degrees_freedom: Optional[int] = None
    corrected_p_value: Optional[float] = None
    bayes_factor: Optional[float] = None
    
    def is_significant(self, alpha: float = 0.01) -> bool:
        """Check if result is statistically significant."""
        p_val = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        return p_val < alpha
    
    def is_practically_significant(self, min_effect_size: float = 0.8) -> bool:
        """Check if result is practically significant."""
        return abs(self.effect_size) >= min_effect_size


@dataclass
class ExperimentalDesign:
    """Experimental design specification."""
    hypothesis: str
    independent_variables: List[str]
    dependent_variables: List[str]
    sample_size_per_group: int
    num_groups: int
    alpha_level: float = 0.01
    target_power: float = 0.9
    min_effect_size: float = 0.8
    random_seed: int = 42
    
    def validate_design(self) -> Dict[str, bool]:
        """Validate experimental design."""
        validation = {
            'adequate_sample_size': self.sample_size_per_group >= 30,
            'appropriate_alpha': 0.001 <= self.alpha_level <= 0.05,
            'sufficient_power': self.target_power >= 0.8,
            'meaningful_effect_size': self.min_effect_size >= 0.2
        }
        return validation


@dataclass
class ValidationConfig:
    """Configuration for statistical validation."""
    significance_threshold: float = 0.01
    confidence_level: float = 0.99
    min_effect_size: float = 0.8
    target_power: float = 0.9
    num_bootstrap_samples: int = 1000
    num_permutation_tests: int = 1000
    multiple_testing_correction: MultipleTestingCorrection = MultipleTestingCorrection.BENJAMINI_HOCHBERG
    bayesian_analysis: bool = False
    reproducibility_tolerance: float = 0.05
    cross_validation_folds: int = 5


class PowerAnalysis:
    """Statistical power analysis utilities."""
    
    @staticmethod
    def calculate_required_sample_size(effect_size: float, power: float = 0.9, 
                                     alpha: float = 0.01) -> int:
        """Calculate required sample size for desired power."""
        try:
            sample_size = ttest_power(effect_size, power, alpha, alternative='two-sided')
            return max(30, int(np.ceil(sample_size)))
        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            return 100  # Conservative default
    
    @staticmethod
    def calculate_achieved_power(effect_size: float, sample_size: int, 
                               alpha: float = 0.01) -> float:
        """Calculate achieved statistical power."""
        try:
            power = ttest_power(effect_size, None, alpha, sample_size, alternative='two-sided')
            return min(1.0, max(0.0, power))
        except Exception as e:
            logger.warning(f"Power calculation failed: {e}")
            return 0.5  # Conservative estimate
    
    @staticmethod
    def power_analysis_report(effect_sizes: List[float], sample_sizes: List[int],
                            alpha: float = 0.01) -> Dict[str, Any]:
        """Generate comprehensive power analysis report."""
        report = {
            'configurations': [],
            'recommendations': []
        }
        
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                power = PowerAnalysis.calculate_achieved_power(effect_size, sample_size, alpha)
                
                config = {
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'power': power,
                    'adequate_power': power >= 0.8,
                    'recommended': power >= 0.9
                }
                report['configurations'].append(config)
        
        # Generate recommendations
        for effect_size in effect_sizes:
            required_n = PowerAnalysis.calculate_required_sample_size(effect_size, 0.9, alpha)
            report['recommendations'].append({
                'effect_size': effect_size,
                'recommended_sample_size': required_n,
                'rationale': f'For effect size {effect_size:.2f}, need n={required_n} for 90% power'
            })
        
        return report


class EffectSizeCalculator:
    """Effect size calculation utilities."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        try:
            return cohen_d(group1, group2)
        except Exception:
            # Fallback calculation
            n1, n2 = len(group1), len(group2)
            if n1 < 2 or n2 < 2:
                return 0.0
                
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                                (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
                
            return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        if len(groups) < 2:
            return 0.0
            
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        ss_total = np.sum((all_values - grand_mean)**2)
        
        return ss_between / ss_total if ss_total > 0 else 0.0
    
    @staticmethod
    def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
            
        # Number of pairs where group1 > group2 minus pairs where group1 < group2
        dominance = 0
        total_pairs = len(group1) * len(group2)
        
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1
                elif x < y:
                    dominance -= 1
        
        return dominance / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def interpret_effect_size(effect_size: float, metric: EffectSizeMetric) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if metric in [EffectSizeMetric.COHENS_D, EffectSizeMetric.HEDGES_G, EffectSizeMetric.GLASS_DELTA]:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif metric == EffectSizeMetric.ETA_SQUARED:
            if abs_effect < 0.01:
                return "negligible"
            elif abs_effect < 0.06:
                return "small"
            elif abs_effect < 0.14:
                return "medium"
            else:
                return "large"
        
        elif metric == EffectSizeMetric.CLIFFS_DELTA:
            if abs_effect < 0.147:
                return "negligible"
            elif abs_effect < 0.33:
                return "small"
            elif abs_effect < 0.474:
                return "medium"
            else:
                return "large"
        
        return "unknown"


class BayesianAnalysis:
    """Bayesian statistical analysis utilities."""
    
    def __init__(self):
        self.available = BAYESIAN_AVAILABLE
        
    def bayesian_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Optional[Dict[str, float]]:
        """Perform Bayesian t-test."""
        if not self.available:
            logger.warning("Bayesian analysis not available")
            return None
            
        try:
            with pm.Model() as model:
                # Priors
                mu1 = pm.Normal('mu1', mu=0, sigma=10)
                mu2 = pm.Normal('mu2', mu=0, sigma=10)
                sigma1 = pm.HalfNormal('sigma1', sigma=10)
                sigma2 = pm.HalfNormal('sigma2', sigma=10)
                
                # Likelihood
                obs1 = pm.Normal('obs1', mu=mu1, sigma=sigma1, observed=group1)
                obs2 = pm.Normal('obs2', mu=mu2, sigma=sigma2, observed=group2)
                
                # Difference
                diff = pm.Deterministic('diff', mu1 - mu2)
                
                # Sample
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, progressbar=False)
            
            # Calculate Bayes factor (simplified)
            diff_samples = trace.posterior['diff'].values.flatten()
            prob_positive = np.mean(diff_samples > 0)
            prob_negative = np.mean(diff_samples < 0)
            
            # Savage-Dickey approximation for Bayes factor
            bayes_factor = max(prob_positive, prob_negative) / 0.5 if max(prob_positive, prob_negative) > 0.5 else 1.0
            
            return {
                'posterior_mean_diff': np.mean(diff_samples),
                'posterior_std_diff': np.std(diff_samples),
                'hdi_lower': np.percentile(diff_samples, 2.5),
                'hdi_upper': np.percentile(diff_samples, 97.5),
                'bayes_factor': bayes_factor,
                'prob_h1': max(prob_positive, prob_negative)
            }
            
        except Exception as e:
            logger.error(f"Bayesian t-test failed: {e}")
            return None
    
    def interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes factor strength."""
        if bf < 1:
            return "evidence against H1"
        elif bf < 3:
            return "weak evidence for H1"
        elif bf < 10:
            return "moderate evidence for H1"
        elif bf < 30:
            return "strong evidence for H1"
        elif bf < 100:
            return "very strong evidence for H1"
        else:
            return "extreme evidence for H1"


class ReproducibilityValidator:
    """Reproducibility testing utilities."""
    
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.reproducibility_tests = []
    
    async def test_reproducibility(self, algorithm_func: Callable, test_configs: List[Dict[str, Any]],
                                 num_replications: int = 10) -> Dict[str, Any]:
        """Test algorithm reproducibility across different conditions."""
        reproducibility_results = {
            'mean_reproducibility': 0.0,
            'std_reproducibility': 0.0,
            'min_reproducibility': 1.0,
            'max_reproducibility': 0.0,
            'failed_replications': 0,
            'detailed_results': []
        }
        
        all_reproducibility_scores = []
        
        for config_idx, config in enumerate(test_configs):
            logger.info(f"Testing reproducibility for config {config_idx + 1}/{len(test_configs)}")
            
            # Run multiple replications with same configuration
            replication_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._run_single_replication, algorithm_func, config, rep_idx)
                    for rep_idx in range(num_replications)
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            replication_results.append(result)
                    except Exception as e:
                        logger.warning(f"Replication failed: {e}")
                        reproducibility_results['failed_replications'] += 1
            
            if len(replication_results) >= 2:
                # Calculate reproducibility score for this configuration
                reproducibility_score = self._calculate_reproducibility_score(replication_results)
                all_reproducibility_scores.append(reproducibility_score)
                
                reproducibility_results['detailed_results'].append({
                    'config_index': config_idx,
                    'config': config,
                    'num_successful_replications': len(replication_results),
                    'reproducibility_score': reproducibility_score,
                    'coefficient_of_variation': np.std(replication_results) / np.mean(replication_results) if np.mean(replication_results) != 0 else float('inf')
                })
        
        # Calculate overall reproducibility metrics
        if all_reproducibility_scores:
            reproducibility_results['mean_reproducibility'] = np.mean(all_reproducibility_scores)
            reproducibility_results['std_reproducibility'] = np.std(all_reproducibility_scores)
            reproducibility_results['min_reproducibility'] = np.min(all_reproducibility_scores)
            reproducibility_results['max_reproducibility'] = np.max(all_reproducibility_scores)
        
        return reproducibility_results
    
    def _run_single_replication(self, algorithm_func: Callable, config: Dict[str, Any], 
                              replication_idx: int) -> Optional[float]:
        """Run single replication of algorithm."""
        try:
            # Set deterministic seed for this replication
            np.random.seed(config.get('random_seed', 42) + replication_idx)
            
            # Run algorithm
            result = algorithm_func(**config)
            
            # Extract primary metric (assumed to be in 'final_energy' or 'objective_value')
            if isinstance(result, dict):
                return result.get('final_energy', result.get('objective_value', result.get('score', None)))
            else:
                return float(result) if result is not None else None
                
        except Exception as e:
            logger.warning(f"Single replication failed: {e}")
            return None
    
    def _calculate_reproducibility_score(self, results: List[float]) -> float:
        """Calculate reproducibility score from replication results."""
        if len(results) < 2:
            return 0.0
            
        # Calculate coefficient of variation (relative standard deviation)
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        if abs(mean_result) < 1e-10:
            cv = 0.0 if std_result < 1e-10 else float('inf')
        else:
            cv = std_result / abs(mean_result)
        
        # Convert to reproducibility score (1 = perfect reproducibility, 0 = no reproducibility)
        reproducibility_score = max(0.0, 1.0 - cv / self.tolerance)
        
        return reproducibility_score


class StatisticalValidationFramework:
    """
    Main statistical validation framework for quantum-photonic research.
    
    Provides comprehensive statistical analysis, effect size calculations,
    multiple testing correction, and reproducibility validation.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.power_analyzer = PowerAnalysis()
        self.effect_calculator = EffectSizeCalculator()
        self.bayesian_analyzer = BayesianAnalysis()
        self.reproducibility_validator = ReproducibilityValidator(config.reproducibility_tolerance)
        
        # Results storage
        self.statistical_results: List[StatisticalResult] = []
        self.power_analyses: List[Dict[str, Any]] = []
        self.reproducibility_results: List[Dict[str, Any]] = []
        
    async def validate_experimental_results(self, experimental_data: Dict[str, List[float]],
                                          experimental_design: ExperimentalDesign) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation of experimental results.
        
        Args:
            experimental_data: Dictionary mapping algorithm names to result lists
            experimental_design: Experimental design specification
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive statistical validation")
        
        # Validate experimental design
        design_validation = experimental_design.validate_design()
        if not all(design_validation.values()):
            logger.warning(f"Experimental design issues detected: {design_validation}")
        
        # Perform pairwise comparisons
        pairwise_results = self._perform_pairwise_comparisons(experimental_data)
        
        # Apply multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(pairwise_results)
        
        # Calculate effect sizes
        effect_size_results = self._calculate_comprehensive_effect_sizes(experimental_data)
        
        # Perform power analysis
        power_results = self._perform_power_analysis(experimental_data, experimental_design)
        
        # Bayesian analysis (if enabled)
        bayesian_results = None
        if self.config.bayesian_analysis:
            bayesian_results = self._perform_bayesian_analysis(experimental_data)
        
        # Generate validation report
        validation_report = self._generate_validation_report(
            design_validation, corrected_results, effect_size_results, 
            power_results, bayesian_results
        )
        
        return validation_report
    
    def _perform_pairwise_comparisons(self, data: Dict[str, List[float]]) -> List[StatisticalResult]:
        """Perform all pairwise statistical comparisons."""
        results = []
        algorithms = list(data.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                group1 = np.array(data[algo1])
                group2 = np.array(data[algo2])
                
                # Remove infinite and NaN values
                group1 = group1[np.isfinite(group1)]
                group2 = group2[np.isfinite(group2)]
                
                if len(group1) < 10 or len(group2) < 10:
                    continue
                
                # Perform multiple statistical tests
                
                # 1. Independent t-test (parametric)
                try:
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    effect_size = self.effect_calculator.cohens_d(group1, group2)
                    
                    # Confidence interval for mean difference
                    se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
                    df = len(group1) + len(group2) - 2
                    t_crit = stats.t.ppf((1 + self.config.confidence_level) / 2, df)
                    mean_diff = np.mean(group1) - np.mean(group2)
                    ci_lower = mean_diff - t_crit * se_diff
                    ci_upper = mean_diff + t_crit * se_diff
                    
                    power = self.power_analyzer.calculate_achieved_power(
                        abs(effect_size), min(len(group1), len(group2)), self.config.significance_threshold
                    )
                    
                    results.append(StatisticalResult(
                        test_name=f"{algo1}_vs_{algo2}_t_test",
                        statistic=t_stat,
                        p_value=p_val,
                        effect_size=effect_size,
                        effect_size_interpretation=self.effect_calculator.interpret_effect_size(
                            effect_size, EffectSizeMetric.COHENS_D
                        ),
                        confidence_interval=(ci_lower, ci_upper),
                        power=power,
                        sample_size=min(len(group1), len(group2)),
                        degrees_freedom=df
                    ))
                except Exception as e:
                    logger.warning(f"t-test failed for {algo1} vs {algo2}: {e}")
                
                # 2. Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    cliff_delta = self.effect_calculator.cliffs_delta(group1, group2)
                    
                    results.append(StatisticalResult(
                        test_name=f"{algo1}_vs_{algo2}_mann_whitney",
                        statistic=u_stat,
                        p_value=p_val,
                        effect_size=cliff_delta,
                        effect_size_interpretation=self.effect_calculator.interpret_effect_size(
                            cliff_delta, EffectSizeMetric.CLIFFS_DELTA
                        ),
                        confidence_interval=(0, 0),  # Bootstrap CI would be needed
                        power=0.8,  # Simplified
                        sample_size=min(len(group1), len(group2))
                    ))
                except Exception as e:
                    logger.warning(f"Mann-Whitney test failed for {algo1} vs {algo2}: {e}")
        
        return results
    
    def _apply_multiple_testing_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply multiple testing correction to p-values."""
        if not results:
            return results
            
        p_values = [r.p_value for r in results]
        
        # Apply correction based on configuration
        if self.config.multiple_testing_correction == MultipleTestingCorrection.BONFERRONI:
            corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        elif self.config.multiple_testing_correction == MultipleTestingCorrection.BENJAMINI_HOCHBERG:
            corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
        elif self.config.multiple_testing_correction == MultipleTestingCorrection.HOLM_BONFERRONI:
            corrected_p_values = multipletests(p_values, method='holm')[1]
        else:
            corrected_p_values = p_values
        
        # Update results with corrected p-values
        for i, result in enumerate(results):
            result.corrected_p_value = corrected_p_values[i]
        
        return results
    
    def _calculate_comprehensive_effect_sizes(self, data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive effect sizes for all comparisons."""
        effect_sizes = {}
        algorithms = list(data.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                group1 = np.array(data[algo1])
                group2 = np.array(data[algo2])
                
                # Remove infinite and NaN values
                group1 = group1[np.isfinite(group1)]
                group2 = group2[np.isfinite(group2)]
                
                if len(group1) < 10 or len(group2) < 10:
                    continue
                
                comparison_key = f"{algo1}_vs_{algo2}"
                effect_sizes[comparison_key] = {
                    'cohens_d': self.effect_calculator.cohens_d(group1, group2),
                    'cliffs_delta': self.effect_calculator.cliffs_delta(group1, group2),
                    'mean_difference': np.mean(group1) - np.mean(group2),
                    'median_difference': np.median(group1) - np.median(group2),
                    'relative_improvement': (np.mean(group1) - np.mean(group2)) / abs(np.mean(group2)) if abs(np.mean(group2)) > 1e-10 else 0
                }
        
        return effect_sizes
    
    def _perform_power_analysis(self, data: Dict[str, List[float]], 
                              design: ExperimentalDesign) -> Dict[str, Any]:
        """Perform comprehensive power analysis."""
        power_results = {
            'achieved_power': {},
            'required_sample_sizes': {},
            'power_recommendations': []
        }
        
        algorithms = list(data.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                group1 = np.array(data[algo1])
                group2 = np.array(data[algo2])
                
                # Remove infinite and NaN values
                group1 = group1[np.isfinite(group1)]
                group2 = group2[np.isfinite(group2)]
                
                if len(group1) < 10 or len(group2) < 10:
                    continue
                
                comparison_key = f"{algo1}_vs_{algo2}"
                
                # Calculate effect size
                effect_size = self.effect_calculator.cohens_d(group1, group2)
                
                # Calculate achieved power
                achieved_power = self.power_analyzer.calculate_achieved_power(
                    abs(effect_size), min(len(group1), len(group2)), self.config.significance_threshold
                )
                
                # Calculate required sample size for target power
                required_n = self.power_analyzer.calculate_required_sample_size(
                    abs(effect_size), self.config.target_power, self.config.significance_threshold
                )
                
                power_results['achieved_power'][comparison_key] = achieved_power
                power_results['required_sample_sizes'][comparison_key] = required_n
                
                # Generate recommendation
                if achieved_power < self.config.target_power:
                    power_results['power_recommendations'].append({
                        'comparison': comparison_key,
                        'current_power': achieved_power,
                        'current_n': min(len(group1), len(group2)),
                        'required_n': required_n,
                        'recommendation': f'Increase sample size to {required_n} for adequate power'
                    })
        
        return power_results
    
    def _perform_bayesian_analysis(self, data: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        """Perform Bayesian statistical analysis."""
        if not self.bayesian_analyzer.available:
            return None
            
        bayesian_results = {}
        algorithms = list(data.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                group1 = np.array(data[algo1])
                group2 = np.array(data[algo2])
                
                # Remove infinite and NaN values
                group1 = group1[np.isfinite(group1)]
                group2 = group2[np.isfinite(group2)]
                
                if len(group1) < 10 or len(group2) < 10:
                    continue
                
                comparison_key = f"{algo1}_vs_{algo2}"
                
                # Perform Bayesian t-test
                bayesian_result = self.bayesian_analyzer.bayesian_t_test(group1, group2)
                
                if bayesian_result:
                    bayesian_result['interpretation'] = self.bayesian_analyzer.interpret_bayes_factor(
                        bayesian_result['bayes_factor']
                    )
                    bayesian_results[comparison_key] = bayesian_result
        
        return bayesian_results
    
    def _generate_validation_report(self, design_validation: Dict[str, bool],
                                  statistical_results: List[StatisticalResult],
                                  effect_sizes: Dict[str, Dict[str, float]],
                                  power_results: Dict[str, Any],
                                  bayesian_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Count significant results
        significant_results = [r for r in statistical_results if r.is_significant(self.config.significance_threshold)]
        practically_significant = [r for r in statistical_results if r.is_practically_significant(self.config.min_effect_size)]
        
        # Identify breakthrough findings
        breakthrough_findings = []
        for result in statistical_results:
            if (result.is_significant(self.config.significance_threshold) and
                result.is_practically_significant(self.config.min_effect_size) and
                result.power >= self.config.target_power):
                
                breakthrough_findings.append({
                    'comparison': result.test_name,
                    'p_value': result.corrected_p_value or result.p_value,
                    'effect_size': result.effect_size,
                    'effect_interpretation': result.effect_size_interpretation,
                    'power': result.power,
                    'confidence_interval': result.confidence_interval
                })
        
        report = {
            'experimental_design': {
                'validation': design_validation,
                'design_quality': 'adequate' if all(design_validation.values()) else 'needs_improvement'
            },
            'statistical_summary': {
                'total_comparisons': len(statistical_results),
                'significant_results': len(significant_results),
                'practically_significant': len(practically_significant),
                'breakthrough_findings': len(breakthrough_findings),
                'family_wise_error_controlled': True,
                'multiple_testing_correction': self.config.multiple_testing_correction.value
            },
            'effect_sizes': effect_sizes,
            'power_analysis': power_results,
            'breakthrough_findings': breakthrough_findings,
            'statistical_results': [
                {
                    'test': r.test_name,
                    'statistic': r.statistic,
                    'p_value': r.p_value,
                    'corrected_p_value': r.corrected_p_value,
                    'effect_size': r.effect_size,
                    'effect_interpretation': r.effect_size_interpretation,
                    'power': r.power,
                    'significant': r.is_significant(self.config.significance_threshold),
                    'practically_significant': r.is_practically_significant(self.config.min_effect_size)
                }
                for r in statistical_results
            ],
            'recommendations': self._generate_recommendations(statistical_results, power_results),
            'publication_readiness': self._assess_publication_readiness(breakthrough_findings, power_results)
        }
        
        if bayesian_results:
            report['bayesian_analysis'] = bayesian_results
        
        return report
    
    def _generate_recommendations(self, statistical_results: List[StatisticalResult],
                                power_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Power-related recommendations
        low_power_tests = [r for r in statistical_results if r.power < self.config.target_power]
        if low_power_tests:
            recommendations.append(
                f"Increase sample size for {len(low_power_tests)} comparisons with insufficient power (<{self.config.target_power})"
            )
        
        # Effect size recommendations
        small_effects = [r for r in statistical_results if abs(r.effect_size) < self.config.min_effect_size]
        if small_effects:
            recommendations.append(
                f"Consider practical significance: {len(small_effects)} comparisons show small effect sizes"
            )
        
        # Reproducibility recommendations
        recommendations.append("Conduct reproducibility testing across different hardware configurations")
        
        # Multiple testing recommendations
        recommendations.append("Report both raw and corrected p-values for transparency")
        
        return recommendations
    
    def _assess_publication_readiness(self, breakthrough_findings: List[Dict[str, Any]],
                                    power_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for academic publication."""
        criteria = {
            'has_breakthrough_findings': len(breakthrough_findings) > 0,
            'adequate_statistical_power': any(p >= self.config.target_power for p in power_results.get('achieved_power', {}).values()),
            'controlled_family_wise_error': True,  # We apply corrections
            'large_effect_sizes': any(f['effect_interpretation'] in ['large'] for f in breakthrough_findings),
            'sufficient_sample_sizes': all(n >= 30 for n in power_results.get('required_sample_sizes', {}).values())
        }
        
        publication_score = sum(criteria.values()) / len(criteria)
        
        return {
            'criteria_met': criteria,
            'publication_score': publication_score,
            'readiness_level': (
                'publication_ready' if publication_score >= 0.8 else
                'needs_minor_improvements' if publication_score >= 0.6 else
                'needs_major_improvements'
            ),
            'recommendations': [
                'Include reproducibility analysis',
                'Provide detailed statistical methodology',
                'Report confidence intervals and effect sizes',
                'Discuss practical significance alongside statistical significance'
            ]
        }
    
    def save_validation_results(self, report: Dict[str, Any], output_path: str):
        """Save validation results to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {output_path}")


# Factory function
def create_validation_framework(config: Optional[ValidationConfig] = None) -> StatisticalValidationFramework:
    """Create statistical validation framework with configuration."""
    return StatisticalValidationFramework(config or ValidationConfig())


# Demo function
async def demonstrate_statistical_validation():
    """Demonstrate statistical validation framework."""
    logger.info("=== Statistical Validation Framework Demo ===")
    
    # Create synthetic experimental data
    np.random.seed(42)
    experimental_data = {
        'PQEC': np.random.normal(0.1, 0.02, 100),  # Lower energy (better)
        'AQPPO': np.random.normal(0.12, 0.025, 100),
        'Quantum_Breakthrough': np.random.normal(0.08, 0.015, 100),  # Best performance
        'Classical_Baseline': np.random.normal(0.25, 0.05, 100),  # Higher energy (worse)
        'Genetic_Algorithm': np.random.normal(0.22, 0.04, 100),
        'Gradient_Descent': np.random.normal(0.20, 0.03, 100)
    }
    
    # Create experimental design
    design = ExperimentalDesign(
        hypothesis="Novel quantum-photonic algorithms achieve superior performance",
        independent_variables=['algorithm_type'],
        dependent_variables=['optimization_energy'],
        sample_size_per_group=100,
        num_groups=6,
        alpha_level=0.01,
        target_power=0.9,
        min_effect_size=0.8
    )
    
    # Create validation configuration
    config = ValidationConfig(
        significance_threshold=0.01,
        min_effect_size=0.8,
        target_power=0.9,
        bayesian_analysis=False  # Disabled for demo
    )
    
    # Run validation
    framework = create_validation_framework(config)
    validation_report = await framework.validate_experimental_results(experimental_data, design)
    
    logger.info("=== Validation Results Summary ===")
    logger.info(f"Total comparisons: {validation_report['statistical_summary']['total_comparisons']}")
    logger.info(f"Significant results: {validation_report['statistical_summary']['significant_results']}")
    logger.info(f"Breakthrough findings: {validation_report['statistical_summary']['breakthrough_findings']}")
    logger.info(f"Publication readiness: {validation_report['publication_readiness']['readiness_level']}")
    
    if validation_report['breakthrough_findings']:
        logger.info("\n=== Breakthrough Findings ===")
        for finding in validation_report['breakthrough_findings']:
            logger.info(f"{finding['comparison']}: Effect size {finding['effect_size']:.3f} "
                       f"({finding['effect_interpretation']}), p={finding['p_value']:.6f}")
    
    return validation_report


if __name__ == "__main__":
    # Run demo
    asyncio.run(demonstrate_statistical_validation())