"""
A/B Testing Framework for Model Preservation System

This module provides comprehensive A/B testing capabilities for machine learning models,
including statistical analysis, traffic routing, and performance evaluation.

Features:
- Model A/B testing with configurable traffic splits
- Statistical significance testing
- Bayesian analysis for continuous monitoring
- Multi-armed bandit algorithms for dynamic allocation
- Canary deployments and feature flag integration
"""

import asyncio
import hashlib
import math
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
from scipy import stats
from scipy.stats import beta, norm

from .base import PreservationError, ModelMetadata
from .versioning import SemanticVersion


class ABTestStatus(Enum):
    """A/B test status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class ABTestVariant(Enum):
    """A/B test variant enumeration"""
    CONTROL = "control"
    TREATMENT = "treatment"
    MODEL_A = "model_a"
    MODEL_B = "model_b"


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    model_a_type: str
    model_a_version: str
    model_b_type: str
    model_b_version: str
    traffic_split: float = 0.5
    duration_days: int = 14
    min_samples: int = 1000
    significance_level: float = 0.05
    power: float = 0.8
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "inference_time"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0 < self.traffic_split < 1:
            raise ValueError(f"Traffic split must be between 0 and 1, got {self.traffic_split}")
        if self.duration_days <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration_days}")
        if self.min_samples <= 0:
            raise ValueError(f"Min samples must be positive, got {self.min_samples}")
        if not 0 < self.significance_level < 1:
            raise ValueError(f"Significance level must be between 0 and 1, got {self.significance_level}")
        if not 0 < self.power < 1:
            raise ValueError(f"Power must be between 0 and 1, got {self.power}")


@dataclass
class ABTestResult:
    """Results from A/B test analysis"""
    test_name: str
    metric: str
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: float
    sample_size_control: int
    sample_size_treatment: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestMetric:
    """Individual metric record for A/B testing"""
    test_name: str
    variant: str
    user_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class ABTestStatistics:
    """Statistical analysis components for A/B testing"""
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Expected baseline conversion rate or metric value
            minimum_detectable_effect: Minimum effect size to detect
            alpha: Type I error rate (significance level)
            power: Statistical power (1 - Type II error rate)
            
        Returns:
            Required sample size per variant
        """
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # For proportions (conversion rates)
        if 0 < baseline_rate < 1:
            p1 = baseline_rate
            p2 = baseline_rate + minimum_detectable_effect
            p_pooled = (p1 + p2) / 2
            
            sample_size = (
                2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2
            ) / (minimum_detectable_effect**2)
        else:
            # For continuous metrics (assume standard deviation equals mean for simplicity)
            # In practice, you'd use historical standard deviation
            std_dev = baseline_rate * 0.2  # Assume 20% coefficient of variation
            
            sample_size = (
                2 * std_dev**2 * (z_alpha + z_beta)**2
            ) / (minimum_detectable_effect**2)
        
        return int(math.ceil(sample_size))
    
    def calculate_significance(
        self,
        control_data: List[float],
        treatment_data: List[float],
        test_type: str = "two_tailed"
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance for A/B test
        
        Args:
            control_data: Control group measurements
            treatment_data: Treatment group measurements
            test_type: Type of test ("two_tailed", "one_tailed_greater", "one_tailed_less")
            
        Returns:
            Dictionary with statistical test results
        """
        if len(control_data) < 2 or len(treatment_data) < 2:
            raise ValueError("Need at least 2 samples per group for statistical testing")
        
        control_mean = statistics.mean(control_data)
        treatment_mean = statistics.mean(treatment_data)
        effect_size = treatment_mean - control_mean
        relative_effect = effect_size / control_mean if control_mean != 0 else 0
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        
        # Adjust p-value for one-tailed tests
        if test_type == "one_tailed_greater":
            p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        elif test_type == "one_tailed_less":
            p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        
        # Calculate confidence interval for effect size
        pooled_std = math.sqrt(
            ((len(control_data) - 1) * statistics.stdev(control_data)**2 +
             (len(treatment_data) - 1) * statistics.stdev(treatment_data)**2) /
            (len(control_data) + len(treatment_data) - 2)
        )
        
        se_diff = pooled_std * math.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(0.975, df)  # 95% confidence interval
        
        ci_lower = effect_size - t_critical * se_diff
        ci_upper = effect_size + t_critical * se_diff
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "effect_size": effect_size,
            "relative_effect": relative_effect,
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "degrees_of_freedom": df,
            "sample_size_control": len(control_data),
            "sample_size_treatment": len(treatment_data),
            "is_significant": p_value < 0.05
        }


class BayesianABTest:
    """Bayesian analysis for A/B testing with continuous monitoring"""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Bayesian A/B test
        
        Args:
            alpha_prior: Prior parameter for beta distribution (successes + 1)
            beta_prior: Prior parameter for beta distribution (failures + 1)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
    
    def update_posterior(
        self,
        control_data: List[float],
        treatment_data: List[float],
        is_binary: bool = True
    ) -> Dict[str, Any]:
        """
        Update posterior distributions and calculate probability of improvement
        
        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            is_binary: Whether data represents binary outcomes (0/1)
            
        Returns:
            Dictionary with posterior analysis results
        """
        if is_binary:
            # Binary outcomes (conversion rates)
            control_successes = sum(control_data)
            control_trials = len(control_data)
            treatment_successes = sum(treatment_data)
            treatment_trials = len(treatment_data)
            
            # Update beta distributions
            control_alpha = self.alpha_prior + control_successes
            control_beta = self.beta_prior + control_trials - control_successes
            treatment_alpha = self.alpha_prior + treatment_successes
            treatment_beta = self.beta_prior + treatment_trials - treatment_successes
            
            # Calculate probability that treatment > control
            prob_treatment_better = self._beta_probability_greater(
                treatment_alpha, treatment_beta, control_alpha, control_beta
            )
            
            return {
                "control_posterior_params": (control_alpha, control_beta),
                "treatment_posterior_params": (treatment_alpha, treatment_beta),
                "control_mean": control_alpha / (control_alpha + control_beta),
                "treatment_mean": treatment_alpha / (treatment_alpha + treatment_beta),
                "prob_treatment_better": prob_treatment_better,
                "credible_interval_control": beta.interval(0.95, control_alpha, control_beta),
                "credible_interval_treatment": beta.interval(0.95, treatment_alpha, treatment_beta)
            }
        else:
            # Continuous outcomes (using normal approximation)
            control_mean = statistics.mean(control_data)
            treatment_mean = statistics.mean(treatment_data)
            control_var = statistics.variance(control_data) if len(control_data) > 1 else 1.0
            treatment_var = statistics.variance(treatment_data) if len(treatment_data) > 1 else 1.0
            
            # Probability that treatment > control (normal approximation)
            diff_mean = treatment_mean - control_mean
            diff_var = control_var / len(control_data) + treatment_var / len(treatment_data)
            diff_std = math.sqrt(diff_var)
            
            prob_treatment_better = 1 - norm.cdf(0, diff_mean, diff_std)
            
            return {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "effect_size": diff_mean,
                "prob_treatment_better": prob_treatment_better,
                "credible_interval_effect": norm.interval(0.95, diff_mean, diff_std)
            }
    
    def _beta_probability_greater(
        self,
        alpha1: float, beta1: float,
        alpha2: float, beta2: float,
        n_samples: int = 10000
    ) -> float:
        """
        Calculate P(Beta(alpha1, beta1) > Beta(alpha2, beta2)) using Monte Carlo
        """
        samples1 = np.random.beta(alpha1, beta1, n_samples)
        samples2 = np.random.beta(alpha2, beta2, n_samples)
        return np.mean(samples1 > samples2)


class MultiArmedBandit:
    """Multi-armed bandit algorithms for dynamic traffic allocation"""
    
    def __init__(self, algorithm: str = "epsilon_greedy", **kwargs):
        """
        Initialize multi-armed bandit
        
        Args:
            algorithm: Algorithm type ("epsilon_greedy", "thompson_sampling", "ucb")
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.arms = {}  # arm_name -> {"rewards": [], "pulls": 0}
        self.total_pulls = 0
    
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Select arm based on bandit algorithm
        
        Args:
            context: Optional contextual information
            
        Returns:
            Selected arm name
        """
        if not self.arms:
            raise ValueError("No arms available for selection")
        
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy_select()
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling_select()
        elif self.algorithm == "ucb":
            return self._ucb_select()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def update_reward(self, arm_name: str, reward: float):
        """
        Update arm with observed reward
        
        Args:
            arm_name: Name of the arm
            reward: Observed reward
        """
        if arm_name not in self.arms:
            self.arms[arm_name] = {"rewards": [], "pulls": 0}
        
        self.arms[arm_name]["rewards"].append(reward)
        self.arms[arm_name]["pulls"] += 1
        self.total_pulls += 1
    
    def add_arm(self, arm_name: str):
        """Add a new arm to the bandit"""
        if arm_name not in self.arms:
            self.arms[arm_name] = {"rewards": [], "pulls": 0}
    
    def _epsilon_greedy_select(self) -> str:
        """Epsilon-greedy arm selection"""
        if random.random() < self.epsilon:
            # Explore: random selection
            return random.choice(list(self.arms.keys()))
        else:
            # Exploit: choose best arm
            best_arm = None
            best_mean = float('-inf')
            
            for arm_name, arm_data in self.arms.items():
                if arm_data["pulls"] == 0:
                    return arm_name  # Always try untested arms
                
                mean_reward = statistics.mean(arm_data["rewards"])
                if mean_reward > best_mean:
                    best_mean = mean_reward
                    best_arm = arm_name
            
            return best_arm
    
    def _thompson_sampling_select(self) -> str:
        """Thompson sampling arm selection"""
        samples = {}
        
        for arm_name, arm_data in self.arms.items():
            if arm_data["pulls"] == 0:
                # Beta(1, 1) for untested arms
                samples[arm_name] = np.random.beta(1, 1)
            else:
                # Assume rewards are in [0, 1] for Thompson sampling
                successes = sum(r for r in arm_data["rewards"] if r > 0.5)  # Simple threshold
                failures = arm_data["pulls"] - successes
                samples[arm_name] = np.random.beta(successes + 1, failures + 1)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def _ucb_select(self) -> str:
        """Upper Confidence Bound arm selection"""
        if self.total_pulls == 0:
            return random.choice(list(self.arms.keys()))
        
        ucb_values = {}
        
        for arm_name, arm_data in self.arms.items():
            if arm_data["pulls"] == 0:
                ucb_values[arm_name] = float('inf')  # Always try untested arms
            else:
                mean_reward = statistics.mean(arm_data["rewards"])
                confidence = math.sqrt(2 * math.log(self.total_pulls) / arm_data["pulls"])
                ucb_values[arm_name] = mean_reward + confidence
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]


class ABTestManager:
    """Main A/B testing manager for model preservation system"""
    
    def __init__(self):
        """Initialize A/B test manager"""
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[ABTestResult]] = {}
        self.test_metrics: Dict[str, List[ABTestMetric]] = {}
        self.bandits: Dict[str, MultiArmedBandit] = {}
        self.statistics = ABTestStatistics()
        self.bayesian = BayesianABTest()
    
    def create_ab_test(
        self,
        test_name: str,
        model_a_type: str,
        model_a_version: str,
        model_b_type: str,
        model_b_version: str,
        traffic_split: float = 0.5,
        test_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new A/B test
        
        Args:
            test_name: Unique name for the test
            model_a_type: Type of model A (control)
            model_a_version: Version of model A
            model_b_type: Type of model B (treatment)
            model_b_version: Version of model B
            traffic_split: Fraction of traffic for model B (0-1)
            test_config: Additional test configuration
            metadata: Optional metadata
            
        Returns:
            Test ID
        """
        if test_name in self.active_tests:
            raise ValueError(f"Test '{test_name}' already exists")
        
        config = ABTestConfig(
            test_name=test_name,
            model_a_type=model_a_type,
            model_a_version=model_a_version,
            model_b_type=model_b_type,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            **(test_config or {}),
            metadata=metadata or {}
        )
        
        self.active_tests[test_name] = config
        self.test_metrics[test_name] = []
        
        # Initialize bandit for dynamic allocation if enabled
        if test_config and test_config.get("enable_bandit"):
            bandit = MultiArmedBandit(
                algorithm=test_config.get("bandit_algorithm", "epsilon_greedy"),
                epsilon=test_config.get("epsilon", 0.1)
            )
            bandit.add_arm("model_a")
            bandit.add_arm("model_b")
            self.bandits[test_name] = bandit
        
        return test_name
    
    def route_ab_test_traffic(
        self,
        test_name: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route traffic for A/B test
        
        Args:
            test_name: Name of the test
            user_id: User identifier for consistent routing
            context: Optional routing context
            
        Returns:
            Variant name ("model_a" or "model_b")
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        config = self.active_tests[test_name]
        
        # Use bandit if available
        if test_name in self.bandits:
            return self.bandits[test_name].select_arm(context)
        
        # Deterministic routing based on user ID hash
        hash_value = int(hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest(), 16)
        assignment_value = (hash_value % 10000) / 10000  # Normalize to [0, 1)
        
        return "model_b" if assignment_value < config.traffic_split else "model_a"
    
    def record_ab_test_metrics(
        self,
        test_name: str,
        variant: str,
        user_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """
        Record metrics for A/B test
        
        Args:
            test_name: Name of the test
            variant: Variant name
            user_id: User identifier
            metrics: Dictionary of metric name -> value
            timestamp: Optional timestamp (defaults to now)
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        timestamp = timestamp or datetime.now()
        
        for metric_name, metric_value in metrics.items():
            metric_record = ABTestMetric(
                test_name=test_name,
                variant=variant,
                user_id=user_id,
                metric_name=metric_name,
                metric_value=metric_value,
                timestamp=timestamp
            )
            self.test_metrics[test_name].append(metric_record)
            
            # Update bandit if available
            if test_name in self.bandits and metric_name == "accuracy":
                self.bandits[test_name].update_reward(variant, metric_value)
    
    def analyze_ab_test(
        self,
        test_name: str,
        metric: str = "accuracy"
    ) -> ABTestResult:
        """
        Analyze A/B test results
        
        Args:
            test_name: Name of the test
            metric: Metric to analyze
            
        Returns:
            Analysis results
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        if test_name not in self.test_metrics:
            raise ValueError(f"No metrics found for test '{test_name}'")
        
        # Extract metric data by variant
        control_data = []
        treatment_data = []
        
        for metric_record in self.test_metrics[test_name]:
            if metric_record.metric_name == metric:
                if metric_record.variant == "model_a":
                    control_data.append(metric_record.metric_value)
                elif metric_record.variant == "model_b":
                    treatment_data.append(metric_record.metric_value)
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            raise ValueError("Insufficient data for statistical analysis")
        
        # Perform statistical analysis
        analysis = self.statistics.calculate_significance(control_data, treatment_data)
        
        return ABTestResult(
            test_name=test_name,
            metric=metric,
            control_mean=analysis["control_mean"],
            treatment_mean=analysis["treatment_mean"],
            effect_size=analysis["effect_size"],
            p_value=analysis["p_value"],
            confidence_interval=analysis["confidence_interval"],
            is_significant=analysis["is_significant"],
            power=0.8,  # Would calculate actual power in practice
            sample_size_control=analysis["sample_size_control"],
            sample_size_treatment=analysis["sample_size_treatment"]
        )
    
    def determine_ab_test_winner(
        self,
        test_name: str,
        confidence_level: float = 0.95,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Determine the winner of an A/B test
        
        Args:
            test_name: Name of the test
            confidence_level: Required confidence level
            metric: Metric to evaluate
            
        Returns:
            Winner determination results
        """
        result = self.analyze_ab_test(test_name, metric)
        alpha = 1 - confidence_level
        
        if result.p_value < alpha:
            if result.treatment_mean > result.control_mean:
                winner = "model_b"
                confidence = 1 - result.p_value
            else:
                winner = "model_a"
                confidence = 1 - result.p_value
        else:
            winner = "inconclusive"
            confidence = confidence_level
        
        return {
            "winner": winner,
            "confidence": confidence,
            "effect_size": result.effect_size,
            "relative_improvement": (result.treatment_mean - result.control_mean) / result.control_mean if result.control_mean != 0 else 0,
            "is_significant": result.is_significant,
            "p_value": result.p_value,
            "sample_sizes": {
                "control": result.sample_size_control,
                "treatment": result.sample_size_treatment
            }
        }
    
    def check_ab_test_early_stopping(
        self,
        test_name: str,
        min_effect_size: float = 0.05,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Check if A/B test should be stopped early
        
        Args:
            test_name: Name of the test
            min_effect_size: Minimum meaningful effect size
            metric: Metric to evaluate
            
        Returns:
            Early stopping recommendation
        """
        try:
            result = self.analyze_ab_test(test_name, metric)
            config = self.active_tests[test_name]
            
            # Check if we have enough samples
            total_samples = result.sample_size_control + result.sample_size_treatment
            has_min_samples = total_samples >= config.min_samples
            
            # Check effect size
            has_meaningful_effect = abs(result.effect_size) >= min_effect_size
            
            # Check significance
            is_significant = result.is_significant
            
            should_stop = has_min_samples and is_significant and has_meaningful_effect
            
            return {
                "should_stop": should_stop,
                "reason": self._get_stopping_reason(has_min_samples, is_significant, has_meaningful_effect),
                "current_samples": total_samples,
                "min_samples": config.min_samples,
                "effect_size": result.effect_size,
                "min_effect_size": min_effect_size,
                "p_value": result.p_value,
                "is_significant": is_significant
            }
        except ValueError as e:
            return {
                "should_stop": False,
                "reason": f"Cannot analyze: {str(e)}",
                "current_samples": 0,
                "min_samples": config.min_samples if test_name in self.active_tests else 0
            }
    
    def _get_stopping_reason(
        self,
        has_min_samples: bool,
        is_significant: bool,
        has_meaningful_effect: bool
    ) -> str:
        """Get reason for stopping decision"""
        if has_min_samples and is_significant and has_meaningful_effect:
            return "Significant result with meaningful effect size"
        elif not has_min_samples:
            return "Insufficient samples"
        elif not is_significant:
            return "Not statistically significant"
        elif not has_meaningful_effect:
            return "Effect size too small to be meaningful"
        else:
            return "Continue test"
    
    def list_active_ab_tests(self) -> List[Dict[str, Any]]:
        """List all active A/B tests"""
        return [
            {
                "test_name": name,
                "model_a": f"{config.model_a_type}:{config.model_a_version}",
                "model_b": f"{config.model_b_type}:{config.model_b_version}",
                "traffic_split": config.traffic_split,
                "duration_days": config.duration_days,
                "metrics": config.metrics,
                "metadata": config.metadata
            }
            for name, config in self.active_tests.items()
        ]
    
    def pause_ab_test(self, test_name: str):
        """Pause an A/B test"""
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        # In a real implementation, this would update the test status in database
        # For now, we'll just mark it in metadata
        self.active_tests[test_name].metadata["status"] = "paused"
        self.active_tests[test_name].metadata["paused_at"] = datetime.now().isoformat()
    
    def resume_ab_test(self, test_name: str):
        """Resume a paused A/B test"""
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        self.active_tests[test_name].metadata["status"] = "active"
        self.active_tests[test_name].metadata["resumed_at"] = datetime.now().isoformat()
    
    def terminate_ab_test(
        self,
        test_name: str,
        reason: str,
        preserve_data: bool = True
    ) -> Dict[str, Any]:
        """
        Terminate an A/B test
        
        Args:
            test_name: Name of the test
            reason: Reason for termination
            preserve_data: Whether to preserve test data
            
        Returns:
            Termination result
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        config = self.active_tests[test_name]
        
        # Create termination record
        termination_result = {
            "test_name": test_name,
            "terminated_at": datetime.now().isoformat(),
            "reason": reason,
            "preserve_data": preserve_data,
            "final_config": config.__dict__ if preserve_data else None,
            "final_metrics_count": len(self.test_metrics.get(test_name, [])) if preserve_data else 0
        }
        
        # Clean up test data if not preserving
        if not preserve_data:
            if test_name in self.test_metrics:
                del self.test_metrics[test_name]
            if test_name in self.test_results:
                del self.test_results[test_name]
            if test_name in self.bandits:
                del self.bandits[test_name]
        
        # Remove from active tests
        del self.active_tests[test_name]
        
        return termination_result