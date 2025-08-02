"""
Comprehensive TDD test suite for model drift detection system.

Tests for statistical drift detection algorithms including:
- Distribution shift detection (KS test, PSI, JS divergence)
- Feature drift detection
- Concept drift detection
- Data quality drift monitoring

Following TDD methodology - these tests will fail initially and drive implementation.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

# Import will fail initially - this drives the TDD implementation
try:
    from src.monitoring.drift_detection import (
        DriftDetector,
        StatisticalDriftAnalyzer,
        FeatureDriftMonitor,
        ConceptDriftDetector,
        DriftAlert,
        DriftSeverity,
        DriftType
    )
except ImportError:
    # Expected to fail initially in TDD approach
    pass


class TestDriftDetector:
    """Test suite for main DriftDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Sample training data (baseline)
        np.random.seed(42)
        self.training_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(2, 0.5, 1000),
            'feature_3': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.3, 1000)
        })
        
        # Sample production data (with drift)
        self.drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(0.5, 1.2, 500),  # Mean shift + variance change
            'feature_2': np.random.normal(1.8, 0.7, 500),  # Small mean shift + variance change
            'feature_3': np.random.exponential(1.5, 500),  # Distribution parameter change
            'target': np.random.binomial(1, 0.4, 500)  # Target distribution shift
        })
        
        # Non-drifted production data
        self.stable_data = pd.DataFrame({
            'feature_1': np.random.normal(0.05, 1.05, 500),  # Minimal drift
            'feature_2': np.random.normal(2.02, 0.52, 500),  # Minimal drift
            'feature_3': np.random.exponential(1.02, 500),  # Minimal drift
            'target': np.random.binomial(1, 0.31, 500)  # Minimal drift
        })
        
    def test_drift_detector_initialization(self):
        """Test DriftDetector proper initialization."""
        config = {
            'ks_test_threshold': 0.05,
            'psi_threshold': 0.2,
            'js_divergence_threshold': 0.1,
            'min_sample_size': 100,
            'statistical_power': 0.8
        }
        
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column='target',
            config=config
        )
        
        assert detector.reference_data is not None
        assert len(detector.feature_columns) == 3
        assert detector.target_column == 'target'
        assert detector.config['ks_test_threshold'] == 0.05
        assert detector.is_fitted is False
        
    def test_drift_detector_fit(self):
        """Test DriftDetector baseline fitting."""
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column='target'
        )
        
        detector.fit()
        
        assert detector.is_fitted is True
        assert detector.baseline_statistics is not None
        assert 'feature_distributions' in detector.baseline_statistics
        assert 'target_distribution' in detector.baseline_statistics
        assert 'correlation_matrix' in detector.baseline_statistics
        
    def test_detect_drift_with_significant_drift(self):
        """Test drift detection with significantly drifted data."""
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column='target'
        )
        detector.fit()
        
        drift_results = detector.detect_drift(self.drifted_data)
        
        assert drift_results is not None
        assert drift_results.has_drift is True
        assert drift_results.overall_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]
        assert len(drift_results.feature_drifts) == 3
        
        # Check specific drift types detected
        feature_1_drift = next(d for d in drift_results.feature_drifts if d.feature_name == 'feature_1')
        assert feature_1_drift.has_drift is True
        assert feature_1_drift.severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]
        
    def test_detect_drift_with_stable_data(self):
        """Test drift detection with stable (non-drifted) data."""
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column='target'
        )
        detector.fit()
        
        drift_results = detector.detect_drift(self.stable_data)
        
        assert drift_results is not None
        assert drift_results.has_drift is False
        assert drift_results.overall_severity == DriftSeverity.NONE
        
        # Most features should show no significant drift
        no_drift_features = [d for d in drift_results.feature_drifts if not d.has_drift]
        assert len(no_drift_features) >= 2  # At least 2 features should be stable
        
    def test_detect_drift_insufficient_samples(self):
        """Test drift detection with insufficient sample size."""
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column='target',
            config={'min_sample_size': 1000}
        )
        detector.fit()
        
        # Small sample data
        small_data = self.drifted_data.head(50)
        
        with pytest.raises(ValueError, match="Insufficient sample size"):
            detector.detect_drift(small_data)
            
    def test_drift_detector_without_target(self):
        """Test drift detector working without target column."""
        detector = DriftDetector(
            reference_data=self.training_data,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            target_column=None
        )
        detector.fit()
        
        drift_results = detector.detect_drift(self.drifted_data)
        
        assert drift_results is not None
        assert drift_results.target_drift is None
        assert len(drift_results.feature_drifts) == 3


class TestStatisticalDriftAnalyzer:
    """Test suite for statistical drift analysis algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Normal distribution baseline
        self.baseline_normal = np.random.normal(0, 1, 1000)
        # Shifted normal distribution
        self.shifted_normal = np.random.normal(0.5, 1, 500)
        # Different variance
        self.different_variance = np.random.normal(0, 2, 500)
        # Different distribution (exponential)
        self.different_dist = np.random.exponential(1, 500)
        
    def test_kolmogorov_smirnov_test(self):
        """Test Kolmogorov-Smirnov test for distribution comparison."""
        analyzer = StatisticalDriftAnalyzer()
        
        # Test with shifted distribution (should detect drift)
        ks_stat, p_value, has_drift = analyzer.kolmogorov_smirnov_test(
            baseline=self.baseline_normal,
            current=self.shifted_normal,
            alpha=0.05
        )
        
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert isinstance(has_drift, bool)
        assert ks_stat >= 0
        assert 0 <= p_value <= 1
        assert has_drift is True  # Should detect drift
        
        # Test with same distribution (should not detect drift)
        same_dist = np.random.normal(0.02, 1.01, 500)  # Very similar
        ks_stat_same, p_value_same, has_drift_same = analyzer.kolmogorov_smirnov_test(
            baseline=self.baseline_normal,
            current=same_dist,
            alpha=0.05
        )
        
        assert has_drift_same is False  # Should not detect drift
        assert p_value_same > 0.05
        
    def test_population_stability_index(self):
        """Test Population Stability Index (PSI) calculation."""
        analyzer = StatisticalDriftAnalyzer()
        
        # Test with shifted distribution
        psi_score, has_drift = analyzer.population_stability_index(
            baseline=self.baseline_normal,
            current=self.shifted_normal,
            n_bins=10,
            threshold=0.2
        )
        
        assert isinstance(psi_score, float)
        assert isinstance(has_drift, bool)
        assert psi_score >= 0
        assert has_drift is True  # Should detect drift
        
        # Test with similar distribution
        similar_dist = np.random.normal(0.01, 1.02, 500)
        psi_score_similar, has_drift_similar = analyzer.population_stability_index(
            baseline=self.baseline_normal,
            current=similar_dist,
            threshold=0.2
        )
        
        assert psi_score_similar < 0.2  # Should be below threshold
        assert has_drift_similar is False
        
    def test_jensen_shannon_divergence(self):
        """Test Jensen-Shannon divergence calculation."""
        analyzer = StatisticalDriftAnalyzer()
        
        # Test with different distributions
        js_divergence, has_drift = analyzer.jensen_shannon_divergence(
            baseline=self.baseline_normal,
            current=self.different_dist,
            n_bins=20,
            threshold=0.1
        )
        
        assert isinstance(js_divergence, float)
        assert isinstance(has_drift, bool)
        assert 0 <= js_divergence <= 1
        assert has_drift is True  # Should detect drift
        
        # Test with similar distributions
        similar_dist = np.random.normal(0.05, 1.05, 500)
        js_div_similar, has_drift_similar = analyzer.jensen_shannon_divergence(
            baseline=self.baseline_normal,
            current=similar_dist,
            threshold=0.1
        )
        
        assert js_div_similar < 0.1  # Should be below threshold
        assert has_drift_similar is False
        
    def test_chi_square_test(self):
        """Test Chi-square test for categorical drift detection."""
        analyzer = StatisticalDriftAnalyzer()
        
        # Categorical baseline data
        baseline_cat = np.random.choice(['A', 'B', 'C', 'D'], size=1000, p=[0.4, 0.3, 0.2, 0.1])
        # Shifted categorical data
        shifted_cat = np.random.choice(['A', 'B', 'C', 'D'], size=500, p=[0.2, 0.4, 0.3, 0.1])
        
        chi2_stat, p_value, has_drift = analyzer.chi_square_test(
            baseline=baseline_cat,
            current=shifted_cat,
            alpha=0.05
        )
        
        assert isinstance(chi2_stat, float)
        assert isinstance(p_value, float)
        assert isinstance(has_drift, bool)
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1
        assert has_drift is True  # Should detect drift
        
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        analyzer = StatisticalDriftAnalyzer()
        
        # Test with shifted distribution
        wd_distance, has_drift = analyzer.wasserstein_distance(
            baseline=self.baseline_normal,
            current=self.shifted_normal,
            threshold=0.2
        )
        
        assert isinstance(wd_distance, float)
        assert isinstance(has_drift, bool)
        assert wd_distance >= 0
        assert has_drift is True  # Should detect drift for shifted distribution


class TestFeatureDriftMonitor:
    """Test suite for feature-specific drift monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.feature_data = pd.DataFrame({
            'numerical_feature': np.random.normal(0, 1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2]),
            'binary_feature': np.random.binomial(1, 0.3, 1000),
            'ordinal_feature': np.random.choice([1, 2, 3, 4, 5], size=1000, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        })
        
        self.drifted_features = pd.DataFrame({
            'numerical_feature': np.random.normal(0.5, 1.5, 500),  # Mean and variance shift
            'categorical_feature': np.random.choice(['A', 'B', 'C'], size=500, p=[0.3, 0.4, 0.3]),  # Proportion shift
            'binary_feature': np.random.binomial(1, 0.6, 500),  # Probability shift
            'ordinal_feature': np.random.choice([1, 2, 3, 4, 5], size=500, p=[0.2, 0.3, 0.3, 0.1, 0.1])  # Order shift
        })
        
    def test_feature_drift_monitor_initialization(self):
        """Test FeatureDriftMonitor initialization."""
        monitor = FeatureDriftMonitor(
            feature_types={
                'numerical_feature': 'numerical',
                'categorical_feature': 'categorical',
                'binary_feature': 'binary',
                'ordinal_feature': 'ordinal'
            }
        )
        
        assert len(monitor.feature_types) == 4
        assert monitor.feature_types['numerical_feature'] == 'numerical'
        assert monitor.baseline_statistics is None
        assert monitor.is_fitted is False
        
    def test_fit_baseline_statistics(self):
        """Test fitting baseline statistics for all feature types."""
        monitor = FeatureDriftMonitor(
            feature_types={
                'numerical_feature': 'numerical',
                'categorical_feature': 'categorical',
                'binary_feature': 'binary',
                'ordinal_feature': 'ordinal'
            }
        )
        
        monitor.fit(self.feature_data)
        
        assert monitor.is_fitted is True
        assert monitor.baseline_statistics is not None
        assert 'numerical_feature' in monitor.baseline_statistics
        assert 'categorical_feature' in monitor.baseline_statistics
        
        # Check numerical feature statistics
        num_stats = monitor.baseline_statistics['numerical_feature']
        assert 'mean' in num_stats
        assert 'std' in num_stats
        assert 'distribution' in num_stats
        
        # Check categorical feature statistics
        cat_stats = monitor.baseline_statistics['categorical_feature']
        assert 'proportions' in cat_stats
        assert 'categories' in cat_stats
        
    def test_detect_numerical_feature_drift(self):
        """Test drift detection for numerical features."""
        monitor = FeatureDriftMonitor(
            feature_types={'numerical_feature': 'numerical'}
        )
        monitor.fit(self.feature_data)
        
        drift_result = monitor.detect_feature_drift(
            feature_name='numerical_feature',
            current_data=self.drifted_features['numerical_feature']
        )
        
        assert drift_result is not None
        assert drift_result.feature_name == 'numerical_feature'
        assert drift_result.feature_type == 'numerical'
        assert drift_result.has_drift is True
        assert drift_result.severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]
        assert 'ks_test' in drift_result.test_results
        assert 'psi' in drift_result.test_results
        
    def test_detect_categorical_feature_drift(self):
        """Test drift detection for categorical features."""
        monitor = FeatureDriftMonitor(
            feature_types={'categorical_feature': 'categorical'}
        )
        monitor.fit(self.feature_data)
        
        drift_result = monitor.detect_feature_drift(
            feature_name='categorical_feature',
            current_data=self.drifted_features['categorical_feature']
        )
        
        assert drift_result is not None
        assert drift_result.feature_name == 'categorical_feature'
        assert drift_result.feature_type == 'categorical'
        assert drift_result.has_drift is True
        assert 'chi_square' in drift_result.test_results
        assert 'psi' in drift_result.test_results
        
    def test_detect_all_features_drift(self):
        """Test batch drift detection for all features."""
        monitor = FeatureDriftMonitor(
            feature_types={
                'numerical_feature': 'numerical',
                'categorical_feature': 'categorical',
                'binary_feature': 'binary',
                'ordinal_feature': 'ordinal'
            }
        )
        monitor.fit(self.feature_data)
        
        drift_results = monitor.detect_all_features_drift(self.drifted_features)
        
        assert len(drift_results) == 4
        assert all(isinstance(result, object) for result in drift_results)  # Assuming FeatureDriftResult objects
        
        # At least some features should show drift
        drifted_features = [r for r in drift_results if r.has_drift]
        assert len(drifted_features) >= 2
        
    def test_feature_importance_weighting(self):
        """Test drift detection with feature importance weighting."""
        feature_importance = {
            'numerical_feature': 0.4,
            'categorical_feature': 0.3,
            'binary_feature': 0.2,
            'ordinal_feature': 0.1
        }
        
        monitor = FeatureDriftMonitor(
            feature_types={
                'numerical_feature': 'numerical',
                'categorical_feature': 'categorical',
                'binary_feature': 'binary',
                'ordinal_feature': 'ordinal'
            },
            feature_importance=feature_importance
        )
        monitor.fit(self.feature_data)
        
        weighted_drift_score = monitor.calculate_weighted_drift_score(self.drifted_features)
        
        assert isinstance(weighted_drift_score, float)
        assert 0 <= weighted_drift_score <= 1
        assert weighted_drift_score > 0  # Should detect some drift


class TestConceptDriftDetector:
    """Test suite for concept drift detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Stable concept: feature-target relationship remains consistent
        self.stable_features = np.random.normal(0, 1, (1000, 3))
        self.stable_targets = (self.stable_features[:, 0] + 0.5 * self.stable_features[:, 1] > 0).astype(int)
        
        # Concept drift: feature-target relationship changes
        self.drift_features = np.random.normal(0, 1, (500, 3))
        # Changed relationship: now depends more on feature 2 than feature 0
        self.drift_targets = (0.3 * self.drift_features[:, 0] + self.drift_features[:, 1] > 0).astype(int)
        
    def test_concept_drift_detector_initialization(self):
        """Test ConceptDriftDetector initialization."""
        detector = ConceptDriftDetector(
            model_type='classification',
            detection_method='prediction_drift',
            window_size=100
        )
        
        assert detector.model_type == 'classification'
        assert detector.detection_method == 'prediction_drift'
        assert detector.window_size == 100
        assert detector.baseline_model is None
        
    def test_fit_baseline_model(self):
        """Test fitting baseline model for concept drift detection."""
        detector = ConceptDriftDetector(
            model_type='classification',
            detection_method='prediction_drift'
        )
        
        detector.fit(self.stable_features, self.stable_targets)
        
        assert detector.baseline_model is not None
        assert detector.is_fitted is True
        assert detector.baseline_predictions is not None
        
    def test_detect_prediction_drift(self):
        """Test concept drift detection using prediction drift method."""
        detector = ConceptDriftDetector(
            model_type='classification',
            detection_method='prediction_drift'
        )
        detector.fit(self.stable_features, self.stable_targets)
        
        # Test with concept drift data
        drift_result = detector.detect_concept_drift(
            features=self.drift_features,
            targets=self.drift_targets
        )
        
        assert drift_result is not None
        assert drift_result.has_concept_drift is True
        assert drift_result.drift_type == DriftType.CONCEPT
        assert drift_result.severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]
        assert 'prediction_drift_score' in drift_result.metrics
        
    def test_detect_performance_degradation(self):
        """Test concept drift detection using performance degradation method."""
        detector = ConceptDriftDetector(
            model_type='classification',
            detection_method='performance_degradation'
        )
        detector.fit(self.stable_features, self.stable_targets)
        
        # Test with concept drift data (should show performance degradation)
        drift_result = detector.detect_concept_drift(
            features=self.drift_features,
            targets=self.drift_targets
        )
        
        assert drift_result is not None
        assert 'accuracy_drop' in drift_result.metrics
        assert 'f1_drop' in drift_result.metrics
        assert drift_result.has_concept_drift is True
        
    def test_windowed_concept_drift_detection(self):
        """Test windowed concept drift detection for streaming data."""
        detector = ConceptDriftDetector(
            model_type='classification',
            detection_method='windowed',
            window_size=200
        )
        detector.fit(self.stable_features, self.stable_targets)
        
        # Simulate streaming data with concept drift
        streaming_features = np.vstack([self.stable_features[:300], self.drift_features])
        streaming_targets = np.concatenate([self.stable_targets[:300], self.drift_targets])
        
        drift_points = detector.detect_streaming_concept_drift(
            features=streaming_features,
            targets=streaming_targets
        )
        
        assert isinstance(drift_points, list)
        assert len(drift_points) > 0  # Should detect drift points
        
        # Drift should be detected around index 300 (where concept changes)
        drift_indices = [dp.change_point for dp in drift_points]
        assert any(250 <= idx <= 350 for idx in drift_indices)


class TestDriftAlert:
    """Test suite for drift alerting system."""
    
    def test_drift_alert_creation(self):
        """Test DriftAlert object creation and properties."""
        alert = DriftAlert(
            alert_id="test_alert_001",
            timestamp=datetime.now(),
            drift_type=DriftType.FEATURE,
            severity=DriftSeverity.MODERATE,
            feature_name="feature_1",
            message="Feature drift detected in feature_1",
            metrics={'psi_score': 0.3, 'ks_statistic': 0.15},
            recommended_actions=['retrain_model', 'investigate_data_source']
        )
        
        assert alert.alert_id == "test_alert_001"
        assert alert.drift_type == DriftType.FEATURE
        assert alert.severity == DriftSeverity.MODERATE
        assert alert.feature_name == "feature_1"
        assert alert.metrics['psi_score'] == 0.3
        assert 'retrain_model' in alert.recommended_actions
        
    def test_drift_severity_ordering(self):
        """Test DriftSeverity enum ordering."""
        assert DriftSeverity.NONE < DriftSeverity.LOW
        assert DriftSeverity.LOW < DriftSeverity.MODERATE
        assert DriftSeverity.MODERATE < DriftSeverity.SEVERE
        assert DriftSeverity.SEVERE < DriftSeverity.CRITICAL
        
    def test_drift_type_categorization(self):
        """Test DriftType enum values."""
        expected_types = [DriftType.FEATURE, DriftType.CONCEPT, DriftType.COVARIATE, DriftType.TARGET]
        assert all(dt in DriftType for dt in expected_types)


class TestDriftDetectionIntegration:
    """Integration tests for the complete drift detection system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        np.random.seed(42)
        # Create realistic market data scenario
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Baseline period (stable market conditions)
        baseline_size = 200
        self.baseline_data = pd.DataFrame({
            'price': np.random.normal(100, 10, baseline_size),
            'volume': np.random.exponential(1000, baseline_size),
            'volatility': np.random.gamma(2, 0.1, baseline_size),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 
                                        baseline_size, p=[0.3, 0.5, 0.2]),
            'returns': np.random.normal(0.001, 0.02, baseline_size),
            'timestamp': dates[:baseline_size]
        })
        
        # Market regime change (post-baseline data with drift)
        drift_size = 100
        self.drift_data = pd.DataFrame({
            'price': np.random.normal(120, 15, drift_size),  # Higher prices, more volatility
            'volume': np.random.exponential(1500, drift_size),  # Higher trading volume
            'volatility': np.random.gamma(3, 0.15, drift_size),  # Increased volatility
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 
                                        drift_size, p=[0.5, 0.3, 0.2]),  # More positive sentiment
            'returns': np.random.normal(0.005, 0.035, drift_size),  # Higher returns and volatility
            'timestamp': dates[baseline_size:baseline_size + drift_size]
        })
        
    def test_end_to_end_drift_detection(self):
        """Test complete end-to-end drift detection workflow."""
        # Initialize drift detector with market data configuration
        detector = DriftDetector(
            reference_data=self.baseline_data,
            feature_columns=['price', 'volume', 'volatility', 'sentiment', 'returns'],
            target_column=None,  # Unsupervised drift detection
            config={
                'ks_test_threshold': 0.05,
                'psi_threshold': 0.2,
                'js_divergence_threshold': 0.1,
                'min_sample_size': 50
            }
        )
        
        # Fit baseline
        detector.fit()
        
        # Detect drift in new market regime
        drift_results = detector.detect_drift(self.drift_data)
        
        # Assertions for complete workflow
        assert drift_results is not None
        assert drift_results.has_drift is True
        assert len(drift_results.feature_drifts) == 5
        
        # Should detect drift in price and volume (most significant changes)
        price_drift = next(d for d in drift_results.feature_drifts if d.feature_name == 'price')
        volume_drift = next(d for d in drift_results.feature_drifts if d.feature_name == 'volume')
        
        assert price_drift.has_drift is True
        assert volume_drift.has_drift is True
        
        # Generate alerts
        alerts = detector.generate_drift_alerts(drift_results)
        assert isinstance(alerts, list)
        assert len(alerts) > 0
        
        # Check alert properties
        for alert in alerts:
            assert isinstance(alert, DriftAlert)
            assert alert.severity != DriftSeverity.NONE
            assert alert.timestamp is not None
            assert len(alert.recommended_actions) > 0
            
    def test_batch_drift_monitoring(self):
        """Test batch processing for multiple time windows."""
        detector = DriftDetector(
            reference_data=self.baseline_data,
            feature_columns=['price', 'volume', 'volatility', 'returns'],
            target_column=None
        )
        detector.fit()
        
        # Create multiple batches of data
        batch_size = 25
        batches = []
        for i in range(0, len(self.drift_data), batch_size):
            batch = self.drift_data.iloc[i:i+batch_size]
            if len(batch) >= batch_size:  # Only full batches
                batches.append(batch)
        
        # Process batches
        batch_results = []
        for i, batch in enumerate(batches):
            result = detector.detect_drift(batch)
            result.batch_id = i
            batch_results.append(result)
        
        assert len(batch_results) > 0
        
        # Analyze batch results trends
        drift_scores = [r.overall_drift_score for r in batch_results]
        assert all(isinstance(score, (int, float)) for score in drift_scores)
        
        # Should show increasing drift scores over time (market regime change)
        assert max(drift_scores) > min(drift_scores)
        
    def test_real_time_drift_monitoring_simulation(self):
        """Test real-time drift monitoring simulation."""
        detector = DriftDetector(
            reference_data=self.baseline_data,
            feature_columns=['price', 'volume', 'volatility'],
            target_column=None,
            config={'min_sample_size': 20}  # Smaller for real-time
        )
        detector.fit()
        
        # Simulate real-time data stream
        combined_data = pd.concat([self.baseline_data, self.drift_data])
        
        real_time_alerts = []
        window_size = 30
        
        for i in range(len(self.baseline_data), len(combined_data) - window_size, 5):
            # Current window
            current_window = combined_data.iloc[i:i+window_size]
            
            if len(current_window) >= 20:  # Minimum sample size
                result = detector.detect_drift(current_window)
                
                if result.has_drift and result.overall_severity.value >= DriftSeverity.MODERATE.value:
                    alerts = detector.generate_drift_alerts(result)
                    real_time_alerts.extend(alerts)
        
        # Should generate some alerts during market regime change
        assert len(real_time_alerts) > 0
        
        # Check alert timing
        alert_timestamps = [alert.timestamp for alert in real_time_alerts]
        assert len(set(alert_timestamps)) > 0  # Multiple unique timestamps


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])