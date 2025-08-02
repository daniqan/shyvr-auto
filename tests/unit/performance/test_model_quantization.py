"""
Test-Driven Development for Model Quantization - Phase 5.2

This module contains comprehensive tests for model quantization including:
- Dynamic quantization for inference acceleration
- Static quantization with calibration
- Quantization-aware training preparation
- Post-training quantization optimization
- Performance benchmarking and validation

Following TDD methodology - these tests will fail initially and drive the implementation.
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class MockQuantizedModel:
    """Mock quantized model for testing"""
    original_size_mb: float
    quantized_size_mb: float
    precision: str = "int8"
    is_quantized: bool = True
    
    def eval(self):
        return self
    
    def to(self, device: str):
        return self
    
    def __call__(self, input_tensor):
        # Simulate quantized inference
        return torch.randn(1, 3)  # Mock output

@dataclass
class QuantizationBenchmark:
    """Benchmark results for quantization"""
    original_latency_ms: float
    quantized_latency_ms: float
    speedup_ratio: float
    accuracy_loss: float
    memory_reduction_ratio: float

class TestDynamicQuantization:
    """Test suite for dynamic quantization capabilities"""
    
    def test_dynamic_quantizer_initialization(self):
        """Test DynamicQuantizer initialization and configuration"""
        from src.ml_analysis.model_quantization import DynamicQuantizer, QuantizationConfig
        
        config = QuantizationConfig(
            target_dtype=torch.qint8,
            backend="fbgemm",
            preserve_precision_layers=["classifier", "output"],
            calibration_batches=100
        )
        
        quantizer = DynamicQuantizer(config)
        
        assert quantizer.config.target_dtype == torch.qint8
        assert quantizer.config.backend == "fbgemm"
        assert quantizer.config.preserve_precision_layers == ["classifier", "output"]
        assert quantizer.is_initialized is True
        assert quantizer.supported_layers is not None
    
    def test_model_analysis_for_quantization(self):
        """Test analysis of model suitability for quantization"""
        from src.ml_analysis.model_quantization import DynamicQuantizer
        
        quantizer = DynamicQuantizer()
        
        # Create mock model
        mock_model = Mock()
        mock_model.named_modules = Mock(return_value=[
            ("layer1", Mock(__class__=torch.nn.Linear)),
            ("layer2", Mock(__class__=torch.nn.Conv2d)),
            ("layer3", Mock(__class__=torch.nn.ReLU)),
            ("classifier", Mock(__class__=torch.nn.Linear))
        ])
        
        # Test quantization compatibility analysis
        analysis_result = quantizer.analyze_model_for_quantization(mock_model)
        
        assert "quantizable_layers" in analysis_result
        assert "non_quantizable_layers" in analysis_result
        assert "expected_speedup" in analysis_result
        assert "expected_memory_reduction" in analysis_result
        assert "precision_sensitive_layers" in analysis_result
        
        # Should identify Linear and Conv2d as quantizable
        quantizable_layers = analysis_result["quantizable_layers"]
        assert len(quantizable_layers) > 0
        
        # Should provide realistic estimates
        assert 1.0 <= analysis_result["expected_speedup"] <= 10.0
        assert 0.1 <= analysis_result["expected_memory_reduction"] <= 0.8
    
    def test_dynamic_quantization_application(self):
        """Test application of dynamic quantization to model"""
        from src.ml_analysis.model_quantization import DynamicQuantizer
        
        quantizer = DynamicQuantizer()
        
        # Create mock model with quantizable layers
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        
        # Test dynamic quantization
        quantized_model = quantizer.apply_dynamic_quantization(
            model=mock_model,
            dtype=torch.qint8,
            qconfig_spec={
                "": torch.quantization.default_dynamic_qconfig,
                "classifier": None  # Skip quantization for classifier
            }
        )
        
        assert quantized_model is not None
        mock_model.eval.assert_called_once()
        
        # Test quantization verification
        is_quantized = quantizer.verify_quantization(quantized_model)
        assert isinstance(is_quantized, bool)
        
        # Test quantized model properties
        model_info = quantizer.get_quantized_model_info(quantized_model)
        assert "quantized_layers" in model_info
        assert "precision" in model_info
        assert "backend" in model_info
        assert "compression_ratio" in model_info
    
    def test_selective_layer_quantization(self):
        """Test selective quantization of specific layers"""
        from src.ml_analysis.model_quantization import SelectiveQuantizer
        
        quantizer = SelectiveQuantizer()
        
        # Test layer selection based on sensitivity analysis
        mock_model = Mock()
        
        layer_sensitivity = quantizer.analyze_layer_sensitivity(
            model=mock_model,
            test_data=[torch.randn(1, 100) for _ in range(10)],
            accuracy_threshold=0.95
        )
        
        assert isinstance(layer_sensitivity, dict)
        for layer_name, sensitivity in layer_sensitivity.items():
            assert "accuracy_impact" in sensitivity
            assert "quantization_friendly" in sensitivity
            assert "recommended_precision" in sensitivity
        
        # Test selective quantization based on sensitivity
        quantization_strategy = quantizer.create_quantization_strategy(
            layer_sensitivity=layer_sensitivity,
            target_speedup=2.0,
            max_accuracy_loss=0.02
        )
        
        assert "quantize_layers" in quantization_strategy
        assert "preserve_layers" in quantization_strategy
        assert "mixed_precision_layers" in quantization_strategy
    
    def test_quantization_performance_validation(self):
        """Test performance validation of quantized models"""
        from src.ml_analysis.model_quantization import QuantizationValidator
        
        validator = QuantizationValidator()
        
        # Create mock original and quantized models
        original_model = Mock()
        quantized_model = MockQuantizedModel(
            original_size_mb=100.0,
            quantized_size_mb=25.0,
            precision="int8"
        )
        
        # Test performance benchmark
        benchmark_result = validator.benchmark_quantized_model(
            original_model=original_model,
            quantized_model=quantized_model,
            test_inputs=[torch.randn(1, 100) for _ in range(100)],
            warmup_iterations=10
        )
        
        assert "latency_comparison" in benchmark_result
        assert "throughput_comparison" in benchmark_result
        assert "memory_usage_comparison" in benchmark_result
        assert "accuracy_comparison" in benchmark_result
        
        # Verify benchmark metrics
        latency_comp = benchmark_result["latency_comparison"]
        assert "original_ms" in latency_comp
        assert "quantized_ms" in latency_comp
        assert "speedup_ratio" in latency_comp
        
        # Test accuracy validation
        accuracy_result = validator.validate_quantization_accuracy(
            original_model=original_model,
            quantized_model=quantized_model,
            validation_data=[torch.randn(1, 100) for _ in range(50)],
            tolerance=0.05
        )
        
        assert "accuracy_preserved" in accuracy_result
        assert "accuracy_loss" in accuracy_result
        assert "max_prediction_difference" in accuracy_result

class TestStaticQuantization:
    """Test suite for static quantization with calibration"""
    
    def test_static_quantizer_initialization(self):
        """Test StaticQuantizer initialization with calibration setup"""
        from src.ml_analysis.model_quantization import StaticQuantizer, CalibrationConfig
        
        calibration_config = CalibrationConfig(
            calibration_data_size=1000,
            num_calibration_batches=50,
            calibration_method="entropy",
            histogram_bins=2048
        )
        
        quantizer = StaticQuantizer(calibration_config)
        
        assert quantizer.calibration_config.calibration_data_size == 1000
        assert quantizer.calibration_config.num_calibration_batches == 50
        assert quantizer.calibration_config.calibration_method == "entropy"
        assert quantizer.is_calibrated is False
        assert quantizer.calibration_data is None
    
    def test_calibration_data_preparation(self):
        """Test preparation of calibration data for static quantization"""
        from src.ml_analysis.model_quantization import CalibrationDataPreparer
        
        preparer = CalibrationDataPreparer()
        
        # Test representative data selection
        full_dataset = [torch.randn(1, 100) for _ in range(1000)]
        
        calibration_data = preparer.select_representative_data(
            dataset=full_dataset,
            num_samples=100,
            selection_method="diverse"
        )
        
        assert len(calibration_data) == 100
        assert all(isinstance(sample, torch.Tensor) for sample in calibration_data)
        
        # Test data distribution analysis
        distribution_analysis = preparer.analyze_data_distribution(calibration_data)
        
        assert "feature_statistics" in distribution_analysis
        assert "activation_ranges" in distribution_analysis
        assert "outlier_detection" in distribution_analysis
        assert "representativeness_score" in distribution_analysis
        
        # Test data quality validation
        quality_report = preparer.validate_calibration_data_quality(
            calibration_data=calibration_data,
            original_dataset=full_dataset
        )
        
        assert "coverage_score" in quality_report
        assert "diversity_score" in quality_report
        assert "distribution_match" in quality_report
        assert "recommended_improvements" in quality_report
    
    def test_model_calibration_process(self):
        """Test the model calibration process for static quantization"""
        from src.ml_analysis.model_quantization import StaticQuantizer
        
        quantizer = StaticQuantizer()
        
        # Create mock model and calibration data
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        calibration_data = [torch.randn(1, 100) for _ in range(50)]
        
        # Test calibration process
        calibration_result = quantizer.calibrate_model(
            model=mock_model,
            calibration_data=calibration_data,
            num_epochs=1
        )
        
        assert "calibration_success" in calibration_result
        assert "activation_statistics" in calibration_result
        assert "quantization_parameters" in calibration_result
        assert "calibration_time_seconds" in calibration_result
        
        # Verify calibration state
        assert quantizer.is_calibrated is True
        
        # Test calibration statistics
        stats = quantizer.get_calibration_statistics()
        assert "layer_statistics" in stats
        assert "quantization_ranges" in stats
        assert "calibration_quality" in stats
    
    def test_static_quantization_application(self):
        """Test application of static quantization after calibration"""
        from src.ml_analysis.model_quantization import StaticQuantizer
        
        quantizer = StaticQuantizer()
        
        # Mock calibrated quantizer
        quantizer.is_calibrated = True
        quantizer.quantization_parameters = {
            "layer1": {"scale": 0.1, "zero_point": 128},
            "layer2": {"scale": 0.05, "zero_point": 127}
        }
        
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        
        # Test static quantization
        quantized_model = quantizer.apply_static_quantization(
            model=mock_model,
            backend="qnnpack"
        )
        
        assert quantized_model is not None
        
        # Test quantization parameter application
        applied_params = quantizer.get_applied_quantization_parameters()
        assert "layer1" in applied_params
        assert "layer2" in applied_params
        
        # Test quantized model validation
        validation_result = quantizer.validate_static_quantization(
            quantized_model=quantized_model,
            test_data=[torch.randn(1, 100) for _ in range(10)]
        )
        
        assert "quantization_accuracy" in validation_result
        assert "parameter_consistency" in validation_result
        assert "inference_correctness" in validation_result
    
    def test_quantization_parameter_optimization(self):
        """Test optimization of quantization parameters"""
        from src.ml_analysis.model_quantization import QuantizationParameterOptimizer
        
        optimizer = QuantizationParameterOptimizer()
        
        # Test parameter search
        mock_model = Mock()
        calibration_data = [torch.randn(1, 100) for _ in range(20)]
        
        optimal_params = optimizer.optimize_quantization_parameters(
            model=mock_model,
            calibration_data=calibration_data,
            optimization_method="grid_search",
            metrics=["accuracy", "latency", "memory"]
        )
        
        assert "optimal_scales" in optimal_params
        assert "optimal_zero_points" in optimal_params
        assert "optimization_score" in optimal_params
        assert "parameter_search_history" in optimal_params
        
        # Test parameter sensitivity analysis
        sensitivity_analysis = optimizer.analyze_parameter_sensitivity(
            model=mock_model,
            base_parameters=optimal_params["optimal_scales"],
            test_data=calibration_data
        )
        
        assert "sensitive_layers" in sensitivity_analysis
        assert "robust_layers" in sensitivity_analysis
        assert "parameter_ranges" in sensitivity_analysis

class TestQuantizationAwareTraining:
    """Test suite for quantization-aware training preparation"""
    
    def test_qat_model_preparation(self):
        """Test preparation of model for quantization-aware training"""
        from src.ml_analysis.model_quantization import QATPreparator
        
        preparator = QATPreparator()
        
        # Create mock model
        mock_model = Mock()
        mock_model.train = Mock(return_value=mock_model)
        
        # Test QAT preparation
        qat_model = preparator.prepare_model_for_qat(
            model=mock_model,
            backend="fbgemm",
            qconfig=None  # Use default
        )
        
        assert qat_model is not None
        
        # Test fake quantization insertion
        fake_quant_info = preparator.get_fake_quantization_info(qat_model)
        assert "fake_quant_modules" in fake_quant_info
        assert "observer_modules" in fake_quant_info
        assert "quantization_scheme" in fake_quant_info
        
        # Test QAT model validation
        validation_result = preparator.validate_qat_preparation(qat_model)
        assert "preparation_success" in validation_result
        assert "fake_quant_count" in validation_result
        assert "observer_count" in validation_result
    
    def test_qat_training_configuration(self):
        """Test configuration for quantization-aware training"""
        from src.ml_analysis.model_quantization import QATTrainingConfig
        
        config = QATTrainingConfig(
            learning_rate=0.001,
            weight_decay=1e-5,
            num_epochs=10,
            freeze_bn_delay=3,
            observer_update_interval=100,
            quantization_scheduler="cosine"
        )
        
        assert config.learning_rate == 0.001
        assert config.freeze_bn_delay == 3
        assert config.observer_update_interval == 100
        assert config.quantization_scheduler == "cosine"
        
        # Test training step simulation
        from src.ml_analysis.model_quantization import QATTrainer
        
        trainer = QATTrainer(config)
        
        # Mock training step
        mock_qat_model = Mock()
        mock_input = torch.randn(8, 100)
        mock_target = torch.randn(8, 3)
        
        training_step_result = trainer.simulate_training_step(
            model=mock_qat_model,
            input_data=mock_input,
            target=mock_target,
            step=50
        )
        
        assert "loss" in training_step_result
        assert "observer_updated" in training_step_result
        assert "bn_frozen" in training_step_result
        assert "quantization_enabled" in training_step_result
    
    def test_qat_model_conversion(self):
        """Test conversion from QAT model to quantized model"""
        from src.ml_analysis.model_quantization import QATConverter
        
        converter = QATConverter()
        
        # Mock QAT model
        mock_qat_model = Mock()
        mock_qat_model.eval = Mock(return_value=mock_qat_model)
        
        # Test conversion to quantized model
        quantized_model = converter.convert_qat_to_quantized(
            qat_model=mock_qat_model,
            backend="qnnpack"
        )
        
        assert quantized_model is not None
        
        # Test conversion validation
        validation_result = converter.validate_qat_conversion(
            qat_model=mock_qat_model,
            quantized_model=quantized_model,
            test_inputs=[torch.randn(1, 100) for _ in range(5)]
        )
        
        assert "conversion_success" in validation_result
        assert "accuracy_preservation" in validation_result
        assert "performance_improvement" in validation_result
        
        # Test quantized model optimization
        optimized_model = converter.optimize_quantized_model(
            quantized_model=quantized_model,
            optimization_passes=["constant_folding", "dead_code_elimination"]
        )
        
        assert optimized_model is not None

class TestQuantizationBackends:
    """Test suite for different quantization backends"""
    
    def test_fbgemm_backend(self):
        """Test FBGEMM backend for x86 CPU quantization"""
        from src.ml_analysis.model_quantization import FBGEMMQuantizer
        
        quantizer = FBGEMMQuantizer()
        
        # Test backend availability
        is_available = quantizer.is_backend_available()
        assert isinstance(is_available, bool)
        
        if is_available:
            # Test FBGEMM-specific quantization
            mock_model = Mock()
            
            quantized_model = quantizer.quantize_with_fbgemm(
                model=mock_model,
                calibration_data=[torch.randn(1, 100) for _ in range(10)]
            )
            
            assert quantized_model is not None
            
            # Test FBGEMM performance characteristics
            perf_characteristics = quantizer.get_performance_characteristics()
            assert "supported_dtypes" in perf_characteristics
            assert "optimal_batch_sizes" in perf_characteristics
            assert "threading_support" in perf_characteristics
    
    def test_qnnpack_backend(self):
        """Test QNNPACK backend for ARM/mobile quantization"""
        from src.ml_analysis.model_quantization import QNNPACKQuantizer
        
        quantizer = QNNPACKQuantizer()
        
        # Test backend availability
        is_available = quantizer.is_backend_available()
        assert isinstance(is_available, bool)
        
        if is_available:
            # Test QNNPACK-specific optimizations
            mock_model = Mock()
            
            mobile_optimized_model = quantizer.optimize_for_mobile(
                model=mock_model,
                target_platform="android"
            )
            
            assert mobile_optimized_model is not None
            
            # Test mobile deployment preparation
            deployment_package = quantizer.prepare_mobile_deployment(
                quantized_model=mobile_optimized_model,
                include_metadata=True
            )
            
            assert "model_file" in deployment_package
            assert "metadata" in deployment_package
            assert "performance_profile" in deployment_package
    
    def test_tensorrt_quantization(self):
        """Test TensorRT quantization for NVIDIA GPU inference"""
        from src.ml_analysis.model_quantization import TensorRTQuantizer
        
        quantizer = TensorRTQuantizer()
        
        # Test TensorRT availability
        is_available = quantizer.is_tensorrt_available()
        assert isinstance(is_available, bool)
        
        if is_available:
            # Test TensorRT INT8 calibration
            mock_model = Mock()
            calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
            
            tensorrt_engine = quantizer.build_tensorrt_engine(
                model=mock_model,
                calibration_data=calibration_data,
                precision="int8",
                max_batch_size=32
            )
            
            assert tensorrt_engine is not None
            
            # Test TensorRT inference optimization
            optimization_profile = quantizer.create_optimization_profile(
                min_shapes={"input": (1, 3, 224, 224)},
                opt_shapes={"input": (8, 3, 224, 224)},
                max_shapes={"input": (32, 3, 224, 224)}
            )
            
            assert "min_shapes" in optimization_profile
            assert "opt_shapes" in optimization_profile
            assert "max_shapes" in optimization_profile

class TestQuantizationProfiling:
    """Test suite for quantization performance profiling and analysis"""
    
    def test_quantization_profiler_initialization(self):
        """Test QuantizationProfiler initialization and setup"""
        from src.ml_analysis.model_quantization import QuantizationProfiler
        
        profiler = QuantizationProfiler(
            enable_layer_profiling=True,
            enable_memory_profiling=True,
            enable_accuracy_profiling=True
        )
        
        assert profiler.enable_layer_profiling is True
        assert profiler.enable_memory_profiling is True
        assert profiler.enable_accuracy_profiling is True
        assert profiler.profiling_data is not None
    
    def test_layer_wise_profiling(self):
        """Test layer-wise quantization impact profiling"""
        from src.ml_analysis.model_quantization import QuantizationProfiler
        
        profiler = QuantizationProfiler()
        
        # Mock model layers
        mock_layers = {
            "layer1": Mock(__class__=torch.nn.Linear),
            "layer2": Mock(__class__=torch.nn.Conv2d),
            "layer3": Mock(__class__=torch.nn.ReLU),
            "classifier": Mock(__class__=torch.nn.Linear)
        }
        
        # Test layer profiling
        layer_profile = profiler.profile_layer_quantization_impact(
            model_layers=mock_layers,
            test_data=[torch.randn(1, 100) for _ in range(20)]
        )
        
        assert isinstance(layer_profile, dict)
        
        for layer_name in mock_layers.keys():
            if layer_name in layer_profile:
                layer_stats = layer_profile[layer_name]
                assert "quantization_speedup" in layer_stats
                assert "accuracy_impact" in layer_stats
                assert "memory_reduction" in layer_stats
                assert "quantization_friendly" in layer_stats
    
    def test_accuracy_degradation_analysis(self):
        """Test accuracy degradation analysis for quantized models"""
        from src.ml_analysis.model_quantization import AccuracyAnalyzer
        
        analyzer = AccuracyAnalyzer()
        
        # Mock models and test data
        original_model = Mock()
        quantized_model = Mock()
        test_data = [(torch.randn(1, 100), torch.randint(0, 3, (1,))) for _ in range(50)]
        
        # Test accuracy analysis
        accuracy_analysis = analyzer.analyze_accuracy_degradation(
            original_model=original_model,
            quantized_model=quantized_model,
            test_data=test_data,
            metrics=["accuracy", "f1_score", "precision", "recall"]
        )
        
        assert "overall_accuracy_loss" in accuracy_analysis
        assert "per_class_impact" in accuracy_analysis
        assert "confidence_distribution" in accuracy_analysis
        assert "prediction_agreement" in accuracy_analysis
        
        # Test accuracy recovery strategies
        recovery_strategies = analyzer.suggest_accuracy_recovery_strategies(
            accuracy_analysis=accuracy_analysis,
            acceptable_loss_threshold=0.02
        )
        
        expected_strategies = [
            "increase_calibration_data", "use_mixed_precision",
            "apply_knowledge_distillation", "fine_tune_quantized_model"
        ]
        
        assert isinstance(recovery_strategies, list)
        for strategy in recovery_strategies:
            assert strategy["strategy_name"] in expected_strategies
            assert "expected_improvement" in strategy
            assert "implementation_complexity" in strategy
    
    def test_memory_usage_profiling(self):
        """Test memory usage profiling for quantized models"""
        from src.ml_analysis.model_quantization import MemoryProfiler
        
        profiler = MemoryProfiler()
        
        # Mock original and quantized models
        original_model = Mock()
        quantized_model = MockQuantizedModel(
            original_size_mb=200.0,
            quantized_size_mb=50.0
        )
        
        # Test memory profiling
        memory_profile = profiler.profile_memory_usage(
            original_model=original_model,
            quantized_model=quantized_model,
            input_shapes=[(1, 100), (8, 100), (32, 100)]
        )
        
        assert "model_size_reduction" in memory_profile
        assert "runtime_memory_usage" in memory_profile
        assert "peak_memory_usage" in memory_profile
        assert "memory_efficiency_score" in memory_profile
        
        # Test memory breakdown analysis
        memory_breakdown = profiler.analyze_memory_breakdown(quantized_model)
        
        assert "quantized_weights" in memory_breakdown
        assert "activation_memory" in memory_breakdown
        assert "buffer_memory" in memory_breakdown
        assert "overhead_memory" in memory_breakdown
    
    def test_inference_latency_profiling(self):
        """Test inference latency profiling for quantized models"""
        from src.ml_analysis.model_quantization import LatencyProfiler
        
        profiler = LatencyProfiler()
        
        # Mock models
        original_model = Mock()
        quantized_model = Mock()
        
        # Test latency profiling
        latency_profile = profiler.profile_inference_latency(
            original_model=original_model,
            quantized_model=quantized_model,
            input_sizes=[(1, 100), (8, 100), (16, 100), (32, 100)],
            num_warmup=10,
            num_iterations=100
        )
        
        assert "batch_size_analysis" in latency_profile
        assert "speedup_analysis" in latency_profile
        assert "latency_distribution" in latency_profile
        assert "bottleneck_analysis" in latency_profile
        
        # Test hardware-specific profiling
        for batch_size in [1, 8, 16, 32]:
            batch_profile = latency_profile["batch_size_analysis"][str(batch_size)]
            assert "original_latency_ms" in batch_profile
            assert "quantized_latency_ms" in batch_profile
            assert "speedup_ratio" in batch_profile
            assert "throughput_improvement" in batch_profile
    
    def test_quantization_robustness_testing(self):
        """Test robustness of quantized models under various conditions"""
        from src.ml_analysis.model_quantization import RobustnessAnalyzer
        
        analyzer = RobustnessAnalyzer()
        
        mock_quantized_model = Mock()
        
        # Test input distribution robustness
        distribution_robustness = analyzer.test_input_distribution_robustness(
            quantized_model=mock_quantized_model,
            reference_data=[torch.randn(1, 100) for _ in range(50)],
            test_distributions=["gaussian", "uniform", "exponential"]
        )
        
        assert "distribution_sensitivity" in distribution_robustness
        assert "performance_degradation" in distribution_robustness
        assert "robustness_score" in distribution_robustness
        
        # Test numerical stability
        stability_analysis = analyzer.test_numerical_stability(
            quantized_model=mock_quantized_model,
            test_cases=[
                torch.randn(1, 100) * 0.01,  # Small values
                torch.randn(1, 100) * 100,   # Large values
                torch.zeros(1, 100),         # Zero values
                torch.ones(1, 100) * 1e-8    # Very small values
            ]
        )
        
        assert "overflow_detection" in stability_analysis
        assert "underflow_detection" in stability_analysis
        assert "gradient_stability" in stability_analysis
        assert "numerical_precision" in stability_analysis

class TestQuantizationIntegration:
    """Test suite for quantization integration with existing ML pipeline"""
    
    def test_lstm_model_quantization_integration(self):
        """Test quantization integration with LSTM models"""
        from src.ml_analysis.model_quantization import LSTMQuantizationIntegrator
        from src.ml_analysis.lstm_model import LSTMPricePredictor
        
        integrator = LSTMQuantizationIntegrator()
        
        # Create mock LSTM predictor
        lstm_config = {"sequence_length": 50, "hidden_size": 128}
        lstm_predictor = LSTMPricePredictor(lstm_config)
        
        # Test LSTM-specific quantization
        quantization_strategy = integrator.create_lstm_quantization_strategy(
            lstm_predictor=lstm_predictor,
            target_speedup=2.0,
            preserve_sequence_accuracy=True
        )
        
        assert "quantize_lstm_layers" in quantization_strategy
        assert "preserve_attention_layers" in quantization_strategy
        assert "sequence_length_optimization" in quantization_strategy
        
        # Test quantized LSTM validation
        validation_result = integrator.validate_quantized_lstm(
            original_predictor=lstm_predictor,
            quantized_predictor=Mock(),
            test_sequences=[torch.randn(1, 50, 10) for _ in range(20)]
        )
        
        assert "sequence_accuracy_preserved" in validation_result
        assert "temporal_consistency" in validation_result
        assert "prediction_stability" in validation_result
    
    def test_dqn_agent_quantization_integration(self):
        """Test quantization integration with DQN agents"""
        from src.ml_analysis.model_quantization import DQNQuantizationIntegrator
        
        integrator = DQNQuantizationIntegrator()
        
        # Mock DQN agent
        mock_dqn_agent = Mock()
        mock_dqn_agent.q_network = Mock()
        mock_dqn_agent.target_network = Mock()
        
        # Test DQN-specific quantization
        quantization_plan = integrator.create_dqn_quantization_plan(
            dqn_agent=mock_dqn_agent,
            quantize_q_network=True,
            quantize_target_network=True,
            preserve_action_selection_accuracy=True
        )
        
        assert "q_network_quantization" in quantization_plan
        assert "target_network_quantization" in quantization_plan
        assert "action_selection_validation" in quantization_plan
        
        # Test quantized DQN performance
        performance_analysis = integrator.analyze_quantized_dqn_performance(
            original_agent=mock_dqn_agent,
            quantized_agent=Mock(),
            test_states=[torch.randn(1, 10) for _ in range(100)]
        )
        
        assert "action_selection_consistency" in performance_analysis
        assert "q_value_approximation_quality" in performance_analysis
        assert "training_stability_impact" in performance_analysis
    
    def test_inference_pipeline_quantization(self):
        """Test quantization integration with inference pipeline"""
        from src.ml_analysis.model_quantization import InferencePipelineQuantizer
        
        quantizer = InferencePipelineQuantizer()
        
        # Test pipeline quantization configuration
        pipeline_config = quantizer.configure_quantized_inference_pipeline(
            model_types=["lstm", "dqn"],
            target_latency_ms=50,
            memory_budget_mb=512,
            accuracy_threshold=0.95
        )
        
        assert "quantization_strategies" in pipeline_config
        assert "memory_allocation" in pipeline_config
        assert "performance_targets" in pipeline_config
        
        # Test quantized pipeline validation
        validation_result = quantizer.validate_quantized_pipeline(
            pipeline_config=pipeline_config,
            test_workload={
                "lstm_requests": 100,
                "dqn_requests": 200,
                "concurrent_requests": 50
            }
        )
        
        assert "latency_targets_met" in validation_result
        assert "memory_budget_respected" in validation_result
        assert "accuracy_thresholds_maintained" in validation_result