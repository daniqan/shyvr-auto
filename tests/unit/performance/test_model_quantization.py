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

class TestTransformerQuantization:
    """Test suite for Transformer-specific quantization capabilities (Phase 2.1)"""
    
    def test_transformer_quantization_compatibility_analysis(self):
        """Test analysis of Transformer models for quantization compatibility"""
        from src.ml_analysis.model_quantization import TransformerQuantizer
        from src.ml_analysis.transformers.base import TransformerBase, TransformerConfig
        
        quantizer = TransformerQuantizer()
        
        # Create mock transformer config
        transformer_config = TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_length=1000
        )
        
        # Create mock transformer model
        mock_transformer = Mock(spec=TransformerBase)
        mock_transformer.transformer_config = transformer_config
        mock_transformer.named_modules = Mock(return_value=[
            ("encoder.layers.0.self_attn", Mock(__class__=torch.nn.MultiheadAttention)),
            ("encoder.layers.0.linear1", Mock(__class__=torch.nn.Linear)),
            ("encoder.layers.0.linear2", Mock(__class__=torch.nn.Linear)),
            ("encoder.layers.1.self_attn", Mock(__class__=torch.nn.MultiheadAttention)),
            ("decoder.output_projection", Mock(__class__=torch.nn.Linear))
        ])
        
        # Test transformer-specific analysis
        compatibility_analysis = quantizer.analyze_transformer_quantization_compatibility(
            transformer_model=mock_transformer,
            preserve_attention_precision=True,
            target_accuracy_retention=0.95
        )
        
        assert "attention_layer_analysis" in compatibility_analysis
        assert "feed_forward_layer_analysis" in compatibility_analysis
        assert "output_projection_analysis" in compatibility_analysis
        assert "sequence_length_impact" in compatibility_analysis
        assert "attention_precision_requirements" in compatibility_analysis
        assert "expected_speedup_transformer" in compatibility_analysis
        assert "memory_reduction_transformer" in compatibility_analysis
        
        # Verify attention layers are marked as sensitive
        attention_analysis = compatibility_analysis["attention_layer_analysis"]
        assert len(attention_analysis) > 0
        for layer_info in attention_analysis:
            assert "precision_sensitive" in layer_info
            assert "quantization_strategy" in layer_info
    
    def test_transformer_specific_quantization_strategies(self):
        """Test development of Transformer-specific quantization strategies"""
        from src.ml_analysis.model_quantization import TransformerQuantizationStrategy
        
        strategy_builder = TransformerQuantizationStrategy()
        
        # Mock transformer architecture analysis
        transformer_architecture = {
            "attention_layers": ["layer.0.attention", "layer.1.attention", "layer.2.attention"],
            "feed_forward_layers": ["layer.0.feed_forward", "layer.1.feed_forward"],
            "output_layers": ["output_projection"],
            "embedding_layers": ["input_embeddings", "positional_embeddings"]
        }
        
        # Test strategy creation for different transformer variants
        for model_type in ["iTransformer", "PatchTST", "TimesMixer", "TransformerPredictor"]:
            quantization_strategy = strategy_builder.create_transformer_strategy(
                model_type=model_type,
                architecture_info=transformer_architecture,
                target_speedup=2.5,
                max_accuracy_loss=0.05,
                preserve_attention_patterns=True
            )
            
            assert "attention_quantization_plan" in quantization_strategy
            assert "feed_forward_quantization_plan" in quantization_strategy
            assert "embedding_quantization_plan" in quantization_strategy
            assert "precision_allocation" in quantization_strategy
            assert "expected_performance_gain" in quantization_strategy
            
            # Verify attention layers get special treatment
            attention_plan = quantization_strategy["attention_quantization_plan"]
            assert "preserve_key_query_precision" in attention_plan
            assert "value_projection_quantization" in attention_plan
            assert "attention_output_quantization" in attention_plan
    
    def test_attention_layer_quantization_handling(self):
        """Test specialized handling of attention layers during quantization"""
        from src.ml_analysis.model_quantization import AttentionLayerQuantizer
        
        attention_quantizer = AttentionLayerQuantizer()
        
        # Mock multi-head attention layer
        mock_attention = Mock()
        mock_attention.num_heads = 8
        mock_attention.embed_dim = 512
        mock_attention.in_proj_weight = Mock()
        mock_attention.out_proj = Mock()
        
        # Test attention-specific quantization
        attention_quant_config = attention_quantizer.create_attention_quantization_config(
            attention_layer=mock_attention,
            preserve_attention_weights=True,
            quantize_key_query=False,  # Keep key/query in higher precision
            quantize_value=True,       # Value can be quantized more aggressively
            quantize_output_proj=True
        )
        
        assert "key_query_precision" in attention_quant_config
        assert "value_precision" in attention_quant_config
        assert "output_projection_precision" in attention_quant_config
        assert "attention_score_handling" in attention_quant_config
        assert "softmax_precision_requirements" in attention_quant_config
        
        # Test quantized attention validation
        quantized_attention = attention_quantizer.apply_attention_quantization(
            attention_layer=mock_attention,
            quantization_config=attention_quant_config
        )
        
        assert quantized_attention is not None
        
        # Test attention pattern preservation
        pattern_preservation = attention_quantizer.validate_attention_pattern_preservation(
            original_attention=mock_attention,
            quantized_attention=quantized_attention,
            test_inputs=[torch.randn(8, 100, 512) for _ in range(10)]
        )
        
        assert "attention_pattern_similarity" in pattern_preservation
        assert "attention_entropy_preservation" in pattern_preservation
        assert "head_specialization_maintained" in pattern_preservation
    
    def test_transformer_model_quantization_application(self):
        """Test application of quantization to complete Transformer models"""
        from src.ml_analysis.model_quantization import TransformerQuantizer
        from src.ml_analysis.transformers.itransformer import iTransformerPredictor
        from src.ml_analysis.transformers.patchtst import PatchTSTPredictor
        from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor
        from src.ml_analysis.transformers.transformer_predictor import TransformerPredictor
        
        quantizer = TransformerQuantizer()
        
        # Test quantization for each transformer variant
        transformer_configs = {
            "iTransformer": {"d_model": 512, "n_heads": 8, "n_layers": 4},
            "PatchTST": {"d_model": 256, "n_heads": 4, "patch_length": 16},
            "TimesMixer": {"d_model": 384, "n_mixing_layers": 3},
            "TransformerPredictor": {"d_model": 512, "n_heads": 8, "n_layers": 6}
        }
        
        for model_name, config in transformer_configs.items():
            # Create mock transformer predictor
            mock_predictor = Mock()
            mock_predictor.model_type = model_name
            mock_predictor.config = config
            mock_predictor.is_model_trained = Mock(return_value=True)
            
            # Test transformer quantization
            quantized_predictor = quantizer.quantize_transformer_model(
                transformer_predictor=mock_predictor,
                quantization_type="dynamic",  # or "static"
                target_dtype=torch.qint8,
                preserve_accuracy_threshold=0.95
            )
            
            assert quantized_predictor is not None
            
            # Test quantized model properties
            quant_properties = quantizer.get_quantized_transformer_properties(quantized_predictor)
            assert "model_type" in quant_properties
            assert "quantization_ratio" in quant_properties
            assert "attention_layers_quantized" in quant_properties
            assert "feed_forward_layers_quantized" in quant_properties
            assert "memory_reduction" in quant_properties
            assert "expected_speedup" in quant_properties
    
    def test_transformer_quantization_accuracy_validation(self):
        """Test accuracy validation for quantized Transformer models"""
        from src.ml_analysis.model_quantization import TransformerAccuracyValidator
        from src.discovery.base import DiscoveredToken
        
        validator = TransformerAccuracyValidator()
        
        # Mock original and quantized transformers
        mock_original_transformer = Mock()
        mock_quantized_transformer = Mock()
        
        # Mock test data
        test_tokens = [
            DiscoveredToken(
                address=f"token_{i}",
                symbol=f"TEST{i}",
                price_usd=100.0 + i,
                market_cap=1000000.0,
                volume_24h=50000.0
            ) for i in range(20)
        ]
        
        # Test accuracy validation
        accuracy_validation = validator.validate_transformer_quantization_accuracy(
            original_transformer=mock_original_transformer,
            quantized_transformer=mock_quantized_transformer,
            test_tokens=test_tokens,
            accuracy_threshold=0.95,
            max_prediction_difference=0.05
        )
        
        assert "overall_accuracy_maintained" in accuracy_validation
        assert "prediction_correlation" in accuracy_validation
        assert "attention_pattern_consistency" in accuracy_validation
        assert "confidence_score_consistency" in accuracy_validation
        assert "direction_prediction_agreement" in accuracy_validation
        assert "per_token_accuracy_analysis" in accuracy_validation
        
        # Test specific transformer accuracy metrics
        transformer_metrics = validator.compute_transformer_specific_accuracy_metrics(
            original_predictions=[],  # Mock predictions
            quantized_predictions=[],  # Mock predictions
            attention_weights_original=torch.randn(4, 8, 100, 100),  # Mock attention
            attention_weights_quantized=torch.randn(4, 8, 100, 100)   # Mock attention
        )
        
        assert "attention_weight_correlation" in transformer_metrics
        assert "temporal_consistency_score" in transformer_metrics
        assert "feature_importance_preservation" in transformer_metrics
        assert "multi_horizon_accuracy_retention" in transformer_metrics
    
    def test_transformer_quantization_performance_benchmarking(self):
        """Test performance benchmarking for quantized Transformer models"""
        from src.ml_analysis.model_quantization import TransformerPerformanceBenchmarker
        
        benchmarker = TransformerPerformanceBenchmarker()
        
        # Mock transformer models
        mock_fp32_transformer = Mock()
        mock_int8_transformer = Mock()
        mock_mixed_precision_transformer = Mock()
        
        # Test comprehensive performance benchmark
        performance_benchmark = benchmarker.benchmark_transformer_quantization_performance(
            fp32_model=mock_fp32_transformer,
            int8_model=mock_int8_transformer,
            mixed_precision_model=mock_mixed_precision_transformer,
            test_sequence_lengths=[64, 128, 256, 512, 1000],
            batch_sizes=[1, 4, 8, 16],
            num_warmup_iterations=20,
            num_benchmark_iterations=100
        )
        
        assert "sequence_length_scaling" in performance_benchmark
        assert "batch_size_scaling" in performance_benchmark
        assert "attention_computation_speedup" in performance_benchmark
        assert "memory_usage_comparison" in performance_benchmark
        assert "throughput_analysis" in performance_benchmark
        
        # Verify sequence length scaling analysis
        seq_scaling = performance_benchmark["sequence_length_scaling"]
        for seq_len in [64, 128, 256, 512, 1000]:
            assert str(seq_len) in seq_scaling
            seq_data = seq_scaling[str(seq_len)]
            assert "fp32_latency_ms" in seq_data
            assert "int8_latency_ms" in seq_data
            assert "mixed_precision_latency_ms" in seq_data
            assert "speedup_int8" in seq_data
            assert "speedup_mixed_precision" in seq_data
        
        # Test memory efficiency analysis
        memory_analysis = benchmarker.analyze_transformer_memory_efficiency(
            fp32_model=mock_fp32_transformer,
            quantized_models={
                "int8": mock_int8_transformer,
                "mixed_precision": mock_mixed_precision_transformer
            },
            max_sequence_length=1000
        )
        
        assert "model_size_reduction" in memory_analysis
        assert "runtime_memory_savings" in memory_analysis
        assert "attention_memory_optimization" in memory_analysis
        assert "peak_memory_usage" in memory_analysis
    
    def test_flash_attention_quantization_compatibility(self):
        """Test compatibility between Flash Attention optimization and quantization"""
        from src.ml_analysis.model_quantization import FlashAttentionQuantizationIntegrator
        
        integrator = FlashAttentionQuantizationIntegrator()
        
        # Mock Flash Attention enabled transformer
        mock_flash_attention_transformer = Mock()
        mock_flash_attention_transformer.config = {"use_flash_attention": True}
        
        # Test Flash Attention + Quantization compatibility
        compatibility_analysis = integrator.analyze_flash_attention_quantization_compatibility(
            flash_attention_model=mock_flash_attention_transformer,
            target_quantization="int8"
        )
        
        assert "flash_attention_quantization_support" in compatibility_analysis
        assert "memory_optimization_stacking" in compatibility_analysis
        assert "performance_compound_effect" in compatibility_analysis
        assert "implementation_challenges" in compatibility_analysis
        
        # Test combined optimization application
        if compatibility_analysis["flash_attention_quantization_support"]:
            optimized_model = integrator.apply_combined_flash_attention_quantization(
                model=mock_flash_attention_transformer,
                quantization_config={
                    "attention_layers": "mixed_precision",
                    "feed_forward_layers": "int8",
                    "preserve_flash_attention": True
                }
            )
            
            assert optimized_model is not None
            
            # Test combined optimization validation
            optimization_validation = integrator.validate_combined_optimization(
                original_model=mock_flash_attention_transformer,
                optimized_model=optimized_model,
                test_inputs=[torch.randn(1, 512, 512) for _ in range(5)]
            )
            
            assert "flash_attention_preserved" in optimization_validation
            assert "quantization_applied_successfully" in optimization_validation
            assert "combined_speedup" in optimization_validation
            assert "memory_savings_combined" in optimization_validation
    
    def test_quantization_aware_training_for_transformers(self):
        """Test quantization-aware training preparation for Transformer models"""
        from src.ml_analysis.model_quantization import TransformerQATPreparator
        
        qat_preparator = TransformerQATPreparator()
        
        # Mock transformer model for QAT
        mock_transformer = Mock()
        mock_transformer.named_modules = Mock(return_value=[
            ("encoder.layers.0.self_attn", Mock(__class__=torch.nn.MultiheadAttention)),
            ("encoder.layers.0.feed_forward.linear1", Mock(__class__=torch.nn.Linear)),
            ("encoder.layers.0.feed_forward.linear2", Mock(__class__=torch.nn.Linear))
        ])
        
        # Test QAT preparation for transformers
        qat_transformer = qat_preparator.prepare_transformer_for_qat(
            transformer_model=mock_transformer,
            qat_config={
                "attention_qat_strategy": "conservative",  # Keep attention in higher precision longer
                "feed_forward_qat_strategy": "aggressive",  # More aggressive quantization for FF layers
                "embedding_qat_strategy": "moderate"
            }
        )
        
        assert qat_transformer is not None
        
        # Test QAT-specific fake quantization
        fake_quant_analysis = qat_preparator.analyze_transformer_fake_quantization(qat_transformer)
        assert "attention_fake_quant_modules" in fake_quant_analysis
        assert "feed_forward_fake_quant_modules" in fake_quant_analysis
        assert "embedding_fake_quant_modules" in fake_quant_analysis
        assert "quantization_simulation_accuracy" in fake_quant_analysis
        
        # Test QAT training configuration for transformers
        transformer_qat_config = qat_preparator.create_transformer_qat_training_config(
            learning_rate=1e-4,
            quantization_warmup_epochs=2,
            attention_quantization_delay=5,  # Delay quantization of attention layers
            fine_tuning_epochs=10
        )
        
        assert "attention_specific_schedule" in transformer_qat_config
        assert "layer_wise_quantization_timing" in transformer_qat_config
        assert "precision_annealing_schedule" in transformer_qat_config
    
    def test_transformer_quantization_deployment_pipeline(self):
        """Test deployment pipeline integration for quantized Transformer models"""
        from src.ml_analysis.model_quantization import TransformerQuantizationDeploymentManager
        
        deployment_manager = TransformerQuantizationDeploymentManager()
        
        # Mock quantized transformer models
        quantized_models = {
            "iTransformer": Mock(),
            "PatchTST": Mock(),
            "TimesMixer": Mock(),
            "TransformerPredictor": Mock()
        }
        
        # Test deployment preparation
        deployment_package = deployment_manager.prepare_quantized_transformer_deployment(
            quantized_models=quantized_models,
            target_environment="production",
            optimization_level="aggressive",
            include_fallback_models=True
        )
        
        assert "quantized_model_artifacts" in deployment_package
        assert "performance_profiles" in deployment_package
        assert "fallback_strategy" in deployment_package
        assert "monitoring_configuration" in deployment_package
        assert "resource_requirements" in deployment_package
        
        # Test production validation
        production_validation = deployment_manager.validate_production_quantized_transformers(
            deployment_package=deployment_package,
            production_workload_simulation={
                "concurrent_predictions": 50,
                "average_sequence_length": 200,
                "peak_throughput_requirements": 1000  # predictions per second
            }
        )
        
        assert "throughput_requirements_met" in production_validation
        assert "latency_sla_compliance" in production_validation
        assert "memory_usage_within_limits" in production_validation
        assert "accuracy_maintained_under_load" in production_validation
        assert "fallback_mechanism_tested" in production_validation
        
        # Test monitoring setup for quantized transformers
        monitoring_setup = deployment_manager.setup_quantized_transformer_monitoring(
            deployed_models=quantized_models,
            monitoring_metrics=[
                "inference_latency", "memory_usage", "accuracy_drift",
                "attention_pattern_stability", "quantization_degradation"
            ]
        )
        
        assert "performance_monitoring_config" in monitoring_setup
        assert "accuracy_monitoring_config" in monitoring_setup
        assert "resource_monitoring_config" in monitoring_setup
        assert "alerting_thresholds" in monitoring_setup