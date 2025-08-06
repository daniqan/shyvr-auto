"""
Comprehensive TDD Tests for Phase 3.2.5 Dynamic Resource Allocation for Transformer Models

This module creates comprehensive failing tests for dynamic resource allocation in GCP Cloud Run
deployment environments. These tests are designed to FAIL initially as the implementation
does not exist yet, following strict TDD methodology.

Test Categories:
1. Dynamic Resource Scaling Tests
2. Transformer-Specific Resource Management
3. Multi-Model Resource Balancing
4. Auto-scaling Policy Tests
5. Resource Optimization Tests
6. Performance Under Load Tests
7. Cost Optimization Tests
8. Memory Management Tests
9. CPU Allocation Tests
10. Container Resource Limits

All tests follow TDD principles:
- Tests are written BEFORE implementation
- Tests define expected behavior precisely
- No production mocks - use real test data and components
- Comprehensive edge case and error handling coverage
- Performance requirements embedded in tests
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime, timedelta

# These imports will FAIL initially as the components don't exist yet
# This is EXPECTED behavior for TDD - tests define what needs to be built
try:
    from src.deploy.dynamic_resource_allocator import (
        DynamicResourceAllocator,
        TransformerResourceManager,
        ResourceScalingPolicy,
        ResourceMetricsCollector,
        AutoScalingController,
        ResourceOptimizer,
        MemoryManager,
        CPUManager,
        ContainerResourceManager
    )
except ImportError:
    # Expected during TDD phase - these will be implemented based on these tests
    pass

try:
    from src.monitoring.resource_monitoring import (
        ResourceUsageMonitor,
        PerformanceMetricsCollector,
        ResourceAlertManager
    )
except ImportError:
    # Expected during TDD phase
    pass


class TestDynamicResourceAllocator:
    """
    Test Suite for Dynamic Resource Allocation System
    
    Requirements Tested:
    - Dynamic scaling based on model type and load
    - Resource allocation optimization for transformer inference
    - Multi-model resource balancing
    - Cost-aware resource management
    - Performance requirements under varying loads
    """
    
    @pytest.fixture
    def resource_allocator(self):
        """Fixture for DynamicResourceAllocator - will fail until implemented"""
        return DynamicResourceAllocator(
            project_id="test-project",
            region="us-central1",
            service_name="shyvr-rlte"
        )
    
    @pytest.fixture
    def sample_resource_config(self):
        """Sample resource configuration for testing"""
        return {
            "transformer_models": {
                "itransformer": {
                    "base_memory": "4Gi",
                    "base_cpu": 2,
                    "max_memory": "12Gi",
                    "max_cpu": 8,
                    "scaling_factor": 1.5
                },
                "patchtst": {
                    "base_memory": "3Gi", 
                    "base_cpu": 2,
                    "max_memory": "10Gi",
                    "max_cpu": 6,
                    "scaling_factor": 1.3
                },
                "timesmixer": {
                    "base_memory": "5Gi",
                    "base_cpu": 3,
                    "max_memory": "14Gi", 
                    "max_cpu": 10,
                    "scaling_factor": 1.7
                },
                "timesfm": {
                    "base_memory": "6Gi",
                    "base_cpu": 4,
                    "max_memory": "16Gi",
                    "max_cpu": 12,
                    "scaling_factor": 2.0
                }
            },
            "performance_targets": {
                "max_inference_latency_ms": 100,
                "min_throughput_requests_per_second": 50,
                "memory_utilization_target": 0.85,
                "cpu_utilization_target": 0.80
            },
            "cost_constraints": {
                "max_hourly_cost_usd": 50.0,
                "cost_optimization_enabled": True,
                "prefer_cpu_over_memory": True
            }
        }
    
    @pytest.fixture
    def mock_load_scenarios(self):
        """Mock load scenarios for testing"""
        return {
            "low_load": {
                "requests_per_second": 10,
                "concurrent_models": 1,
                "avg_sequence_length": 100,
                "complexity_factor": 0.5
            },
            "medium_load": {
                "requests_per_second": 50,
                "concurrent_models": 2,
                "avg_sequence_length": 200,
                "complexity_factor": 1.0
            },
            "high_load": {
                "requests_per_second": 150,
                "concurrent_models": 4,
                "avg_sequence_length": 500,
                "complexity_factor": 2.0
            },
            "peak_load": {
                "requests_per_second": 300,
                "concurrent_models": 4,
                "avg_sequence_length": 800,
                "complexity_factor": 3.5
            }
        }
    
    def test_initialize_resource_allocator(self, resource_allocator):
        """Test DynamicResourceAllocator initialization"""
        # Test will fail until DynamicResourceAllocator is implemented
        assert resource_allocator.project_id == "test-project"
        assert resource_allocator.region == "us-central1"
        assert resource_allocator.service_name == "shyvr-rlte"
        assert hasattr(resource_allocator, 'transformer_resource_manager')
        assert hasattr(resource_allocator, 'scaling_policies')
        assert hasattr(resource_allocator, 'metrics_collector')
    
    def test_calculate_base_resources_for_transformer_models(
        self, resource_allocator, sample_resource_config
    ):
        """Test calculating base resource requirements for different transformer models"""
        # Test different transformer models have appropriate base resources
        for model_type in ["itransformer", "patchtst", "timesmixer", "timesfm"]:
            base_resources = resource_allocator.calculate_base_resources(
                model_type=model_type,
                config=sample_resource_config
            )
            
            expected_config = sample_resource_config["transformer_models"][model_type]
            assert base_resources["memory"] == expected_config["base_memory"]
            assert base_resources["cpu"] == expected_config["base_cpu"]
            assert base_resources["scaling_factor"] == expected_config["scaling_factor"]
            
            # Validate resource values are within reasonable bounds
            memory_gb = int(base_resources["memory"].rstrip("Gi"))
            assert 2 <= memory_gb <= 16, f"Memory {memory_gb}Gi outside reasonable range"
            assert 1 <= base_resources["cpu"] <= 12, f"CPU {base_resources['cpu']} outside reasonable range"
    
    def test_dynamic_scaling_based_on_load(
        self, resource_allocator, sample_resource_config, mock_load_scenarios
    ):
        """Test dynamic resource scaling based on different load scenarios"""
        for scenario_name, load_data in mock_load_scenarios.items():
            scaled_resources = resource_allocator.calculate_scaled_resources(
                model_type="itransformer",
                load_metrics=load_data,
                config=sample_resource_config
            )
            
            # Validate scaling behavior
            base_memory = 4  # 4Gi for iTransformer
            base_cpu = 2
            
            if scenario_name == "low_load":
                # Should use base resources or slightly above
                memory_gb = int(scaled_resources["memory"].rstrip("Gi"))
                assert memory_gb <= base_memory + 1
                assert scaled_resources["cpu"] <= base_cpu + 1
                
            elif scenario_name == "high_load":
                # Should scale significantly
                memory_gb = int(scaled_resources["memory"].rstrip("Gi"))
                assert memory_gb >= base_memory + 2
                assert scaled_resources["cpu"] >= base_cpu + 2
                
            elif scenario_name == "peak_load":
                # Should approach maximum limits
                memory_gb = int(scaled_resources["memory"].rstrip("Gi"))
                max_memory = int(sample_resource_config["transformer_models"]["itransformer"]["max_memory"].rstrip("Gi"))
                assert memory_gb >= max_memory * 0.8  # At least 80% of max
    
    def test_multi_model_resource_balancing(
        self, resource_allocator, sample_resource_config
    ):
        """Test resource balancing across multiple concurrent transformer models"""
        active_models = [
            {"type": "itransformer", "priority": 1, "load_factor": 0.8},
            {"type": "patchtst", "priority": 2, "load_factor": 0.6}, 
            {"type": "timesmixer", "priority": 1, "load_factor": 0.9}
        ]
        
        total_budget = {"memory": "24Gi", "cpu": 16}
        
        allocation = resource_allocator.balance_resources_across_models(
            active_models=active_models,
            total_budget=total_budget,
            config=sample_resource_config
        )
        
        # Validate allocation results
        assert len(allocation) == 3
        
        total_allocated_memory = 0
        total_allocated_cpu = 0
        
        for model_allocation in allocation.values():
            memory_gb = int(model_allocation["memory"].rstrip("Gi"))
            total_allocated_memory += memory_gb
            total_allocated_cpu += model_allocation["cpu"]
        
        # Should not exceed total budget
        assert total_allocated_memory <= 24
        assert total_allocated_cpu <= 16
        
        # High priority and high load models should get more resources
        timesmixer_allocation = allocation["timesmixer"]
        patchtst_allocation = allocation["patchtst"]
        
        timesmixer_memory = int(timesmixer_allocation["memory"].rstrip("Gi"))
        patchtst_memory = int(patchtst_allocation["memory"].rstrip("Gi"))
        
        assert timesmixer_memory >= patchtst_memory  # Higher load should get more
    
    def test_auto_scaling_policies(self, resource_allocator):
        """Test auto-scaling policy creation and enforcement"""
        policy_config = {
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "scale_up_cooldown_seconds": 180,
            "scale_down_cooldown_seconds": 300,
            "max_scale_factor": 3.0,
            "min_scale_factor": 0.5
        }
        
        scaling_policy = resource_allocator.create_scaling_policy(
            model_type="itransformer",
            policy_config=policy_config
        )
        
        # Test scaling decisions
        current_metrics = {
            "cpu_utilization": 0.85,
            "memory_utilization": 0.75,
            "request_queue_length": 50,
            "avg_response_time_ms": 120
        }
        
        current_resources = {"memory": "4Gi", "cpu": 2}
        
        scaling_decision = scaling_policy.evaluate_scaling(
            current_metrics=current_metrics,
            current_resources=current_resources
        )
        
        # Should recommend scaling up due to high CPU utilization
        assert scaling_decision["action"] == "scale_up"
        assert scaling_decision["target_memory"] > current_resources["memory"]
        assert scaling_decision["target_cpu"] > current_resources["cpu"]
        assert scaling_decision["confidence"] >= 0.7
    
    def test_resource_optimization_for_cost_efficiency(
        self, resource_allocator, sample_resource_config
    ):
        """Test resource optimization focused on cost efficiency"""
        performance_requirements = {
            "max_latency_ms": 100,
            "min_throughput_rps": 50,
            "max_memory_usage_pct": 85
        }
        
        cost_constraints = {
            "max_hourly_cost": 25.0,
            "cost_per_gb_memory_hour": 0.18,
            "cost_per_cpu_hour": 0.05
        }
        
        optimized_config = resource_allocator.optimize_for_cost(
            model_type="patchtst",
            performance_requirements=performance_requirements,
            cost_constraints=cost_constraints,
            config=sample_resource_config
        )
        
        # Validate optimization results
        assert "recommended_memory" in optimized_config
        assert "recommended_cpu" in optimized_config
        assert "estimated_hourly_cost" in optimized_config
        assert "performance_score" in optimized_config
        
        # Cost should be within constraints
        assert optimized_config["estimated_hourly_cost"] <= cost_constraints["max_hourly_cost"]
        
        # Performance score should be acceptable (>0.8)
        assert optimized_config["performance_score"] >= 0.8
        
        # Should prefer CPU over memory when cost_optimization_enabled
        memory_gb = int(optimized_config["recommended_memory"].rstrip("Gi"))
        cpu_count = optimized_config["recommended_cpu"]
        
        # CPU-to-memory ratio should be higher for cost optimization
        cpu_memory_ratio = cpu_count / memory_gb
        assert cpu_memory_ratio >= 0.4  # At least 0.4 CPU per GB memory
    
    def test_performance_under_varying_loads(
        self, resource_allocator, mock_load_scenarios
    ):
        """Test system performance under varying load conditions"""
        performance_results = {}
        
        for scenario, load_data in mock_load_scenarios.items():
            start_time = time.time()
            
            # Simulate resource allocation under load
            allocation_result = resource_allocator.allocate_resources_for_load(
                load_scenario=load_data,
                performance_targets={
                    "max_latency_ms": 100,
                    "min_throughput_rps": load_data["requests_per_second"] * 0.95
                }
            )
            
            allocation_time = (time.time() - start_time) * 1000  # Convert to ms
            
            performance_results[scenario] = {
                "allocation_time_ms": allocation_time,
                "resource_efficiency": allocation_result.get("efficiency_score", 0),
                "cost_efficiency": allocation_result.get("cost_score", 0),
                "allocation_success": allocation_result.get("success", False)
            }
        
        # Validate performance requirements
        for scenario, results in performance_results.items():
            # Resource allocation should complete quickly
            assert results["allocation_time_ms"] < 500, f"Allocation too slow for {scenario}"
            
            # Should successfully allocate resources
            assert results["allocation_success"], f"Failed to allocate resources for {scenario}"
            
            # Resource efficiency should be reasonable
            assert results["resource_efficiency"] >= 0.7, f"Poor efficiency for {scenario}"
    
    def test_memory_management_optimization(self, resource_allocator):
        """Test memory management and optimization strategies"""
        memory_config = {
            "enable_memory_optimization": True,
            "memory_pressure_threshold": 0.85,
            "gc_trigger_threshold": 0.80,
            "model_cache_limit_gb": 8,
            "enable_model_offloading": True
        }
        
        memory_manager = resource_allocator.get_memory_manager(memory_config)
        
        # Test memory pressure detection
        current_usage = {
            "total_memory_gb": 8,
            "used_memory_gb": 7.2,  # 90% usage
            "model_cache_gb": 4.5,
            "active_models": 3
        }
        
        memory_action = memory_manager.evaluate_memory_pressure(current_usage)
        
        # Should detect high memory pressure and recommend action
        assert memory_action["pressure_level"] == "high"
        assert memory_action["recommended_action"] in ["offload_models", "reduce_cache", "scale_up"]
        assert memory_action["urgency"] >= 0.8
        
        # Test memory optimization strategies
        optimization = memory_manager.optimize_memory_usage(current_usage)
        
        assert "target_memory_usage_gb" in optimization
        assert "actions" in optimization
        assert len(optimization["actions"]) > 0
        
        # Target usage should be below pressure threshold
        target_usage_pct = optimization["target_memory_usage_gb"] / current_usage["total_memory_gb"]
        assert target_usage_pct <= memory_config["memory_pressure_threshold"]
    
    def test_cpu_allocation_strategies(self, resource_allocator):
        """Test CPU allocation strategies for transformer inference"""
        cpu_scenarios = [
            {
                "name": "attention_heavy", 
                "attention_layers": 12,
                "sequence_length": 512,
                "batch_size": 8,
                "expected_cpu_factor": 1.5
            },
            {
                "name": "memory_bound",
                "attention_layers": 6, 
                "sequence_length": 1024,
                "batch_size": 4,
                "expected_cpu_factor": 1.0
            },
            {
                "name": "compute_intensive",
                "attention_layers": 16,
                "sequence_length": 256, 
                "batch_size": 16,
                "expected_cpu_factor": 2.0
            }
        ]
        
        for scenario in cpu_scenarios:
            cpu_allocation = resource_allocator.calculate_optimal_cpu_allocation(
                workload_characteristics=scenario,
                base_cpu=2,
                max_cpu=8
            )
            
            assert "recommended_cpu" in cpu_allocation
            assert "cpu_utilization_target" in cpu_allocation
            assert "threading_strategy" in cpu_allocation
            
            # Validate CPU allocation makes sense
            recommended_cpu = cpu_allocation["recommended_cpu"]
            assert 2 <= recommended_cpu <= 8
            
            # Higher expected factors should get more CPU
            if scenario["expected_cpu_factor"] >= 2.0:
                assert recommended_cpu >= 4
            elif scenario["expected_cpu_factor"] <= 1.0:
                assert recommended_cpu <= 3
    
    def test_container_resource_limits(self, resource_allocator):
        """Test container resource limit calculation and enforcement"""
        container_specs = [
            {
                "model_type": "itransformer",
                "expected_memory": "6Gi",
                "expected_cpu": 3,
                "max_requests_per_container": 20
            },
            {
                "model_type": "timesfm",
                "expected_memory": "10Gi", 
                "expected_cpu": 6,
                "max_requests_per_container": 15
            }
        ]
        
        for spec in container_specs:
            limits = resource_allocator.calculate_container_limits(
                model_type=spec["model_type"],
                expected_resources={
                    "memory": spec["expected_memory"],
                    "cpu": spec["expected_cpu"]
                },
                safety_margin=0.2  # 20% safety margin
            )
            
            assert "memory_limit" in limits
            assert "cpu_limit" in limits
            assert "memory_request" in limits
            assert "cpu_request" in limits
            
            # Limits should be higher than requests due to safety margin
            request_memory_gb = int(limits["memory_request"].rstrip("Gi"))
            limit_memory_gb = int(limits["memory_limit"].rstrip("Gi"))
            assert limit_memory_gb > request_memory_gb
            
            # Requests should match expected resources
            expected_memory_gb = int(spec["expected_memory"].rstrip("Gi"))
            assert abs(request_memory_gb - expected_memory_gb) <= 1
    
    def test_resource_allocation_edge_cases(self, resource_allocator):
        """Test resource allocation under edge cases and error conditions"""
        edge_cases = [
            {
                "name": "zero_load",
                "requests_per_second": 0,
                "expected_behavior": "minimum_resources"
            },
            {
                "name": "extreme_load", 
                "requests_per_second": 1000,
                "expected_behavior": "maximum_resources"
            },
            {
                "name": "memory_constrained",
                "available_memory": "2Gi",
                "model_type": "timesfm",  # Normally needs 6Gi
                "expected_behavior": "allocation_failure"
            },
            {
                "name": "cpu_constrained",
                "available_cpu": 1,
                "concurrent_models": 4,
                "expected_behavior": "resource_balancing"
            }
        ]
        
        for case in edge_cases:
            if case["expected_behavior"] == "allocation_failure":
                with pytest.raises(Exception) as exc_info:
                    resource_allocator.allocate_resources(
                        constraints=case,
                        strict_mode=True
                    )
                assert "insufficient" in str(exc_info.value).lower()
                
            else:
                result = resource_allocator.allocate_resources(
                    constraints=case,
                    strict_mode=False
                )
                
                if case["expected_behavior"] == "minimum_resources":
                    # Should allocate minimal but functional resources
                    memory_gb = int(result["memory"].rstrip("Gi"))
                    assert memory_gb <= 4  # Minimal memory allocation
                    
                elif case["expected_behavior"] == "maximum_resources":
                    # Should allocate maximum available resources
                    memory_gb = int(result["memory"].rstrip("Gi"))
                    assert memory_gb >= 12  # High memory allocation
    
    @pytest.mark.asyncio
    async def test_real_time_resource_adjustment(self, resource_allocator):
        """Test real-time resource adjustment based on metrics"""
        # Simulate real-time metrics feed
        metrics_feed = [
            {"timestamp": time.time(), "cpu": 0.3, "memory": 0.4, "latency": 50},
            {"timestamp": time.time() + 1, "cpu": 0.6, "memory": 0.7, "latency": 80},
            {"timestamp": time.time() + 2, "cpu": 0.9, "memory": 0.85, "latency": 150},  # Stress
            {"timestamp": time.time() + 3, "cpu": 0.95, "memory": 0.92, "latency": 200},  # Critical
        ]
        
        adjustments = []
        
        for metrics in metrics_feed:
            adjustment = await resource_allocator.evaluate_real_time_adjustment(
                current_metrics=metrics,
                adjustment_sensitivity=0.8
            )
            
            if adjustment:
                adjustments.append(adjustment)
        
        # Should have triggered adjustments for high load periods
        assert len(adjustments) >= 2
        
        # Last adjustment should be significant due to critical metrics
        critical_adjustment = adjustments[-1]
        assert critical_adjustment["urgency"] == "critical"
        assert critical_adjustment["scale_factor"] >= 1.5
        assert critical_adjustment["estimated_application_time_seconds"] <= 60


class TestTransformerResourceManager:
    """
    Test Suite for Transformer-Specific Resource Management
    
    Tests transformer model-specific resource requirements, optimization strategies,
    and performance characteristics under various deployment scenarios.
    """
    
    @pytest.fixture
    def transformer_manager(self):
        """Fixture for TransformerResourceManager"""
        return TransformerResourceManager(
            supported_models=["itransformer", "patchtst", "timesmixer", "timesfm"],
            resource_database_path="/tmp/test_resources.db"
        )
    
    def test_model_specific_resource_profiles(self, transformer_manager):
        """Test resource profiles for different transformer architectures"""
        for model_type in ["itransformer", "patchtst", "timesmixer", "timesfm"]:
            profile = transformer_manager.get_resource_profile(model_type)
            
            assert "memory_requirements" in profile
            assert "cpu_requirements" in profile  
            assert "attention_complexity" in profile
            assert "inference_characteristics" in profile
            
            # Validate model-specific characteristics
            if model_type == "timesfm":
                # TimesFM should have highest memory requirements
                memory_base = int(profile["memory_requirements"]["base"].rstrip("Gi"))
                assert memory_base >= 6
                
            elif model_type == "patchtst":
                # PatchTST should be more memory efficient
                memory_base = int(profile["memory_requirements"]["base"].rstrip("Gi"))
                assert memory_base <= 4
            
            # All models should have reasonable attention complexity scores
            assert 0.5 <= profile["attention_complexity"] <= 3.0
    
    def test_attention_based_resource_scaling(self, transformer_manager):
        """Test resource scaling based on attention mechanism characteristics"""
        attention_scenarios = [
            {
                "sequence_length": 100,
                "attention_heads": 8,
                "layers": 6,
                "expected_scale": 1.0
            },
            {
                "sequence_length": 500,
                "attention_heads": 12,
                "layers": 12,
                "expected_scale": 2.5
            },
            {
                "sequence_length": 1000,
                "attention_heads": 16,
                "layers": 18,
                "expected_scale": 4.0
            }
        ]
        
        for scenario in attention_scenarios:
            scale_factor = transformer_manager.calculate_attention_scale_factor(
                sequence_length=scenario["sequence_length"],
                attention_heads=scenario["attention_heads"],
                num_layers=scenario["layers"]
            )
            
            # Scale factor should be reasonable and match expected magnitude
            assert 0.5 <= scale_factor <= 5.0
            
            # Longer sequences and more attention should increase scaling
            if scenario["expected_scale"] >= 3.0:
                assert scale_factor >= 2.5
            elif scenario["expected_scale"] <= 1.5:
                assert scale_factor <= 2.0
    
    def test_model_cache_resource_management(self, transformer_manager):
        """Test resource management for model caching strategies"""
        cache_config = {
            "max_cache_size_gb": 16,
            "cache_eviction_policy": "lru",
            "preload_models": ["itransformer", "patchtst"],
            "cache_warmup_enabled": True
        }
        
        cache_manager = transformer_manager.get_cache_manager(cache_config)
        
        # Test cache resource allocation
        allocation = cache_manager.allocate_cache_resources(
            active_models=["itransformer", "patchtst", "timesmixer"],
            priority_weights={"itransformer": 0.5, "patchtst": 0.3, "timesmixer": 0.2}
        )
        
        assert "per_model_allocation" in allocation
        assert "total_allocated_gb" in allocation
        assert "cache_hit_ratio_estimate" in allocation
        
        # Total allocation should not exceed limit
        assert allocation["total_allocated_gb"] <= cache_config["max_cache_size_gb"]
        
        # Higher priority models should get more cache space
        itransformer_cache = allocation["per_model_allocation"]["itransformer"]
        timesmixer_cache = allocation["per_model_allocation"]["timesmixer"]
        assert itransformer_cache >= timesmixer_cache
    
    def test_inference_pipeline_resource_optimization(self, transformer_manager):
        """Test resource optimization for transformer inference pipelines"""
        pipeline_configs = [
            {
                "batch_size": 1,
                "enable_torch_compile": True,
                "use_flash_attention": True,
                "expected_speedup": 1.3
            },
            {
                "batch_size": 8,
                "enable_torch_compile": False,
                "use_flash_attention": False,
                "expected_speedup": 0.8
            },
            {
                "batch_size": 16,
                "enable_torch_compile": True,
                "use_flash_attention": True,
                "expected_speedup": 2.0
            }
        ]
        
        for config in pipeline_configs:
            optimization = transformer_manager.optimize_inference_pipeline(
                model_type="itransformer",
                pipeline_config=config,
                target_latency_ms=100
            )
            
            assert "optimized_batch_size" in optimization
            assert "memory_allocation" in optimization
            assert "cpu_allocation" in optimization
            assert "expected_performance_improvement" in optimization
            
            # Performance improvement should align with expectations
            improvement = optimization["expected_performance_improvement"]
            expected = config["expected_speedup"]
            assert abs(improvement - expected) <= 0.3  # Allow some variance


class TestResourceScalingPolicy:
    """
    Test Suite for Resource Scaling Policies
    
    Tests scaling policy creation, evaluation, and enforcement for transformer
    deployments under various load and performance conditions.
    """
    
    @pytest.fixture
    def scaling_policy(self):
        """Fixture for ResourceScalingPolicy"""
        return ResourceScalingPolicy(
            policy_name="transformer_scaling_v1",
            model_types=["itransformer", "patchtst", "timesmixer", "timesfm"]
        )
    
    def test_scaling_policy_creation(self, scaling_policy):
        """Test creation and configuration of scaling policies"""
        policy_config = {
            "cpu_scale_up_threshold": 0.8,
            "memory_scale_up_threshold": 0.85,
            "latency_scale_up_threshold_ms": 100,
            "cpu_scale_down_threshold": 0.3,
            "memory_scale_down_threshold": 0.4,
            "cooldown_periods": {
                "scale_up": 180,
                "scale_down": 300
            },
            "scaling_factors": {
                "aggressive": 2.0,
                "moderate": 1.5,
                "conservative": 1.2
            }
        }
        
        scaling_policy.configure(policy_config)
        
        # Validate policy configuration
        assert scaling_policy.cpu_scale_up_threshold == 0.8
        assert scaling_policy.memory_scale_up_threshold == 0.85
        assert scaling_policy.latency_scale_up_threshold_ms == 100
        
        # Test policy validation
        validation_result = scaling_policy.validate_configuration()
        assert validation_result["is_valid"] == True
        assert len(validation_result["warnings"]) == 0
    
    def test_scaling_decision_evaluation(self, scaling_policy):
        """Test scaling decision evaluation under various conditions"""
        scaling_scenarios = [
            {
                "name": "scale_up_cpu",
                "metrics": {"cpu": 0.85, "memory": 0.6, "latency": 80},
                "expected_decision": "scale_up",
                "expected_factor": "moderate"
            },
            {
                "name": "scale_up_memory",
                "metrics": {"cpu": 0.6, "memory": 0.90, "latency": 70},
                "expected_decision": "scale_up", 
                "expected_factor": "aggressive"
            },
            {
                "name": "scale_up_latency",
                "metrics": {"cpu": 0.7, "memory": 0.75, "latency": 150},
                "expected_decision": "scale_up",
                "expected_factor": "aggressive"
            },
            {
                "name": "scale_down",
                "metrics": {"cpu": 0.25, "memory": 0.35, "latency": 40},
                "expected_decision": "scale_down",
                "expected_factor": "conservative"
            },
            {
                "name": "no_action",
                "metrics": {"cpu": 0.6, "memory": 0.65, "latency": 70},
                "expected_decision": "no_action",
                "expected_factor": None
            }
        ]
        
        for scenario in scaling_scenarios:
            decision = scaling_policy.evaluate_scaling_decision(
                current_metrics=scenario["metrics"],
                current_timestamp=time.time()
            )
            
            assert decision["action"] == scenario["expected_decision"]
            
            if scenario["expected_factor"]:
                assert decision["scaling_factor"] == scenario["expected_factor"]
                assert decision["confidence"] >= 0.7
    
    def test_cooldown_period_enforcement(self, scaling_policy):
        """Test cooldown period enforcement for scaling operations"""
        initial_time = time.time()
        
        # First scaling decision
        decision1 = scaling_policy.evaluate_scaling_decision(
            current_metrics={"cpu": 0.9, "memory": 0.85, "latency": 120},
            current_timestamp=initial_time
        )
        assert decision1["action"] == "scale_up"
        
        # Record the scaling action
        scaling_policy.record_scaling_action(
            action="scale_up",
            timestamp=initial_time,
            resources_before={"memory": "4Gi", "cpu": 2},
            resources_after={"memory": "6Gi", "cpu": 3}
        )
        
        # Second scaling decision within cooldown period (should be blocked)
        decision2 = scaling_policy.evaluate_scaling_decision(
            current_metrics={"cpu": 0.95, "memory": 0.90, "latency": 140},
            current_timestamp=initial_time + 120  # 2 minutes later, within 3-minute cooldown
        )
        assert decision2["action"] == "no_action"
        assert "cooldown" in decision2["reason"]
        
        # Third scaling decision after cooldown period
        decision3 = scaling_policy.evaluate_scaling_decision(
            current_metrics={"cpu": 0.95, "memory": 0.90, "latency": 140},
            current_timestamp=initial_time + 200  # 3+ minutes later, outside cooldown
        )
        assert decision3["action"] == "scale_up"
    
    def test_model_specific_scaling_policies(self, scaling_policy):
        """Test model-specific scaling policy customization"""
        model_policies = {
            "timesfm": {
                "memory_scale_up_threshold": 0.75,  # More aggressive due to high memory usage
                "scaling_factors": {"moderate": 1.8, "aggressive": 2.5}
            },
            "patchtst": {
                "cpu_scale_up_threshold": 0.75,  # More CPU-sensitive
                "scaling_factors": {"moderate": 1.3, "aggressive": 1.6}
            }
        }
        
        scaling_policy.configure_model_specific_policies(model_policies)
        
        # Test TimesFM-specific scaling
        timesfm_decision = scaling_policy.evaluate_scaling_decision(
            current_metrics={"cpu": 0.6, "memory": 0.80, "latency": 90},
            current_timestamp=time.time(),
            model_type="timesfm"
        )
        
        # Should scale due to lower memory threshold for TimesFM
        assert timesfm_decision["action"] == "scale_up"
        
        # Test PatchTST-specific scaling
        patchtst_decision = scaling_policy.evaluate_scaling_decision(
            current_metrics={"cpu": 0.80, "memory": 0.6, "latency": 90},
            current_timestamp=time.time(),
            model_type="patchtst"
        )
        
        # Should scale due to lower CPU threshold for PatchTST
        assert patchtst_decision["action"] == "scale_up"


# Performance and Load Testing
class TestResourceAllocationPerformance:
    """
    Test Suite for Resource Allocation Performance Under Load
    
    Tests system performance and resource allocation efficiency under various
    load conditions and stress scenarios.
    """
    
    @pytest.fixture
    def performance_tester(self):
        """Fixture for performance testing utilities"""
        from src.deploy.performance_tester import ResourceAllocationPerformanceTester
        return ResourceAllocationPerformanceTester()
    
    @pytest.mark.asyncio
    async def test_allocation_latency_under_load(self, performance_tester):
        """Test resource allocation latency under various load conditions"""
        load_scenarios = [10, 50, 100, 200, 500]  # Requests per second
        latency_results = {}
        
        for rps in load_scenarios:
            start_time = time.time()
            
            # Simulate concurrent allocation requests
            allocation_tasks = []
            for _ in range(min(rps, 100)):  # Cap concurrent requests for testing
                task = asyncio.create_task(
                    performance_tester.simulate_allocation_request(
                        model_type="itransformer",
                        load_factor=rps / 100.0
                    )
                )
                allocation_tasks.append(task)
            
            results = await asyncio.gather(*allocation_tasks)
            total_time = time.time() - start_time
            
            # Calculate performance metrics
            successful_allocations = sum(1 for r in results if r["success"])
            avg_latency = sum(r["latency_ms"] for r in results) / len(results)
            
            latency_results[rps] = {
                "total_time": total_time,
                "success_rate": successful_allocations / len(results),
                "avg_latency_ms": avg_latency,
                "throughput_rps": successful_allocations / total_time
            }
        
        # Validate performance requirements
        for rps, metrics in latency_results.items():
            # Success rate should remain high under load
            assert metrics["success_rate"] >= 0.95, f"Low success rate at {rps} RPS"
            
            # Average latency should remain reasonable
            if rps <= 100:
                assert metrics["avg_latency_ms"] <= 100, f"High latency at {rps} RPS"
            else:
                assert metrics["avg_latency_ms"] <= 200, f"Very high latency at {rps} RPS"
    
    @pytest.mark.asyncio
    async def test_memory_allocation_efficiency(self, performance_tester):
        """Test memory allocation efficiency and optimization"""
        memory_scenarios = [
            {"total_memory": "8Gi", "models": 1, "expected_efficiency": 0.85},
            {"total_memory": "16Gi", "models": 2, "expected_efficiency": 0.80},
            {"total_memory": "32Gi", "models": 4, "expected_efficiency": 0.75}
        ]
        
        for scenario in memory_scenarios:
            efficiency_result = await performance_tester.test_memory_efficiency(
                total_memory=scenario["total_memory"],
                concurrent_models=scenario["models"],
                test_duration_seconds=30
            )
            
            assert efficiency_result["memory_utilization"] >= scenario["expected_efficiency"]
            assert efficiency_result["allocation_failures"] == 0
            assert efficiency_result["memory_fragmentation"] <= 0.15  # Max 15% fragmentation
    
    def test_cost_optimization_performance(self, performance_tester):
        """Test cost optimization algorithms performance"""
        cost_scenarios = [
            {
                "budget": 10.0,  # $10/hour
                "models": ["itransformer"],
                "performance_req": {"latency_ms": 100, "throughput_rps": 20}
            },
            {
                "budget": 25.0,  # $25/hour
                "models": ["itransformer", "patchtst"],
                "performance_req": {"latency_ms": 100, "throughput_rps": 50}
            },
            {
                "budget": 50.0,  # $50/hour
                "models": ["itransformer", "patchtst", "timesmixer", "timesfm"],
                "performance_req": {"latency_ms": 80, "throughput_rps": 100}
            }
        ]
        
        for scenario in cost_scenarios:
            start_time = time.time()
            
            optimization_result = performance_tester.optimize_for_cost(
                budget_per_hour=scenario["budget"],
                active_models=scenario["models"],
                performance_requirements=scenario["performance_req"]
            )
            
            optimization_time = time.time() - start_time
            
            # Optimization should complete quickly
            assert optimization_time <= 2.0, "Cost optimization too slow"
            
            # Should stay within budget
            assert optimization_result["estimated_cost_per_hour"] <= scenario["budget"]
            
            # Should meet performance requirements
            assert optimization_result["predicted_latency_ms"] <= scenario["performance_req"]["latency_ms"] + 20
            assert optimization_result["predicted_throughput_rps"] >= scenario["performance_req"]["throughput_rps"] * 0.9


# Integration Tests
class TestDynamicResourceAllocationIntegration:
    """
    Integration Test Suite for Dynamic Resource Allocation
    
    Tests integration with GCP Cloud Run, monitoring systems, and existing
    transformer deployment infrastructure.
    """
    
    @pytest.fixture
    def integration_setup(self):
        """Setup integration test environment"""
        return {
            "project_id": "test-shyvr-rlte",
            "region": "us-central1", 
            "service_name": "shyvr-rlte-transformers",
            "monitoring_enabled": True,
            "alert_channels": ["test-channel"]
        }
    
    @pytest.mark.asyncio
    async def test_gcp_cloud_run_integration(self, integration_setup):
        """Test integration with GCP Cloud Run service management"""
        from src.deploy.gcp_integration import GCPCloudRunIntegrator
        
        integrator = GCPCloudRunIntegrator(**integration_setup)
        
        # Test service configuration update
        new_config = {
            "memory": "8Gi",
            "cpu": 4,
            "max_instances": 10,
            "min_instances": 1,
            "concurrency": 15
        }
        
        update_result = await integrator.update_service_configuration(
            configuration=new_config,
            deployment_strategy="blue_green"
        )
        
        assert update_result["success"] == True
        assert update_result["deployment_id"] is not None
        assert update_result["rollout_status"] == "in_progress"
        
        # Test configuration verification
        verification = await integrator.verify_configuration_applied(
            deployment_id=update_result["deployment_id"],
            timeout_seconds=300
        )
        
        assert verification["configuration_applied"] == True
        assert verification["health_check_passed"] == True
    
    @pytest.mark.asyncio 
    async def test_monitoring_integration(self, integration_setup):
        """Test integration with monitoring and alerting systems"""
        from src.monitoring.resource_monitoring_integration import ResourceMonitoringIntegrator
        
        monitor = ResourceMonitoringIntegrator(**integration_setup)
        
        # Test custom metrics creation for resource allocation
        metrics_config = {
            "resource_allocation_latency": {
                "type": "gauge",
                "description": "Time taken to allocate resources",
                "unit": "ms"
            },
            "resource_utilization_efficiency": {
                "type": "gauge", 
                "description": "Resource utilization efficiency score",
                "unit": "ratio"
            },
            "cost_optimization_score": {
                "type": "gauge",
                "description": "Cost optimization effectiveness",
                "unit": "score"
            }
        }
        
        metrics_result = await monitor.create_custom_metrics(metrics_config)
        assert metrics_result["success"] == True
        assert len(metrics_result["created_metrics"]) == 3
        
        # Test alert policy creation
        alert_policies = [
            {
                "name": "high_allocation_latency",
                "condition": "resource_allocation_latency > 1000",
                "notification_channels": integration_setup["alert_channels"]
            },
            {
                "name": "low_resource_efficiency",
                "condition": "resource_utilization_efficiency < 0.7",
                "notification_channels": integration_setup["alert_channels"]
            }
        ]
        
        alert_result = await monitor.create_alert_policies(alert_policies)
        assert alert_result["success"] == True
        assert len(alert_result["created_policies"]) == 2
    
    @pytest.mark.asyncio
    async def test_end_to_end_resource_allocation_flow(self, integration_setup):
        """Test complete end-to-end resource allocation flow"""
        from src.deploy.dynamic_resource_allocator import DynamicResourceAllocator
        
        allocator = DynamicResourceAllocator(**integration_setup)
        
        # Test complete allocation flow
        allocation_request = {
            "model_types": ["itransformer", "patchtst"],
            "expected_load": {
                "requests_per_second": 75,
                "concurrent_users": 20,
                "avg_sequence_length": 300
            },
            "performance_requirements": {
                "max_latency_ms": 100,
                "min_throughput_rps": 70,
                "target_availability": 0.999
            },
            "cost_constraints": {
                "max_hourly_cost": 30.0
            }
        }
        
        # Execute end-to-end allocation
        allocation_result = await allocator.execute_allocation(allocation_request)
        
        # Validate allocation success
        assert allocation_result["success"] == True
        assert allocation_result["deployment_id"] is not None
        assert allocation_result["estimated_cost_per_hour"] <= 30.0
        
        # Validate performance predictions
        perf_predictions = allocation_result["performance_predictions"]
        assert perf_predictions["expected_latency_ms"] <= 100
        assert perf_predictions["expected_throughput_rps"] >= 70
        
        # Test monitoring data collection
        monitoring_data = await allocator.collect_post_allocation_metrics(
            deployment_id=allocation_result["deployment_id"],
            collection_duration_seconds=60
        )
        
        assert "resource_utilization" in monitoring_data
        assert "performance_metrics" in monitoring_data
        assert "cost_metrics" in monitoring_data
        
        # Validate monitoring data quality
        assert monitoring_data["resource_utilization"]["cpu"] > 0
        assert monitoring_data["resource_utilization"]["memory"] > 0
        assert monitoring_data["performance_metrics"]["avg_latency_ms"] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])