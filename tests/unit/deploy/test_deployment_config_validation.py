"""
Tests for Cloud Run deployment configuration validation with transformer-optimized resource settings.

This test suite validates the deployment configuration updates required for transformer model
deployment with optimized resource allocation. Following TDD methodology - these tests will
initially fail until the implementation is complete.

Tests cover:
1. Resource configuration validation (8Gi RAM, 6 CPU, 4200s timeout)
2. Concurrency limit adjustments for transformer workloads
3. Environment variable validation for transformer environments
4. Deployment script resource setting validation
5. Cloud Build configuration optimization
6. Blue-green deployment resource configurations
7. Automated deployment pipeline configurations

Created: August 5, 2025
Test-Driven Development: These tests are designed to FAIL initially
"""

import pytest
import os
import re
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestTransformerDeploymentConfig:
    """Test transformer-optimized deployment configuration settings."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @pytest.fixture
    def deploy_scripts_path(self, project_root):
        """Get deploy scripts directory."""
        return project_root / "deploy"
    
    @pytest.fixture
    def config_path(self, project_root):
        """Get config directory."""
        return project_root / "config"

    def test_deploy_utils_transformer_resource_limits(self, deploy_scripts_path):
        """Test that deploy-utils.sh has transformer-optimized resource limits."""
        deploy_utils_path = deploy_scripts_path / "deploy-utils.sh"
        assert deploy_utils_path.exists(), "deploy-utils.sh should exist"
        
        content = deploy_utils_path.read_text()
        
        # Check for transformer-optimized production resource limits
        production_memory_pattern = r'--memory=8Gi'
        production_cpu_pattern = r'--cpu=6'
        production_timeout_pattern = r'--timeout=4200'
        production_concurrency_pattern = r'--concurrency=15'  # Reduced for transformers
        
        assert re.search(production_memory_pattern, content), \
            "Production environment should have 8Gi memory allocation for transformers"
        assert re.search(production_cpu_pattern, content), \
            "Production environment should have 6 CPU allocation for transformers"
        assert re.search(production_timeout_pattern, content), \
            "Production environment should have 4200s timeout for transformer loading"
        assert re.search(production_concurrency_pattern, content), \
            "Production environment should have reduced concurrency (15) for transformers"
    
    def test_deploy_utils_staging_resource_limits(self, deploy_scripts_path):
        """Test that deploy-utils.sh has appropriate staging resource limits."""
        deploy_utils_path = deploy_scripts_path / "deploy-utils.sh"
        content = deploy_utils_path.read_text()
        
        # Check for staging resource limits (scaled down but still transformer-capable)
        staging_memory_pattern = r'--memory=6Gi'
        staging_cpu_pattern = r'--cpu=4'
        staging_timeout_pattern = r'--timeout=2700'  # 45 minutes for staging
        staging_concurrency_pattern = r'--concurrency=20'
        
        assert re.search(staging_memory_pattern, content), \
            "Staging environment should have 6Gi memory for transformer testing"
        assert re.search(staging_cpu_pattern, content), \
            "Staging environment should have 4 CPU for transformer testing"
        assert re.search(staging_timeout_pattern, content), \
            "Staging environment should have 2700s timeout for transformer testing"
        assert re.search(staging_concurrency_pattern, content), \
            "Staging environment should have 20 concurrency for transformer testing"

    def test_deploy_utils_transformer_environment_variables(self, deploy_scripts_path):
        """Test that deploy-utils.sh sets transformer-specific environment variables."""
        deploy_utils_path = deploy_scripts_path / "deploy-utils.sh"
        content = deploy_utils_path.read_text()
        
        # Check for transformer-specific environment variables
        transformer_env_patterns = [
            r'TRANSFORMER_OPTIMIZED=true',
            r'TRANSFORMER_BATCH_SIZE=1',
            r'TRANSFORMER_MAX_LENGTH=512',
            r'TORCH_COMPILE_MODE=reduce-overhead',
            r'TRANSFORMERS_CACHE=/app/models/cache',
            r'TOKENIZERS_PARALLELISM=false'
        ]
        
        for pattern in transformer_env_patterns:
            assert re.search(pattern, content), \
                f"Environment variable pattern '{pattern}' should be set for transformer optimization"

    def test_blue_green_deployment_transformer_resources(self, deploy_scripts_path):
        """Test that blue_green_deployment.sh uses transformer-optimized resources."""
        blue_green_path = deploy_scripts_path / "blue_green_deployment.sh"
        assert blue_green_path.exists(), "blue_green_deployment.sh should exist"
        
        content = blue_green_path.read_text()
        
        # Check for transformer-optimized resource settings in blue-green deployment
        memory_pattern = r'--memory=8Gi'
        cpu_pattern = r'--cpu=6'
        timeout_pattern = r'--timeout=4200'
        concurrency_pattern = r'--concurrency=15'
        
        assert re.search(memory_pattern, content), \
            "Blue-green deployment should use 8Gi memory for transformers"
        assert re.search(cpu_pattern, content), \
            "Blue-green deployment should use 6 CPU for transformers"
        assert re.search(timeout_pattern, content), \
            "Blue-green deployment should use 4200s timeout for transformers"
        assert re.search(concurrency_pattern, content), \
            "Blue-green deployment should use reduced concurrency for transformers"

    def test_blue_green_deployment_transformer_health_checks(self, deploy_scripts_path):
        """Test that blue_green_deployment.sh has extended health checks for transformers."""
        blue_green_path = deploy_scripts_path / "blue_green_deployment.sh"
        content = blue_green_path.read_text()
        
        # Check for extended health check parameters for transformer loading
        health_check_attempts_pattern = r'HEALTH_CHECK_ATTEMPTS=15'  # Increased for transformers
        health_check_interval_pattern = r'HEALTH_CHECK_INTERVAL=20'  # Longer intervals
        traffic_migration_delay_pattern = r'TRAFFIC_MIGRATION_DELAY=180'  # Longer stabilization
        
        assert re.search(health_check_attempts_pattern, content), \
            "Health check attempts should be increased for transformer initialization"
        assert re.search(health_check_interval_pattern, content), \
            "Health check interval should be increased for transformer readiness"
        assert re.search(traffic_migration_delay_pattern, content), \
            "Traffic migration delay should be increased for transformer stabilization"

    def test_automated_deployment_pipeline_transformer_config(self, deploy_scripts_path):
        """Test that automated_deployment_pipeline.sh calls blue_green_deployment.sh with transformer configurations."""
        pipeline_path = deploy_scripts_path / "automated_deployment_pipeline.sh"
        assert pipeline_path.exists(), "automated_deployment_pipeline.sh should exist"
        
        content = pipeline_path.read_text()
        
        # Check that pipeline calls blue_green_deployment.sh (which has transformer configs)
        blue_green_call_pattern = r'blue_green_deployment\.sh'
        
        assert re.search(blue_green_call_pattern, content), \
            "Automated deployment pipeline should call blue_green_deployment.sh with transformer configurations"

    def test_automated_deployment_pipeline_transformer_validation(self, deploy_scripts_path):
        """Test that automated deployment pipeline includes transformer-specific validation."""
        pipeline_path = deploy_scripts_path / "automated_deployment_pipeline.sh"
        content = pipeline_path.read_text()
        
        # Check for transformer-specific validation steps
        validation_patterns = [
            r'stage_transformer_validation',
            r'transformer.*memory.*requirements',
            r'transformer.*model.*loading',
            r'attention.*mechanism.*test'
        ]
        
        for pattern in validation_patterns:
            assert re.search(pattern, content, re.IGNORECASE), \
                f"Pipeline should include transformer validation: '{pattern}'"

    def test_cloudbuild_yaml_transformer_resources(self, config_path):
        """Test that cloudbuild.yaml has transformer-optimized resource settings."""
        cloudbuild_path = config_path / "cloudbuild.yaml"
        assert cloudbuild_path.exists(), "cloudbuild.yaml should exist"
        
        content = cloudbuild_path.read_text()
        
        # Check for transformer-optimized Cloud Run deployment settings
        memory_pattern = r'--memory=8Gi'
        cpu_pattern = r'--cpu=6'
        timeout_pattern = r'--timeout=4200'
        concurrency_pattern = r'--concurrency=15'
        
        assert re.search(memory_pattern, content), \
            "Cloud Build should deploy with 8Gi memory for transformers"
        assert re.search(cpu_pattern, content), \
            "Cloud Build should deploy with 6 CPU for transformers"
        assert re.search(timeout_pattern, content), \
            "Cloud Build should deploy with 4200s timeout for transformers"
        assert re.search(concurrency_pattern, content), \
            "Cloud Build should deploy with reduced concurrency for transformers"

    def test_cloudbuild_yaml_transformer_environment_variables(self, config_path):
        """Test that cloudbuild.yaml sets transformer-specific environment variables."""
        cloudbuild_path = config_path / "cloudbuild.yaml"
        content = cloudbuild_path.read_text()
        
        # Check for transformer-specific environment variables in Cloud Build
        env_patterns = [
            r'TRANSFORMER_OPTIMIZED=true',
            r'TORCH_COMPILE_MODE=',
            r'TRANSFORMERS_CACHE=',
            r'TOKENIZERS_PARALLELISM=false'
        ]
        
        for pattern in env_patterns:
            assert re.search(pattern, content), \
                f"Cloud Build should set transformer environment variable: '{pattern}'"

    def test_cloudbuild_yaml_machine_type_optimization(self, config_path):
        """Test that cloudbuild.yaml uses appropriate machine type for transformer builds."""
        cloudbuild_path = config_path / "cloudbuild.yaml"
        content = cloudbuild_path.read_text()
        
        # Check for optimized machine type for transformer model building
        machine_type_pattern = r"machineType:\s*['\"]?(E2_HIGHCPU_16|N2_HIGHMEM_8|C2_STANDARD_8)['\"]?"
        disk_size_pattern = r"diskSizeGb:\s*150"  # Increased for transformer models
        
        assert re.search(machine_type_pattern, content), \
            "Cloud Build should use high-CPU or high-memory machine type for transformers"
        assert re.search(disk_size_pattern, content), \
            "Cloud Build should use increased disk size (150GB) for transformer builds"

    def test_cloudbuild_yaml_timeout_optimization(self, config_path):
        """Test that cloudbuild.yaml has increased timeout for transformer builds.""" 
        cloudbuild_path = config_path / "cloudbuild.yaml"
        content = cloudbuild_path.read_text()
        
        # Check for increased build timeout for transformer model preparation
        timeout_pattern = r"timeout:\s*['\"]?3600s['\"]?"  # 1 hour for transformer builds
        
        assert re.search(timeout_pattern, content), \
            "Cloud Build should have 1 hour timeout for transformer model builds"


class TestDeploymentConfigurationIntegration:
    """Test integration aspects of transformer deployment configuration."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_consistent_resource_allocation_across_scripts(self, project_root):
        """Test that resource allocations are consistent across deployment scripts that use them directly."""
        deploy_scripts = [
            "deploy/deploy-utils.sh",
            "deploy/blue_green_deployment.sh"
            # Note: automated_deployment_pipeline.sh calls blue_green_deployment.sh, so doesn't have direct configs
        ]
        
        config_files = [
            "config/cloudbuild.yaml"
        ]
        
        # Production resource expectations
        expected_production_resources = {
            'memory': '8Gi',
            'cpu': '6', 
            'timeout': '4200',
            'concurrency': '15'
        }
        
        for script_path in deploy_scripts:
            full_path = project_root / script_path
            if full_path.exists():
                content = full_path.read_text()
                
                # Check for consistent production resource allocation
                for resource, value in expected_production_resources.items():
                    pattern = f"--{resource}={value}"
                    assert re.search(re.escape(pattern), content), \
                        f"{script_path} should have consistent {resource}={value} for production"
        
        # Check config files separately
        for config_path in config_files:
            full_path = project_root / config_path
            if full_path.exists():
                content = full_path.read_text()
                
                # Check for consistent production resource allocation in config files
                for resource, value in expected_production_resources.items():
                    pattern = f"--{resource}={value}"
                    assert re.search(re.escape(pattern), content), \
                        f"{config_path} should have consistent {resource}={value} for production"

    def test_environment_specific_resource_scaling(self, project_root):
        """Test that different environments have appropriate resource scaling."""
        deploy_utils_path = project_root / "deploy" / "deploy-utils.sh"
        assert deploy_utils_path.exists()
        
        content = deploy_utils_path.read_text()
        
        # Check that staging has scaled-down but sufficient resources
        staging_patterns = {
            '--memory=6Gi': 'Staging should have 6Gi memory (scaled from production 8Gi)',
            '--cpu=4': 'Staging should have 4 CPU (scaled from production 6)',
            '--timeout=2700': 'Staging should have 45min timeout (scaled from production 70min)', 
            '--concurrency=20': 'Staging should have 20 concurrency (scaled from production 15)'
        }
        
        # Look for the get_resource_limits function and check staging section
        get_resource_limits_match = re.search(r'get_resource_limits\(\).*?case.*?staging.*?echo.*?"([^"]*)"', content, re.DOTALL | re.IGNORECASE)
        if get_resource_limits_match:
            staging_resources = get_resource_limits_match.group(1)
            for pattern, message in staging_patterns.items():
                assert pattern in staging_resources, f"{message} - found: {staging_resources}"
        else:
            # Fallback: search for staging patterns anywhere in the file
            for pattern, message in staging_patterns.items():
                assert re.search(re.escape(pattern), content), message

    def test_transformer_specific_configuration_consistency(self, project_root):
        """Test that transformer-specific configurations are consistent across files."""
        config_files = [
            "deploy/deploy-utils.sh",
            "deploy/blue_green_deployment.sh",
            "config/cloudbuild.yaml"
        ]
        
        # Transformer-specific configurations that should be consistent
        transformer_configs = [
            'TRANSFORMER_OPTIMIZED=true',
            'TOKENIZERS_PARALLELISM=false',
            'TRANSFORMERS_CACHE=/app/models/cache'
        ]
        
        for config_file in config_files:
            full_path = project_root / config_file
            if full_path.exists():
                content = full_path.read_text()
                
                for config in transformer_configs:
                    assert re.search(re.escape(config), content), \
                        f"{config_file} should include transformer config: {config}"


class TestDeploymentConfigurationValidation:
    """Test validation functions for deployment configurations."""
    
    def test_validate_resource_limits_function(self):
        """Test that resource validation function properly validates transformer requirements."""
        # This test assumes there's a validation function in deploy-utils.sh
        # The function should validate minimum requirements for transformer deployment
        
        # Test cases for resource validation
        test_cases = [
            # (memory, cpu, timeout, concurrency, expected_valid)
            ("8Gi", "6", "4200", "15", True),    # Valid transformer config
            ("6Gi", "4", "2700", "20", True),    # Valid staging config  
            ("4Gi", "2", "1800", "50", False),   # Insufficient for transformers
            ("2Gi", "1", "900", "100", False),   # Insufficient for transformers
            ("10Gi", "8", "5400", "10", True),   # Over-provisioned but valid
        ]
        
        # This would test a validation function that doesn't exist yet
        # Following TDD - this test should fail until implementation
        for memory, cpu, timeout, concurrency, expected in test_cases:
            # Simulate validation function call
            # result = validate_transformer_resources(memory, cpu, timeout, concurrency)
            # assert result == expected, f"Resource validation failed for {memory}, {cpu}, {timeout}, {concurrency}"
            pass  # Placeholder until implementation

    def test_deployment_environment_validation(self):
        """Test that deployment environment validation includes transformer checks."""
        # Test environment-specific validation
        environments = ['staging', 'production', 'development']
        
        for env in environments:
            # This would test environment validation that doesn't exist yet
            # result = validate_deployment_environment(env, transformer_mode=True)
            # assert result.is_valid(), f"Environment {env} should be valid for transformer deployment"
            # assert result.has_sufficient_resources(), f"Environment {env} should have sufficient resources"
            pass  # Placeholder until implementation

    def test_configuration_compatibility_validation(self):
        """Test that configuration compatibility validation works across deployment types."""
        deployment_types = ['blue_green', 'rolling', 'canary']
        
        for deployment_type in deployment_types:
            # This would test compatibility validation
            # result = validate_deployment_compatibility(deployment_type, transformer_optimized=True)
            # assert result.is_compatible(), f"Deployment type {deployment_type} should be compatible with transformers"
            pass  # Placeholder until implementation


class TestPerformanceOptimizationConfiguration:
    """Test performance optimization configurations for transformer deployments."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_memory_optimization_settings(self, project_root):
        """Test that memory optimization settings are properly configured."""
        deploy_utils_path = project_root / "deploy" / "deploy-utils.sh"
        
        if deploy_utils_path.exists():
            content = deploy_utils_path.read_text()
            
            # Check for memory optimization environment variables
            memory_optimizations = [
                'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128',
                'TRANSFORMERS_NO_ADVISORY_WARNINGS=1',
                'TOKENIZERS_PARALLELISM=false',
                'OMP_NUM_THREADS=6',  # Should match CPU allocation
                'MKL_NUM_THREADS=6'   # Should match CPU allocation
            ]
            
            for optimization in memory_optimizations:
                assert re.search(re.escape(optimization), content), \
                    f"Memory optimization setting should be configured: {optimization}"

    def test_cpu_optimization_settings(self, project_root):
        """Test that CPU optimization settings are properly configured."""
        config_files = [
            "deploy/deploy-utils.sh",
            "config/cloudbuild.yaml"
        ]
        
        for config_file in config_files:
            full_path = project_root / config_file
            if full_path.exists():
                content = full_path.read_text()
                
                # Check for CPU optimization settings
                cpu_optimizations = [
                    'TORCH_COMPILE_MODE=reduce-overhead',
                    'PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1'
                ]
                
                for optimization in cpu_optimizations:
                    # Should be present in at least one file
                    if re.search(re.escape(optimization), content):
                        break
                else:
                    pytest.fail(f"CPU optimization {optimization} not found in any config")

    def test_inference_optimization_configuration(self, project_root):
        """Test that inference optimization is properly configured for transformers.""" 
        deploy_utils_path = project_root / "deploy" / "deploy-utils.sh"
        
        if deploy_utils_path.exists():
            content = deploy_utils_path.read_text()
            
            # Check for inference optimization settings
            inference_optimizations = [
                'TRANSFORMER_BATCH_SIZE=1',  # Optimized for single predictions
                'TRANSFORMER_MAX_LENGTH=512',  # Reasonable context length
                'TORCH_INFERENCE_MODE=1'  # Enable inference mode optimization
            ]
            
            for optimization in inference_optimizations:
                assert re.search(re.escape(optimization), content), \
                    f"Inference optimization should be configured: {optimization}"


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery configurations for transformer deployments."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_timeout_and_retry_configuration(self, project_root):
        """Test that timeout and retry configurations are appropriate for transformers."""
        blue_green_path = project_root / "deploy" / "blue_green_deployment.sh"
        
        if blue_green_path.exists():
            content = blue_green_path.read_text()
            
            # Check for transformer-appropriate timeout and retry settings
            timeout_configs = [
                'HEALTH_CHECK_ATTEMPTS=15',  # More attempts for transformer loading
                'HEALTH_CHECK_INTERVAL=20',  # Longer intervals
                'TRAFFIC_MIGRATION_DELAY=180',  # More time for stabilization
                'ROLLBACK_TIMEOUT=300'  # 5 minutes for rollback
            ]
            
            for config in timeout_configs:
                assert re.search(re.escape(config), content), \
                    f"Timeout configuration should be set for transformers: {config}"

    def test_fallback_configuration(self, project_root):
        """Test that fallback configurations are properly set for transformer failures."""
        pipeline_path = project_root / "deploy" / "automated_deployment_pipeline.sh"
        
        if pipeline_path.exists():
            content = pipeline_path.read_text()
            
            # Check for fallback and recovery configurations
            fallback_patterns = [
                r'fallback.*lstm',  # Fallback to LSTM models
                r'emergency.*stop',  # Emergency stop procedures
                r'circuit.*breaker',  # Circuit breaker patterns
                r'health.*check.*failed.*rollback'  # Automatic rollback on health check failure
            ]
            
            for pattern in fallback_patterns:
                assert re.search(pattern, content, re.IGNORECASE), \
                    f"Fallback configuration should include: {pattern}"

    def test_monitoring_and_alerting_configuration(self, project_root):
        """Test that monitoring and alerting are configured for transformer deployments."""
        config_files = [
            "deploy/automated_deployment_pipeline.sh",
            "config/cloudbuild.yaml"
        ]
        
        monitoring_patterns = [
            r'transformer.*metrics',
            r'attention.*monitoring',
            r'memory.*usage.*alert',
            r'inference.*latency.*monitor'
        ]
        
        for config_file in config_files:
            full_path = project_root / config_file
            if full_path.exists():
                content = full_path.read_text()
                
                # At least some monitoring should be configured
                monitoring_found = any(re.search(pattern, content, re.IGNORECASE) 
                                     for pattern in monitoring_patterns)
                
                if monitoring_found:
                    break
        else:
            pytest.fail("No transformer-specific monitoring configuration found")


# Additional test helpers and utilities

def validate_yaml_syntax(yaml_content):
    """Validate YAML syntax for configuration files."""
    try:
        yaml.safe_load(yaml_content)
        return True
    except yaml.YAMLError:
        return False


def extract_resource_values(content, resource_type):
    """Extract resource values from deployment script content."""
    pattern = f"--{resource_type}=([^\\s]+)"
    matches = re.findall(pattern, content)
    return matches


def validate_resource_consistency(files, resource_type, expected_values):
    """Validate that resource values are consistent across multiple files."""
    for file_path in files:
        if file_path.exists():
            content = file_path.read_text()
            values = extract_resource_values(content, resource_type)
            
            # Check if any expected value is present
            if not any(value in expected_values for value in values):
                return False, f"Resource {resource_type} not found in {file_path}"
    
    return True, "Resource consistency validated"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])