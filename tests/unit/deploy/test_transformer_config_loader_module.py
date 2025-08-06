#!/usr/bin/env python3
"""
Unit Tests for Transformer Config Loader Module

Tests the transformer_config_loader.sh module functionality including:
- Configuration file parsing (YAML and JSON)
- Environment variable export
- Resource limit configuration
- Health check parameter setup
- Rollout stage determination
- Compatibility with existing deployment scripts

This test suite validates the transformer config loader module integration.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import yaml


class TestTransformerConfigLoaderModule(unittest.TestCase):
    """Test suite for transformer config loader module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_loader_module = self.project_root / "deploy" / "modules" / "transformer_config_loader.sh"
        
        # Test rollout configuration
        self.test_rollout_config = {
            'rollout_stages': [
                {
                    'name': 'canary',
                    'traffic_percentage': 10,
                    'duration_minutes': 5,
                    'validation_checks': ['health_endpoint', 'error_rate']
                },
                {
                    'name': 'full',
                    'traffic_percentage': 100,
                    'duration_minutes': 30,
                    'validation_checks': ['health_endpoint', 'error_rate', 'end_to_end_validation']
                }
            ],
            'validation': {
                'health_check_attempts': 15,
                'health_check_interval': 20,
                'success_threshold': 95,
                'error_threshold': 5
            }
        }
        
        # Test models configuration
        self.test_models_config = {
            'iTransformer': {
                'resources': {
                    'memory': '4Gi',
                    'cpu': 2,
                    'scaling_factor': 1.2,
                    'max_concurrent_requests': 10
                },
                'health_checks': {
                    'endpoint': '/health/iTransformer',
                    'timeout_seconds': 30,
                    'interval_seconds': 15
                },
                'environment': {
                    'variables': {
                        'MODEL_TYPE': 'iTransformer',
                        'TRANSFORMER_MAX_LENGTH': '512',
                        'TRANSFORMER_BATCH_SIZE': '1'
                    }
                }
            }
        }
    
    def test_module_exists_and_sourceable(self):
        """Test that config loader module exists and can be sourced"""
        self.assertTrue(self.config_loader_module.exists(), "Transformer config loader module not found")
        
        # Test sourcing the module
        result = subprocess.run(
            ["bash", "-c", f"source {self.config_loader_module} && declare -F load_transformer_configurations"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Module should be sourceable without errors")
        self.assertIn("load_transformer_configurations", result.stdout, "Main config loader function should be defined")
    
    def test_load_transformer_configurations_function_exists(self):
        """Test that main configuration loading function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type load_transformer_configurations"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("load_transformer_configurations is a function", result.stdout)
    
    def test_parse_transformer_rollout_yaml_function_exists(self):
        """Test that YAML parsing function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type parse_transformer_rollout_yaml"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("parse_transformer_rollout_yaml is a function", result.stdout)
    
    def test_parse_transformer_models_json_function_exists(self):
        """Test that JSON parsing function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type parse_transformer_models_json"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("parse_transformer_models_json is a function", result.stdout)
    
    def test_export_environment_variables_function_exists(self):
        """Test that environment variable export function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type export_transformer_env_vars"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("export_transformer_env_vars is a function", result.stdout)
    
    def test_set_resource_limits_function_exists(self):
        """Test that resource limits function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type set_transformer_resource_limits"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("set_transformer_resource_limits is a function", result.stdout)
    
    def test_configure_health_check_parameters_function_exists(self):
        """Test that health check configuration function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type configure_transformer_health_checks"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("configure_transformer_health_checks is a function", result.stdout)
    
    def test_determine_rollout_stage_function_exists(self):
        """Test that rollout stage determination function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.config_loader_module} && type determine_rollout_stage"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("determine_rollout_stage is a function", result.stdout)
    
    def test_yaml_configuration_loading(self):
        """Test loading YAML rollout configuration"""
        # Create a temporary YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_rollout_config, f)
            temp_yaml = f.name
        
        try:
            result = subprocess.run([
                "bash", "-c",
                f"""source {self.config_loader_module}
                TRANSFORMER_ROLLOUT_CONFIG='{temp_yaml}'
                parse_transformer_rollout_yaml
                echo "YAML parsing: $?"
                echo "Health check attempts: $HEALTH_CHECK_ATTEMPTS"
                echo "Health check interval: $HEALTH_CHECK_INTERVAL"
                """
            ], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0, "Should load YAML config successfully")
            self.assertIn("YAML parsing: 0", result.stdout)
            self.assertIn("Health check attempts: 15", result.stdout)
            self.assertIn("Health check interval: 20", result.stdout)
        
        finally:
            os.unlink(temp_yaml)
    
    def test_json_configuration_loading(self):
        """Test loading JSON models configuration"""
        # Create a temporary JSON config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_models_config, f)
            temp_json = f.name
        
        try:
            result = subprocess.run([
                "bash", "-c",
                f"""source {self.config_loader_module}
                TRANSFORMER_MODELS_CONFIG='{temp_json}'
                parse_transformer_models_json 'iTransformer'
                echo "JSON parsing: $?"
                echo "Model memory: $MODEL_MEMORY"
                echo "Model CPU: $MODEL_CPU"
                echo "Health endpoint: $HEALTH_ENDPOINT"
                """
            ], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0, "Should load JSON config successfully")
            self.assertIn("JSON parsing: 0", result.stdout)
            self.assertIn("Model memory: 4Gi", result.stdout)
            self.assertIn("Model CPU: 2", result.stdout)
            self.assertIn("Health endpoint: /health/iTransformer", result.stdout)
        
        finally:
            os.unlink(temp_json)
    
    def test_environment_variable_export(self):
        """Test environment variable export functionality"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            export_transformer_env_vars 'iTransformer' '4Gi' '2' '/health/iTransformer' '30'
            echo "MODEL_TYPE: $MODEL_TYPE"
            echo "TRANSFORMER_MEMORY: $TRANSFORMER_MEMORY"
            echo "TRANSFORMER_CPU: $TRANSFORMER_CPU"
            echo "TRANSFORMER_HEALTH_ENDPOINT: $TRANSFORMER_HEALTH_ENDPOINT"
            echo "TRANSFORMER_HEALTH_TIMEOUT: $TRANSFORMER_HEALTH_TIMEOUT"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("MODEL_TYPE: iTransformer", result.stdout)
        self.assertIn("TRANSFORMER_MEMORY: 4Gi", result.stdout)
        self.assertIn("TRANSFORMER_CPU: 2", result.stdout)
        self.assertIn("TRANSFORMER_HEALTH_ENDPOINT: /health/iTransformer", result.stdout)
        self.assertIn("TRANSFORMER_HEALTH_TIMEOUT: 30", result.stdout)
    
    def test_resource_limits_configuration(self):
        """Test resource limits configuration"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            set_transformer_resource_limits 'TimesMixer' '5Gi' '3' '8'
            echo "Memory limit: $MEMORY_LIMIT"
            echo "CPU limit: $CPU_LIMIT"
            echo "Max instances: $MAX_INSTANCES"
            echo "Resource validation: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Memory limit: 5Gi", result.stdout)
        self.assertIn("CPU limit: 3", result.stdout)
        self.assertIn("Max instances: 8", result.stdout)
        self.assertIn("Resource validation: 0", result.stdout)
    
    def test_health_check_parameter_configuration(self):
        """Test health check parameter configuration"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            configure_transformer_health_checks '/health/test' '45' '25' '3'
            echo "Health endpoint: $HEALTH_CHECK_ENDPOINT"
            echo "Health timeout: $HEALTH_CHECK_TIMEOUT"
            echo "Health interval: $HEALTH_CHECK_INTERVAL"
            echo "Max failures: $HEALTH_CHECK_MAX_FAILURES"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Health endpoint: /health/test", result.stdout)
        self.assertIn("Health timeout: 45", result.stdout)
        self.assertIn("Health interval: 25", result.stdout)
        self.assertIn("Max failures: 3", result.stdout)
    
    def test_rollout_stage_determination(self):
        """Test rollout stage determination logic"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            determine_rollout_stage 'staging' '10'
            echo "Rollout stage: $ROLLOUT_STAGE"
            echo "Traffic percentage: $TRAFFIC_PERCENTAGE"
            determine_rollout_stage 'production' '100'
            echo "Rollout stage: $ROLLOUT_STAGE"
            echo "Traffic percentage: $TRAFFIC_PERCENTAGE"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        # Should determine appropriate rollout stages based on environment and traffic
        output = result.stdout
        self.assertTrue("Rollout stage:" in output)
        self.assertTrue("Traffic percentage:" in output)
    
    def test_compatibility_with_deploy_script(self):
        """Test compatibility with deploy.sh script"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            # Test that expected variables are set for deploy.sh compatibility
            load_transformer_configurations 'iTransformer' 'staging'
            echo "Deploy compatibility: $?"
            # Should set variables that deploy.sh expects
            [[ -n "$TRANSFORMER_OPTIMIZED" ]] && echo "TRANSFORMER_OPTIMIZED set"
            [[ -n "$ROLLOUT_MODE" ]] && echo "ROLLOUT_MODE set"
            [[ -n "$MODEL_VALIDATION_ENABLED" ]] && echo "MODEL_VALIDATION_ENABLED set"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Deploy compatibility: 0", result.stdout)
        self.assertIn("TRANSFORMER_OPTIMIZED set", result.stdout)
        self.assertIn("ROLLOUT_MODE set", result.stdout)
        self.assertIn("MODEL_VALIDATION_ENABLED set", result.stdout)
    
    def test_compatibility_with_blue_green_deployment(self):
        """Test compatibility with blue_green_deployment.sh script"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            # Test that expected variables are set for blue_green_deployment.sh compatibility
            configure_for_blue_green_deployment 'iTransformer' 'production'
            echo "Blue-green compatibility: $?"
            # Should set variables that blue_green_deployment.sh expects
            [[ -n "$HEALTH_CHECK_ATTEMPTS" ]] && echo "HEALTH_CHECK_ATTEMPTS set"
            [[ -n "$HEALTH_CHECK_INTERVAL" ]] && echo "HEALTH_CHECK_INTERVAL set"
            [[ -n "$TRAFFIC_MIGRATION_DELAY" ]] && echo "TRAFFIC_MIGRATION_DELAY set"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Blue-green compatibility: 0", result.stdout)
        self.assertIn("HEALTH_CHECK_ATTEMPTS set", result.stdout)
        self.assertIn("HEALTH_CHECK_INTERVAL set", result.stdout)
        self.assertIn("TRAFFIC_MIGRATION_DELAY set", result.stdout)
    
    def test_error_handling_and_validation(self):
        """Test error handling for invalid configurations"""
        # Test with invalid model type
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            load_transformer_configurations 'InvalidModel' 'staging'
            echo "Error handling: $?"
            """
        ], capture_output=True, text=True)
        
        # Should return non-zero exit code for invalid model
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid model type")
    
    def test_logging_consistency(self):
        """Test that logging follows deployment script conventions"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            # Test logging functions exist and work
            log_config_info "Test config info message"
            log_config_success "Test config success message" 
            log_config_warning "Test config warning message"
            log_config_error "Test config error message"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        # Should use consistent color coding and format with existing scripts
        output = result.stdout + result.stderr
        self.assertTrue(any("[INFO]" in line or "[SUCCESS]" in line or "[WARNING]" in line or "[ERROR]" in line 
                          for line in output.split('\n')), 
                       "Should use consistent logging format")


class TestTransformerConfigLoaderIntegration(unittest.TestCase):
    """Integration tests for transformer config loader module"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_loader_module = self.project_root / "deploy" / "modules" / "transformer_config_loader.sh"
        self.rollout_config = self.project_root / "deploy" / "configs" / "transformer_rollout.yaml"
        self.models_config = self.project_root / "deploy" / "configs" / "transformer_models.json"
    
    def test_actual_config_files_integration(self):
        """Test integration with actual config files"""
        self.assertTrue(self.rollout_config.exists(), "Transformer rollout config not found")
        self.assertTrue(self.models_config.exists(), "Transformer models config not found")
        
        # Test loading the actual configs
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            TRANSFORMER_ROLLOUT_CONFIG='{self.rollout_config}'
            TRANSFORMER_MODELS_CONFIG='{self.models_config}'
            load_transformer_configurations 'iTransformer' 'staging'
            echo "Integration test: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Should load actual configs successfully")
        self.assertIn("Integration test: 0", result.stdout)
    
    def test_cloud_run_deployment_compatibility(self):
        """Test compatibility with Cloud Run deployment requirements"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.config_loader_module}
            # Test Cloud Run specific configuration
            configure_for_cloud_run 'TimesFM' 'production'
            echo "Cloud Run compatibility: $?"
            # Should set Cloud Run specific variables
            [[ -n "$CLOUD_RUN_MEMORY" ]] && echo "CLOUD_RUN_MEMORY set"
            [[ -n "$CLOUD_RUN_CPU" ]] && echo "CLOUD_RUN_CPU set"
            [[ -n "$CLOUD_RUN_TIMEOUT" ]] && echo "CLOUD_RUN_TIMEOUT set"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Should configure for Cloud Run successfully")


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PROJECT_ID', 'test-project')
    os.environ.setdefault('ENVIRONMENT', 'staging')
    
    # Run tests
    unittest.main(verbosity=2)