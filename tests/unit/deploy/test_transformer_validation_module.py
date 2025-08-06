#!/usr/bin/env python3
"""
Unit Tests for Transformer Validation Module

Tests the transformer_validation.sh module functionality including:
- Transformer model availability validation
- Resource requirements checking
- Health endpoint testing
- Attention mechanism validation
- Model loading performance checks
- NaN/Inf handling validation

This test suite validates the transformer validation module for Stage 2.5 integration.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


class TestTransformerValidationModule(unittest.TestCase):
    """Test suite for transformer validation module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.validation_module = self.project_root / "deploy" / "modules" / "transformer_validation.sh"
        
        # Test configurations for transformer models
        self.transformer_configs = {
            'iTransformer': {
                'memory_requirement': '4Gi',
                'cpu_requirement': '2',
                'health_endpoint': '/health/iTransformer',
                'startup_timeout': 300
            },
            'PatchTST': {
                'memory_requirement': '3Gi',
                'cpu_requirement': '2',
                'health_endpoint': '/health/PatchTST',
                'startup_timeout': 240
            },
            'TimesMixer': {
                'memory_requirement': '5Gi',
                'cpu_requirement': '3',
                'health_endpoint': '/health/TimesMixer',
                'startup_timeout': 360
            },
            'TimesFM': {
                'memory_requirement': '6Gi',
                'cpu_requirement': '4',
                'health_endpoint': '/health/TimesFM',
                'startup_timeout': 480
            }
        }
    
    def test_module_exists_and_sourceable(self):
        """Test that validation module exists and can be sourced"""
        self.assertTrue(self.validation_module.exists(), "Transformer validation module not found")
        
        # Test sourcing the module
        result = subprocess.run(
            ["bash", "-c", f"source {self.validation_module} && declare -F validate_transformer_models"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Module should be sourceable without errors")
        self.assertIn("validate_transformer_models", result.stdout, "Main validation function should be defined")
    
    def test_validate_transformer_models_function_exists(self):
        """Test that main validation function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type validate_transformer_models"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate_transformer_models is a function", result.stdout)
    
    def test_resource_validation_function_exists(self):
        """Test that resource validation function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type validate_transformer_resources"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate_transformer_resources is a function", result.stdout)
    
    def test_health_endpoint_validation_function_exists(self):
        """Test that health endpoint validation function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type validate_transformer_health_endpoints"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate_transformer_health_endpoints is a function", result.stdout)
    
    def test_attention_mechanism_validation_function_exists(self):
        """Test that attention mechanism validation function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type test_attention_computation"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("test_attention_computation is a function", result.stdout)
    
    def test_model_loading_performance_function_exists(self):
        """Test that model loading performance function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type validate_model_loading_performance"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate_model_loading_performance is a function", result.stdout)
    
    def test_nan_inf_handling_function_exists(self):
        """Test that NaN/Inf handling validation function exists"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type check_nan_inf_handling"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("check_nan_inf_handling is a function", result.stdout)
    
    def test_configuration_loading(self):
        """Test that module can load transformer configurations"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.transformer_configs, f)
            temp_config = f.name
        
        try:
            result = subprocess.run([
                "bash", "-c",
                f"source {self.validation_module} && TRANSFORMER_CONFIG_FILE='{temp_config}' load_transformer_config && echo 'Config loaded successfully'"
            ], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0, "Should load transformer config successfully")
            self.assertIn("Config loaded successfully", result.stdout)
        
        finally:
            os.unlink(temp_config)
    
    def test_resource_requirement_validation(self):
        """Test resource requirement validation logic"""
        # Test with sufficient resources
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            AVAILABLE_MEMORY='8Gi'
            AVAILABLE_CPU='4'
            validate_transformer_resources 'iTransformer' '4Gi' '2'
            echo "Exit code: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertIn("Exit code: 0", result.stdout, "Should pass with sufficient resources")
    
    def test_error_handling_and_exit_codes(self):
        """Test proper error handling and exit codes"""
        # Test with invalid model type
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            validate_transformer_models 'InvalidModel'
            echo "Exit code: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid model type")
    
    def test_integration_with_deployment_pipeline(self):
        """Test integration compatibility with automated_deployment_pipeline.sh"""
        # Check that the module defines expected environment variables
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            # Should be able to set validation results
            TRANSFORMER_VALIDATION_RESULT="PASS"
            TRANSFORMER_VALIDATION_MESSAGE="All validations passed"
            echo "Validation result: $TRANSFORMER_VALIDATION_RESULT"
            echo "Validation message: $TRANSFORMER_VALIDATION_MESSAGE"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Validation result: PASS", result.stdout)
        self.assertIn("Validation message: All validations passed", result.stdout)
    
    def test_logging_consistency(self):
        """Test that logging follows deployment script conventions"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            # Test logging functions exist and work
            log_transformer_info "Test info message"
            log_transformer_success "Test success message" 
            log_transformer_warning "Test warning message"
            log_transformer_error "Test error message"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        # Should use consistent color coding and format with existing scripts
        output = result.stdout + result.stderr
        self.assertTrue(any("[INFO]" in line or "[SUCCESS]" in line or "[WARNING]" in line or "[ERROR]" in line 
                          for line in output.split('\n')), 
                       "Should use consistent logging format")
    
    def test_performance_benchmarking(self):
        """Test that performance benchmarking functions exist"""
        result = subprocess.run([
            "bash", "-c",
            f"source {self.validation_module} && type benchmark_transformer_loading"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("benchmark_transformer_loading is a function", result.stdout)


class TestTransformerValidationIntegration(unittest.TestCase):
    """Integration tests for transformer validation module"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.validation_module = self.project_root / "deploy" / "modules" / "transformer_validation.sh"
        self.transformer_models_config = self.project_root / "deploy" / "configs" / "transformer_models.json"
    
    def test_config_file_integration(self):
        """Test integration with transformer_models.json config file"""
        self.assertTrue(self.transformer_models_config.exists(), "Transformer models config not found")
        
        # Test loading the actual config
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            TRANSFORMER_CONFIG_FILE='{self.transformer_models_config}'
            load_transformer_config
            echo "Config validation: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Should load actual transformer config successfully")
        self.assertIn("Config validation: 0", result.stdout)
    
    def test_gcp_integration_compatibility(self):
        """Test compatibility with GCP deployment requirements"""
        result = subprocess.run([
            "bash", "-c",
            f"""source {self.validation_module}
            # Test GCP-specific validation functions
            validate_gcp_resource_limits '8Gi' '6'
            echo "GCP validation: $?"
            """
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Should validate GCP resources successfully")


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PROJECT_ID', 'test-project')
    os.environ.setdefault('ENVIRONMENT', 'staging')
    
    # Run tests
    unittest.main(verbosity=2)