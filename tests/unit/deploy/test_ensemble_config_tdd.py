#!/usr/bin/env python3
"""
TDD Tests for Ensemble Configuration Simplification

These tests define the expected behavior for Phase 1 configuration simplification:
1. Single ensemble configuration in transformer_models.json
2. Environment-based configuration selection
3. Simplified rollout configuration

Following TDD: Write failing tests FIRST, then implement to make them pass.
"""

import os
import json
import yaml
import tempfile
import unittest
from pathlib import Path


class TestEnsembleConfigurationTDD(unittest.TestCase):
    """TDD test cases for ensemble configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_dir = self.project_root / "deploy" / "configs"
        self.transformer_models_config = self.config_dir / "transformer_models.json"
        self.transformer_rollout_config = self.config_dir / "transformer_rollout.yaml"
        self.environments_config = self.config_dir / "environments.json"
    
    def test_ensemble_configuration_exists(self):
        """Test that ensemble configuration is defined in transformer_models.json"""
        # This test will initially fail - we need to implement it
        self.assertTrue(self.transformer_models_config.exists(), 
                       "transformer_models.json should exist")
        
        with open(self.transformer_models_config, 'r') as f:
            config = json.load(f)
        
        # Should have single "ensemble" configuration
        self.assertIn("ensemble", config, 
                     "Config should contain 'ensemble' key")
        
        ensemble_config = config["ensemble"]
        
        # Test resource allocation
        self.assertEqual(ensemble_config["resources"]["memory"], "8Gi",
                        "Ensemble should use 8Gi memory")
        self.assertEqual(ensemble_config["resources"]["cpu"], 6,
                        "Ensemble should use 6 CPU")
        
        # Test model list
        expected_models = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        self.assertEqual(ensemble_config["models"], expected_models,
                        "Ensemble should list all 5 models")
        
        # Should NOT have individual transformer configurations
        individual_models = ["iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        for model in individual_models:
            self.assertNotIn(model, config,
                           f"Should not have individual config for {model}")
    
    def test_environments_configuration_exists(self):
        """Test that environments.json exists with dev/staging/prod modes"""
        # This test will initially fail - we need to create the file
        self.assertTrue(self.environments_config.exists(),
                       "environments.json should exist")
        
        with open(self.environments_config, 'r') as f:
            config = json.load(f)
        
        # Test development mode
        self.assertIn("development", config)
        dev_config = config["development"]
        self.assertEqual(dev_config["models"], ["lstm"],
                        "Development should only use LSTM")
        self.assertEqual(dev_config["resources"]["memory"], "2Gi",
                        "Development should use 2Gi memory")
        self.assertEqual(dev_config["resources"]["cpu"], 1,
                        "Development should use 1 CPU")
        
        # Test production mode
        self.assertIn("production", config)
        prod_config = config["production"]
        expected_models = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        self.assertEqual(prod_config["models"], expected_models,
                        "Production should use all models")
        self.assertEqual(prod_config["resources"]["memory"], "8Gi",
                        "Production should use 8Gi memory")
        self.assertEqual(prod_config["resources"]["cpu"], 6,
                        "Production should use 6 CPU")
        
        # Test staging mode (optional)
        if "staging" in config:
            staging_config = config["staging"]
            self.assertEqual(staging_config["models"], expected_models,
                            "Staging should use all models")
            self.assertEqual(staging_config["resources"]["memory"], "4Gi",
                            "Staging should use 4Gi memory")
            self.assertEqual(staging_config["resources"]["cpu"], 3,
                            "Staging should use 3 CPU")
    
    def test_rollout_configuration_simplified(self):
        """Test that rollout config is simplified for ensemble-only deployment"""
        self.assertTrue(self.transformer_rollout_config.exists(),
                       "transformer_rollout.yaml should exist")
        
        with open(self.transformer_rollout_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should maintain progressive rollout percentages
        stages = config["rollout_stages"]
        percentages = [stage["traffic_percentage"] for stage in stages]
        expected_percentages = [10, 25, 50, 100]
        self.assertEqual(percentages, expected_percentages,
                        "Should maintain 10% -> 25% -> 50% -> 100% rollout")
        
        # Should not have model-specific validation checks
        for stage in stages:
            validation_checks = stage["validation_checks"]
            # Should have ensemble-level checks
            self.assertIn("health_endpoint", validation_checks)
            self.assertIn("error_rate", validation_checks)
            self.assertIn("response_time", validation_checks)
            
            # Should not have model-specific checks
            model_specific_checks = ["transformer_model_accuracy", "iTransformer_health"]
            for check in model_specific_checks:
                self.assertNotIn(check, validation_checks,
                               f"Should not have model-specific check: {check}")
    
    def test_config_loader_supports_environment_variable(self):
        """Test that config loader supports ENVIRONMENT variable"""
        config_loader_module = self.project_root / "deploy" / "modules" / "transformer_config_loader.sh"
        self.assertTrue(config_loader_module.exists(),
                       "Config loader module should exist")
        
        with open(config_loader_module, 'r') as f:
            content = f.read()
        
        # Should support ENVIRONMENT variable
        self.assertIn("ENVIRONMENT", content,
                     "Config loader should reference ENVIRONMENT variable")
        
        # Should NOT reference TRANSFORMER_MODEL_TYPE
        self.assertNotIn("TRANSFORMER_MODEL_TYPE", content,
                        "Config loader should not reference TRANSFORMER_MODEL_TYPE")
        
        # Should default to production if ENVIRONMENT not set
        self.assertIn("production", content.lower(),
                     "Config loader should default to production")
    
    def test_no_transformer_model_type_references(self):
        """Test that TRANSFORMER_MODEL_TYPE is completely removed"""
        files_to_check = [
            self.transformer_models_config,
            self.transformer_rollout_config
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                self.assertNotIn("TRANSFORMER_MODEL_TYPE", content,
                               f"{file_path.name} should not reference TRANSFORMER_MODEL_TYPE")


class TestEnvironmentConfigLoading(unittest.TestCase):
    """Test cases for environment-based configuration loading"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # Expected environment configurations
        self.expected_dev_config = {
            "development": {
                "models": ["lstm"],
                "resources": {
                    "memory": "2Gi",
                    "cpu": 1,
                    "timeout_seconds": 2400
                },
                "description": "Development mode with LSTM only"
            }
        }
        
        self.expected_prod_config = {
            "production": {
                "models": ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"],
                "resources": {
                    "memory": "8Gi",
                    "cpu": 6,
                    "timeout_seconds": 4200
                },
                "description": "Production mode with full ensemble"
            }
        }
    
    def test_development_environment_configuration(self):
        """Test development environment loads correct configuration"""
        environments_config = self.project_root / "deploy" / "configs" / "environments.json"
        
        # This test will fail until we create the file
        self.assertTrue(environments_config.exists(),
                       "environments.json should exist")
        
        with open(environments_config, 'r') as f:
            config = json.load(f)
        
        # Test development configuration matches expected structure
        dev_config = config["development"]
        expected_dev = self.expected_dev_config["development"]
        
        self.assertEqual(dev_config["models"], expected_dev["models"])
        self.assertEqual(dev_config["resources"]["memory"], expected_dev["resources"]["memory"])
        self.assertEqual(dev_config["resources"]["cpu"], expected_dev["resources"]["cpu"])
    
    def test_production_environment_configuration(self):
        """Test production environment loads correct configuration"""
        environments_config = self.project_root / "deploy" / "configs" / "environments.json"
        
        self.assertTrue(environments_config.exists(),
                       "environments.json should exist")
        
        with open(environments_config, 'r') as f:
            config = json.load(f)
        
        # Test production configuration matches expected structure
        prod_config = config["production"]
        expected_prod = self.expected_prod_config["production"]
        
        self.assertEqual(prod_config["models"], expected_prod["models"])
        self.assertEqual(prod_config["resources"]["memory"], expected_prod["resources"]["memory"])
        self.assertEqual(prod_config["resources"]["cpu"], expected_prod["resources"]["cpu"])


if __name__ == '__main__':
    # Run TDD tests - these should initially fail
    unittest.main(verbosity=2)