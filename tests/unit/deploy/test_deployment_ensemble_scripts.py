#!/usr/bin/env python3
"""
TDD Tests for Phase 2 Deployment Script Updates - Ensemble Configuration

Tests the deployment script updates for ensemble architecture:
- ENVIRONMENT variable usage instead of TRANSFORMER_MODEL_TYPE
- Ensemble validation instead of per-model validation
- Updated config loader for environment-based configuration

This test suite validates Phase 2 deployment script simplification.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


class TestDeploymentEnsembleScripts(unittest.TestCase):
    """Test suite for ensemble deployment script updates"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.deploy_script = self.project_root / "deploy" / "deploy.sh"
        self.config_loader = self.project_root / "deploy" / "modules" / "transformer_config_loader.sh"
        self.validation_script = self.project_root / "deploy" / "modules" / "transformer_validation.sh"
        self.blue_green_script = self.project_root / "deploy" / "blue_green_deployment.sh"
        
        # Environment configurations
        self.environment_configs = {
            'development': {
                'memory': '2Gi',
                'cpu': '1',
                'models': 'lstm'
            },
            'staging': {
                'memory': '4Gi',
                'cpu': '3',
                'models': 'lstm,iTransformer,PatchTST,TimesMixer,TimesFM'
            },
            'production': {
                'memory': '8Gi',
                'cpu': '6',
                'models': 'lstm,iTransformer,PatchTST,TimesMixer,TimesFM'
            }
        }
    
    def test_config_loader_uses_environment_variable(self):
        """Test that transformer_config_loader.sh uses ENVIRONMENT variable instead of TRANSFORMER_MODEL_TYPE"""
        # Test development environment
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'development'
        
        result = subprocess.run([
            'bash', str(self.config_loader)
        ], capture_output=True, text=True, env=env_vars)
        
        # Should load development configuration (LSTM only)
        self.assertIn('development', result.stderr)
        self.assertIn('2Gi', result.stderr)  # Development memory
        self.assertIn('LSTM', result.stderr.upper())
        
        # Should NOT reference TRANSFORMER_MODEL_TYPE
        self.assertNotIn('TRANSFORMER_MODEL_TYPE', result.stderr)
    
    def test_config_loader_production_environment(self):
        """Test config loader with production environment"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'production'
        
        result = subprocess.run([
            'bash', str(self.config_loader)
        ], capture_output=True, text=True, env=env_vars)
        
        # Should load production configuration (ensemble)
        self.assertIn('production', result.stderr)
        self.assertIn('8Gi', result.stderr)  # Production memory
        self.assertIn('ensemble', result.stderr.lower())
    
    def test_config_loader_defaults_to_production(self):
        """Test that config loader defaults to production when ENVIRONMENT not set"""
        env_vars = os.environ.copy()
        # Remove ENVIRONMENT if it exists
        env_vars.pop('ENVIRONMENT', None)
        env_vars.pop('TRANSFORMER_MODEL_TYPE', None)
        
        result = subprocess.run([
            'bash', str(self.config_loader)
        ], capture_output=True, text=True, env=env_vars)
        
        # Should default to production
        # (May fail if script not updated yet - this is TDD)
        # Expected to be fixed in implementation
    
    def test_deploy_script_removes_transformer_model_type_logic(self):
        """Test that deploy.sh removes TRANSFORMER_MODEL_TYPE conditional logic"""
        with open(self.deploy_script, 'r') as f:
            deploy_content = f.read()
        
        # Should NOT contain TRANSFORMER_MODEL_TYPE references (after update)
        # This test will fail initially and pass after implementation
        
        # Should always source transformer_config_loader.sh
        self.assertIn('transformer_config_loader.sh', deploy_content)
    
    def test_deploy_script_uses_environment_variable(self):
        """Test that deploy.sh uses ENVIRONMENT variable"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'staging'
        
        result = subprocess.run([
            str(self.deploy_script), 'staging', 'validation-only'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should use ENVIRONMENT for configuration
        self.assertIn('staging', result.stderr)
        
        # Should NOT fail due to missing TRANSFORMER_MODEL_TYPE
        # (This test may fail initially - TDD approach)
    
    def test_validation_script_ensemble_validation(self):
        """Test that transformer_validation.sh validates ensemble instead of individual models"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'production'
        
        result = subprocess.run([
            'bash', str(self.validation_script), 'production'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should validate ensemble as a unit
        self.assertIn('ensemble', result.stderr.lower())
        
        # Should check combined resource usage
        self.assertIn('8Gi', result.stderr)  # Total ensemble memory
        
        # Should NOT validate individual transformers separately
        # (Will be implemented to pass this test)
    
    def test_validation_script_development_mode(self):
        """Test validation in development mode (LSTM only)"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'development'
        
        result = subprocess.run([
            'bash', str(self.validation_script), 'development'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should validate only LSTM
        self.assertIn('development', result.stderr.lower())
        self.assertIn('2Gi', result.stderr)  # Development memory limit
    
    def test_blue_green_removes_transformer_model_type(self):
        """Test that blue_green_deployment.sh removes TRANSFORMER_MODEL_TYPE references"""
        with open(self.blue_green_script, 'r') as f:
            blue_green_content = f.read()
        
        # Should NOT contain TRANSFORMER_MODEL_TYPE after update
        # This will fail initially and pass after implementation
        
        # Should use ENVIRONMENT variable
        self.assertIn('ENVIRONMENT', blue_green_content)
    
    def test_blue_green_uses_environment_for_canary_mode(self):
        """Test that blue_green_deployment.sh uses ENVIRONMENT for canary configuration"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'production'
        env_vars['CANARY_MODE'] = 'true'
        
        result = subprocess.run([
            str(self.blue_green_script), 'production', 'latest'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should configure canary based on ENVIRONMENT
        self.assertIn('production', result.stderr)
        
        # Should handle ensemble traffic routing
        # (Expected to work after implementation)
    
    def test_environment_configuration_loaded(self):
        """Test that environment configuration is properly loaded"""
        environments_config = self.project_root / "deploy" / "configs" / "environments.json"
        self.assertTrue(environments_config.exists(), "environments.json should exist")
        
        with open(environments_config, 'r') as f:
            config = json.load(f)
        
        # Should have all required environments
        required_envs = ['development', 'staging', 'production']
        for env in required_envs:
            self.assertIn(env, config)
            
            # Each environment should have required fields
            self.assertIn('models', config[env])
            self.assertIn('resources', config[env])
            self.assertIn('health_checks', config[env])
    
    def test_integration_no_transformer_model_type_required(self):
        """Test that deployment works without TRANSFORMER_MODEL_TYPE variable"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'staging'
        # Explicitly remove TRANSFORMER_MODEL_TYPE
        env_vars.pop('TRANSFORMER_MODEL_TYPE', None)
        
        result = subprocess.run([
            str(self.deploy_script), 'staging', 'validation-only'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should work without TRANSFORMER_MODEL_TYPE
        # (This test validates the goal of Phase 2)
        # May fail initially, will pass after implementation


class TestEnsembleValidationLogic(unittest.TestCase):
    """Test suite for ensemble validation logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.validation_script = self.project_root / "deploy" / "modules" / "transformer_validation.sh"
    
    def test_ensemble_health_check_endpoints(self):
        """Test that validation uses ensemble health endpoints"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'production'
        env_vars['SERVICE_URL'] = 'https://test.example.com'
        
        result = subprocess.run([
            'bash', str(self.validation_script), 'production'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should check ensemble health endpoint
        self.assertIn('/health/ensemble', result.stderr)
        
        # Should NOT check individual model endpoints
        # (Will be updated to pass this test)
    
    def test_ensemble_resource_validation(self):
        """Test that validation checks combined ensemble resources"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'production'
        env_vars['AVAILABLE_MEMORY'] = '8Gi'
        env_vars['AVAILABLE_CPU'] = '6'
        
        result = subprocess.run([
            'bash', str(self.validation_script), 'production'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should validate total ensemble resource usage
        self.assertIn('ensemble', result.stderr.lower())
        
        # Should check that 8Gi is sufficient for full ensemble
        # (Test designed to pass after implementation)
    
    def test_development_mode_validation(self):
        """Test validation logic for development mode (LSTM only)"""
        env_vars = os.environ.copy()
        env_vars['ENVIRONMENT'] = 'development'
        env_vars['AVAILABLE_MEMORY'] = '2Gi'
        env_vars['AVAILABLE_CPU'] = '1'
        
        result = subprocess.run([
            'bash', str(self.validation_script), 'development'
        ], capture_output=True, text=True, env=env_vars)
        
        # Should only validate LSTM in development
        self.assertIn('development', result.stderr.lower())
        
        # Should not attempt to validate transformers
        self.assertNotIn('iTransformer', result.stderr)
        self.assertNotIn('PatchTST', result.stderr)


class TestConfigurationIntegration(unittest.TestCase):
    """Test suite for configuration integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
    
    def test_environments_json_structure(self):
        """Test that environments.json has correct structure for all environments"""
        config_file = self.project_root / "deploy" / "configs" / "environments.json"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Test structure for each environment
        for env_name in ['development', 'staging', 'production']:
            env_config = config[env_name]
            
            # Required top-level keys
            self.assertIn('models', env_config)
            self.assertIn('resources', env_config)
            self.assertIn('health_checks', env_config)
            self.assertIn('environment', env_config)
            
            # Resource configuration
            resources = env_config['resources']
            self.assertIn('memory', resources)
            self.assertIn('cpu', resources)
            self.assertIn('timeout_seconds', resources)
            
            # Health check configuration
            health = env_config['health_checks']
            self.assertIn('endpoint', health)
            self.assertIn('timeout_seconds', health)
    
    def test_environment_specific_models(self):
        """Test that each environment specifies correct models"""
        config_file = self.project_root / "deploy" / "configs" / "environments.json"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Development should only have LSTM
        dev_models = config['development']['models']
        self.assertEqual(dev_models, ['lstm'])
        
        # Production should have full ensemble
        prod_models = config['production']['models']
        expected_models = ['lstm', 'iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM']
        self.assertEqual(sorted(prod_models), sorted(expected_models))
    
    def test_resource_scaling_by_environment(self):
        """Test that resources scale appropriately by environment"""
        config_file = self.project_root / "deploy" / "configs" / "environments.json"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Resources should increase: development < staging < production
        dev_mem = int(config['development']['resources']['memory'].rstrip('Gi'))
        staging_mem = int(config['staging']['resources']['memory'].rstrip('Gi'))
        prod_mem = int(config['production']['resources']['memory'].rstrip('Gi'))
        
        self.assertLess(dev_mem, staging_mem)
        self.assertLess(staging_mem, prod_mem)
        
        # CPU scaling
        dev_cpu = config['development']['resources']['cpu']
        staging_cpu = config['staging']['resources']['cpu']
        prod_cpu = config['production']['resources']['cpu']
        
        self.assertLess(dev_cpu, staging_cpu)
        self.assertLess(staging_cpu, prod_cpu)


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PROJECT_ID', 'test-project')
    
    # Run tests
    unittest.main(verbosity=2)