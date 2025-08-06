#!/usr/bin/env python3
"""
Test suite for transformer rollout configuration files.
Following TDD approach - tests created first.
"""

import os
import yaml
import json
import pytest
from pathlib import Path


class TestTransformerRolloutConfig:
    """Test suite for transformer rollout configuration."""
    
    @pytest.fixture
    def config_dir(self):
        """Get the configs directory path."""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "configs"
    
    @pytest.fixture
    def rollout_config_path(self, config_dir):
        """Get rollout config file path."""
        return config_dir / "transformer_rollout.yaml"
    
    @pytest.fixture
    def models_config_path(self, config_dir):
        """Get models config file path."""
        return config_dir / "transformer_models.json"
    
    def test_config_directory_exists(self, config_dir):
        """Test that deploy/configs directory exists."""
        assert config_dir.exists(), "deploy/configs directory should exist"
    
    def test_rollout_config_file_exists(self, rollout_config_path):
        """Test that transformer_rollout.yaml exists."""
        assert rollout_config_path.exists(), "transformer_rollout.yaml should exist"
    
    def test_models_config_file_exists(self, models_config_path):
        """Test that transformer_models.json exists."""
        assert models_config_path.exists(), "transformer_models.json should exist"
    
    def test_rollout_config_structure(self, rollout_config_path):
        """Test rollout config has required structure."""
        with open(rollout_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Top-level structure
        assert 'rollout_stages' in config
        assert 'canary_deployment' in config
        assert 'validation' in config
        assert 'rollback_triggers' in config
        assert 'integration' in config
        
        # Rollout stages
        stages = config['rollout_stages']
        assert isinstance(stages, list)
        assert len(stages) >= 3  # At least 3 stages
        
        for stage in stages:
            assert 'name' in stage
            assert 'traffic_percentage' in stage
            assert 'duration_minutes' in stage
            assert 'validation_checks' in stage
        
        # Canary deployment
        canary = config['canary_deployment']
        assert 'initial_traffic_percent' in canary
        assert 'intermediate_traffic_percent' in canary
        assert 'final_traffic_percent' in canary
        
        # Validation thresholds
        validation = config['validation']
        assert 'health_check_attempts' in validation
        assert 'health_check_interval' in validation
        assert 'success_threshold' in validation
        assert 'error_threshold' in validation
        
        # Rollback triggers
        rollback = config['rollback_triggers']
        assert 'error_rate_threshold' in rollback
        assert 'response_time_threshold' in rollback
        assert 'health_check_failures' in rollback
    
    def test_models_config_structure(self, models_config_path):
        """Test models config has required structure."""
        with open(models_config_path, 'r') as f:
            config = json.load(f)
        
        # Should have all transformer models
        expected_models = ['iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM']
        for model in expected_models:
            assert model in config, f"{model} should be in config"
            
            model_config = config[model]
            assert 'resources' in model_config
            assert 'health_checks' in model_config
            assert 'timeouts' in model_config
            assert 'environment' in model_config
            
            # Resources
            resources = model_config['resources']
            assert 'memory' in resources
            assert 'cpu' in resources
            assert 'scaling_factor' in resources
            
            # Health checks
            health = model_config['health_checks']
            assert 'endpoint' in health
            assert 'timeout_seconds' in health
            assert 'interval_seconds' in health
            assert 'max_failures' in health
            
            # Timeouts
            timeouts = model_config['timeouts']
            assert 'request_timeout' in timeouts
            assert 'startup_timeout' in timeouts
            assert 'shutdown_timeout' in timeouts
            
            # Environment
            env = model_config['environment']
            assert 'variables' in env
            assert isinstance(env['variables'], dict)
    
    def test_resource_requirements_match_phase326(self, models_config_path):
        """Test that resource requirements match Phase 3.2.6 specifications."""
        with open(models_config_path, 'r') as f:
            config = json.load(f)
        
        # Expected requirements from Phase 3.2.6
        expected_requirements = {
            'iTransformer': {'memory': '4Gi', 'cpu': 2},
            'PatchTST': {'memory': '3Gi', 'cpu': 2},
            'TimesMixer': {'memory': '5Gi', 'cpu': 3},
            'TimesFM': {'memory': '6Gi', 'cpu': 4}
        }
        
        for model, expected in expected_requirements.items():
            actual = config[model]['resources']
            assert actual['memory'] == expected['memory']
            assert actual['cpu'] == expected['cpu']
    
    def test_integration_with_blue_green_deployment(self, rollout_config_path):
        """Test integration section for blue_green_deployment.sh."""
        with open(rollout_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        integration = config['integration']
        assert 'blue_green_deployment' in integration
        
        bg_config = integration['blue_green_deployment']
        assert 'source_script' in bg_config
        assert bg_config['source_script'] == './blue_green_deployment.sh'
        assert 'config_injection_method' in bg_config
        assert 'environment_variables' in bg_config
    
    def test_config_files_are_valid_yaml_json(self, rollout_config_path, models_config_path):
        """Test that config files are valid YAML/JSON."""
        # Test YAML parsing
        with open(rollout_config_path, 'r') as f:
            yaml.safe_load(f)
        
        # Test JSON parsing
        with open(models_config_path, 'r') as f:
            json.load(f)
    
    def test_rollout_stages_progressive(self, rollout_config_path):
        """Test that rollout stages have progressive traffic percentages."""
        with open(rollout_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config['rollout_stages']
        traffic_percentages = [stage['traffic_percentage'] for stage in stages]
        
        # Should be in ascending order
        assert traffic_percentages == sorted(traffic_percentages)
        # Final stage should be 100%
        assert traffic_percentages[-1] == 100


class TestModulesDirectory:
    """Test suite for deploy/modules directory."""
    
    @pytest.fixture
    def modules_dir(self):
        """Get the modules directory path."""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "modules"
    
    def test_modules_directory_exists(self, modules_dir):
        """Test that deploy/modules directory exists."""
        assert modules_dir.exists(), "deploy/modules directory should exist"
    
    def test_modules_directory_structure(self, modules_dir):
        """Test modules directory has proper structure for future use."""
        # Should be empty for now but ready for modular components
        assert modules_dir.is_dir(), "modules should be a directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])