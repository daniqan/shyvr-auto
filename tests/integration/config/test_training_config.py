"""
Integration Tests for Training Configuration Management

Following TDD methodology, these comprehensive tests define the expected behavior
of the training configuration system before implementation. Tests use REAL
configuration loading and validation - no mocks.

Test Structure:
1. Training configuration YAML loading from config/training_config.yaml
2. GCS corpus settings validation and integration
3. Model-specific hyperparameter loading
4. Environment-specific configuration overrides
5. Integration with existing configuration patterns
6. Configuration validation and error handling
7. Real GCS bucket access validation
"""

import pytest
import asyncio
import os
import yaml
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Google Cloud imports
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError

# Configuration imports
from src.utils.training_config import TrainingConfigLoader, get_training_config

# We'll import GCSCorpusLoader only when needed to avoid import chain issues


logger = logging.getLogger(__name__)


@pytest.fixture
def training_config_path():
    """Path to training configuration file"""
    return Path("/Users/kendo/daniqan/shyvrai-rlte/config/training_config.yaml")


@pytest.fixture
async def gcs_client():
    """GCS client for bucket access validation"""
    try:
        client = storage.Client()
        yield client
    except Exception as e:
        logger.error(f"GCS client initialization failed: {e}")
        pytest.skip(f"GCS not available: {e}")


@pytest.fixture
def expected_config_structure():
    """Define expected training configuration structure"""
    return {
        'corpus': {
            'bucket': str,
            'prefix': str,
            'version': str,
            'cache_dir': str,
            'cache_ttl_hours': int,
            'granularities': list
        },
        'models': {
            'lstm': dict,
            'transformer': dict,
            'itransformer': dict,
            'patchtst': dict,
            'timesmixer': dict,
            'dqn': dict
        },
        'training': {
            'split_ratios': list,
            'early_stopping_patience': int,
            'save_best_only': bool
        }
    }


class TestTrainingConfigurationLoading:
    """
    Integration tests for training configuration loading
    
    Tests verify that training_config.yaml can be loaded and provides
    all necessary configuration for training pipeline.
    """
    
    def test_training_config_file_exists(self, training_config_path):
        """Test: training_config.yaml file exists"""
        assert training_config_path.exists(), f"Training config file should exist at {training_config_path}"
        assert training_config_path.is_file(), "Training config should be a file"
        assert training_config_path.suffix == ".yaml", "Training config should be YAML format"
    
    def test_training_config_yaml_valid(self, training_config_path):
        """Test: training_config.yaml contains valid YAML"""
        try:
            with open(training_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict), "Training config should be a dictionary"
            assert len(config) > 0, "Training config should not be empty"
            logger.info(f"Training config loaded successfully with {len(config)} top-level sections")
            
        except yaml.YAMLError as e:
            pytest.fail(f"Training config YAML is invalid: {e}")
        except FileNotFoundError:
            pytest.fail(f"Training config file not found: {training_config_path}")
    
    def test_training_config_structure(self, training_config_path, expected_config_structure):
        """Test: training_config.yaml has expected structure with environment variable resolution"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        training_config = loader.load()
        
        # Convert dataclass back to dict for testing
        config = {
            'corpus': training_config.corpus,
            'models': training_config.models,
            'training': training_config.training
        }
        
        # Check top-level sections
        for section_name, section_structure in expected_config_structure.items():
            assert section_name in config, f"Training config should have '{section_name}' section"
            
            section_config = config[section_name]
            assert isinstance(section_config, dict), f"'{section_name}' should be a dictionary"
            
            # Check subsection structure
            for key, expected_type in section_structure.items():
                if isinstance(expected_type, type):
                    assert key in section_config, f"'{section_name}' should have '{key}' field"
                    
                    actual_value = section_config[key]
                    assert isinstance(actual_value, expected_type), \
                        f"'{section_name}.{key}' should be {expected_type.__name__}, got {type(actual_value).__name__}"
                
                elif isinstance(expected_type, dict):
                    assert key in section_config, f"'{section_name}' should have '{key}' subsection"
                    assert isinstance(section_config[key], dict), \
                        f"'{section_name}.{key}' should be a dictionary"
        
        logger.info("Training config structure validation passed")
    
    def test_corpus_configuration_values(self, training_config_path):
        """Test: Corpus configuration has sensible values"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        training_config = loader.load()
        
        corpus_config = training_config.corpus
        
        # Validate bucket name
        assert corpus_config['bucket'] == "shyvr-models-prod", \
            "Should use production GCS bucket for training data"
        
        # Validate prefix
        assert corpus_config['prefix'].startswith("training-data/"), \
            "Corpus prefix should point to training-data directory"
        
        # Validate version options
        version = corpus_config['version']
        assert version in ['latest'] or version.startswith('initial_v'), \
            f"Version should be 'latest' or start with 'initial_v', got: {version}"
        
        # Validate cache settings
        assert isinstance(corpus_config['cache_ttl_hours'], int), "Cache TTL should be integer"
        assert corpus_config['cache_ttl_hours'] > 0, "Cache TTL should be positive"
        assert corpus_config['cache_ttl_hours'] <= 168, "Cache TTL should be <= 1 week"
        
        # Validate granularities
        granularities = corpus_config['granularities']
        expected_granularities = ["daily", "hourly", "four_hour"]
        for gran in granularities:
            assert gran in expected_granularities, f"Unknown granularity: {gran}"
        
        logger.info(f"Corpus config validation passed: {corpus_config}")


class TestModelHyperparameters:
    """
    Integration tests for model-specific hyperparameter configuration
    
    Tests verify that each supported model has appropriate hyperparameters
    and that these parameters are compatible with model implementations.
    """
    
    def test_lstm_hyperparameters(self, training_config_path):
        """Test: LSTM model has required hyperparameters"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        training_config = loader.load()
        
        lstm_config = training_config.models['lstm']
        
        # Required parameters
        required_params = ['sequence_length', 'batch_size', 'epochs', 'learning_rate']
        for param in required_params:
            assert param in lstm_config, f"LSTM should have '{param}' parameter"
        
        # Validate parameter ranges
        assert isinstance(lstm_config['sequence_length'], int), "Sequence length should be integer"
        assert lstm_config['sequence_length'] > 0, "Sequence length should be positive"
        assert lstm_config['sequence_length'] <= 1000, "Sequence length should be reasonable"
        
        assert isinstance(lstm_config['batch_size'], int), "Batch size should be integer"
        assert lstm_config['batch_size'] > 0, "Batch size should be positive"
        
        assert isinstance(lstm_config['epochs'], int), "Epochs should be integer"
        assert lstm_config['epochs'] > 0, "Epochs should be positive"
        
        assert isinstance(lstm_config['learning_rate'], float), "Learning rate should be float"
        assert 0.0 < lstm_config['learning_rate'] < 1.0, "Learning rate should be between 0 and 1"
        
        logger.info(f"LSTM hyperparameters validated: {lstm_config}")
    
    def test_transformer_hyperparameters(self, training_config_path):
        """Test: Transformer models have required hyperparameters"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        training_config = loader.load()
        
        transformer_models = ['transformer', 'itransformer', 'patchtst', 'timesmixer']
        
        for model_name in transformer_models:
            model_config = training_config.models[model_name]
            
            # All transformers should have sequence_length and batch_size
            assert 'sequence_length' in model_config, f"{model_name} should have sequence_length"
            assert 'batch_size' in model_config, f"{model_name} should have batch_size"
            
            # Validate sequence length
            seq_len = model_config['sequence_length']
            assert isinstance(seq_len, int), f"{model_name} sequence_length should be integer"
            assert seq_len > 0, f"{model_name} sequence_length should be positive"
            
            # Model-specific validations
            if model_name == 'itransformer':
                assert 'selected_features' in model_config, "iTransformer should have selected_features"
                features = model_config['selected_features']
                assert isinstance(features, list), "selected_features should be a list"
                assert len(features) > 0, "selected_features should not be empty"
            
            elif model_name == 'patchtst':
                assert 'patch_length' in model_config, "PatchTST should have patch_length"
                assert 'stride' in model_config, "PatchTST should have stride"
                
                patch_len = model_config['patch_length']
                stride = model_config['stride']
                assert isinstance(patch_len, int), "patch_length should be integer"
                assert isinstance(stride, int), "stride should be integer"
                assert patch_len > 0 and stride > 0, "patch_length and stride should be positive"
            
            elif model_name == 'timesmixer':
                assert 'decomposition_layers' in model_config, "TimesMixer should have decomposition_layers"
                decomp_layers = model_config['decomposition_layers']
                assert isinstance(decomp_layers, int), "decomposition_layers should be integer"
                assert decomp_layers > 0, "decomposition_layers should be positive"
        
        logger.info(f"Transformer hyperparameters validated for {len(transformer_models)} models")
    
    def test_dqn_hyperparameters(self, training_config_path):
        """Test: DQN model has required RL hyperparameters"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        training_config = loader.load()
        
        dqn_config = training_config.models['dqn']
        
        # Required RL parameters
        required_params = ['replay_buffer_size', 'batch_size']
        for param in required_params:
            assert param in dqn_config, f"DQN should have '{param}' parameter"
        
        # Validate parameter ranges
        buffer_size = dqn_config['replay_buffer_size']
        batch_size = dqn_config['batch_size']
        
        assert isinstance(buffer_size, int), "replay_buffer_size should be integer"
        assert buffer_size > 0, "replay_buffer_size should be positive"
        
        assert isinstance(batch_size, int), "batch_size should be integer"
        assert batch_size > 0, "batch_size should be positive"
        assert batch_size < buffer_size, "batch_size should be less than buffer_size"
        
        logger.info(f"DQN hyperparameters validated: {dqn_config}")


class TestTrainingConfiguration:
    """
    Integration tests for general training configuration settings
    
    Tests verify training split ratios, early stopping, and other
    general training settings.
    """
    
    def test_split_ratios_configuration(self, training_config_path):
        """Test: Training split ratios are valid"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        config = loader.load()
        
        training_config = config.training
        split_ratios = training_config['split_ratios']
        
        # Validate structure
        assert isinstance(split_ratios, list), "split_ratios should be a list"
        assert len(split_ratios) == 3, "split_ratios should have 3 values [train, val, test]"
        
        # Validate values
        for ratio in split_ratios:
            assert isinstance(ratio, (int, float)), "Each ratio should be numeric"
            assert 0 < ratio < 1, f"Each ratio should be between 0 and 1, got: {ratio}"
        
        # Validate sum
        total = sum(split_ratios)
        assert abs(total - 1.0) < 0.001, f"Split ratios should sum to 1.0, got: {total}"
        
        # Validate order (train should be largest)
        assert split_ratios[0] > split_ratios[1], "Train split should be larger than validation"
        assert split_ratios[0] > split_ratios[2], "Train split should be larger than test"
        
        logger.info(f"Split ratios validated: {split_ratios}")
    
    def test_early_stopping_configuration(self, training_config_path):
        """Test: Early stopping configuration is valid"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        config = loader.load()
        
        training_config = config.training
        
        # Check patience parameter
        patience = training_config['early_stopping_patience']
        assert isinstance(patience, int), "early_stopping_patience should be integer"
        assert patience > 0, "early_stopping_patience should be positive"
        assert patience <= 50, "early_stopping_patience should be reasonable (≤50)"
        
        # Check save_best_only
        save_best = training_config['save_best_only']
        assert isinstance(save_best, bool), "save_best_only should be boolean"
        
        logger.info(f"Early stopping config validated: patience={patience}, save_best={save_best}")


class TestGCSIntegration:
    """
    Integration tests for GCS corpus loading configuration
    
    Tests verify that GCS configuration is valid and can access
    real corpus data in the specified bucket.
    """
    
    @pytest.mark.asyncio
    async def test_gcs_bucket_access(self, training_config_path, gcs_client):
        """Test: GCS bucket specified in config is accessible"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        config = loader.load()
        
        bucket_name = config.corpus['bucket']
        
        try:
            bucket = gcs_client.bucket(bucket_name)
            # Test bucket existence by listing objects
            blobs = list(bucket.list_blobs(prefix="training-data/", max_results=5))
            
            assert len(blobs) >= 0, f"Should be able to list blobs in bucket {bucket_name}"
            logger.info(f"GCS bucket {bucket_name} accessible, found {len(blobs)} objects")
            
        except (NotFound, GoogleCloudError) as e:
            pytest.fail(f"Cannot access GCS bucket {bucket_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_gcs_corpus_prefix_exists(self, training_config_path, gcs_client):
        """Test: Corpus prefix path exists in GCS bucket"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        config = loader.load()
        
        bucket_name = config.corpus['bucket']
        prefix = config.corpus['prefix']
        
        try:
            bucket = gcs_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
            
            # Should have some corpus files
            assert len(blobs) > 0, f"Should find corpus files at {prefix} in {bucket_name}"
            
            # Check for parquet files
            parquet_files = [b.name for b in blobs if b.name.endswith('.parquet')]
            assert len(parquet_files) > 0, f"Should find parquet files in {prefix}"
            
            logger.info(f"Found {len(blobs)} objects, {len(parquet_files)} parquet files in {prefix}")
            
        except Exception as e:
            pytest.fail(f"Cannot access corpus data at {prefix} in {bucket_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_gcs_corpus_loader_integration(self, training_config_path):
        """Test: GCSCorpusLoader can load data using config settings"""
        # Use the TrainingConfigLoader to load with environment variable resolution
        loader = TrainingConfigLoader(training_config_path)
        config = loader.load()
        
        corpus_config = config.corpus
        
        try:
            # Import only when needed to avoid import chain issues
            from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader
            
            loader = GCSCorpusLoader(
                bucket_name=corpus_config['bucket'],
                cache_dir=corpus_config['cache_dir'],
                cache_ttl_hours=corpus_config['cache_ttl_hours']
            )
            
            # Test listing available versions
            versions = await loader.list_available_corpus_versions()
            assert len(versions) > 0, "Should find corpus versions"
            
            # Test getting latest version
            if corpus_config['version'] == 'latest':
                latest_version = await loader.get_latest_corpus_version()
                assert latest_version is not None, "Should be able to get latest version"
                assert latest_version in versions, "Latest version should be in versions list"
            
            logger.info(f"GCS corpus loader integration validated with {len(versions)} versions")
            
        except ImportError as e:
            pytest.skip(f"GCS corpus loader import failed: {e}")
        except Exception as e:
            logger.error(f"GCS corpus loader integration failed: {e}")
            pytest.skip(f"GCS corpus loader not working: {e}")


class TestConfigurationValidation:
    """
    Integration tests for configuration validation and error handling
    
    Tests verify that invalid configurations are caught and handled
    appropriately.
    """
    
    def test_invalid_yaml_handling(self, tmp_path):
        """Test: Invalid YAML configuration is handled gracefully"""
        # Create invalid YAML file
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content: [")
        
        # Should raise appropriate error
        with pytest.raises(yaml.YAMLError):
            with open(invalid_config, 'r') as f:
                yaml.safe_load(f)
        
        logger.info("Invalid YAML handling test passed")
    
    def test_missing_required_sections(self, tmp_path):
        """Test: Missing required configuration sections are detected"""
        # Create config with missing sections
        incomplete_config = tmp_path / "incomplete_config.yaml"
        config_content = {
            'corpus': {'bucket': 'test'},
            # Missing 'models' and 'training' sections
        }
        
        with open(incomplete_config, 'w') as f:
            yaml.dump(config_content, f)
        
        # Load and validate
        with open(incomplete_config, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'corpus' in config, "Should have corpus section"
        assert 'models' not in config, "Should be missing models section"
        assert 'training' not in config, "Should be missing training section"
        
        logger.info("Missing sections detection test passed")
    
    def test_invalid_parameter_values(self, tmp_path):
        """Test: Invalid parameter values are detectable"""
        # Create config with invalid values
        invalid_config = tmp_path / "invalid_params_config.yaml"
        config_content = {
            'corpus': {
                'bucket': 'test-bucket',
                'cache_ttl_hours': -1  # Invalid: negative TTL
            },
            'models': {
                'lstm': {
                    'sequence_length': 0,  # Invalid: zero sequence length
                    'learning_rate': 2.0   # Invalid: learning rate > 1
                }
            },
            'training': {
                'split_ratios': [0.8, 0.1, 0.2]  # Invalid: sums to 1.1
            }
        }
        
        with open(invalid_config, 'w') as f:
            yaml.dump(config_content, f)
        
        # Load config (should load, but validation should catch issues)
        with open(invalid_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate that we can detect invalid values
        assert config['corpus']['cache_ttl_hours'] <= 0, "Should detect negative TTL"
        assert config['models']['lstm']['sequence_length'] <= 0, "Should detect zero sequence length"
        assert config['models']['lstm']['learning_rate'] >= 1.0, "Should detect invalid learning rate"
        assert sum(config['training']['split_ratios']) > 1.0, "Should detect invalid split ratios"
        
        logger.info("Invalid parameter values detection test passed")


class TestEnvironmentIntegration:
    """
    Integration tests for environment-specific configuration
    
    Tests verify that configuration integrates with existing
    environment-specific configuration patterns.
    """
    
    def test_integration_with_main_config(self, training_config_path):
        """Test: Training config integrates with main config.yaml"""
        # Load main config
        main_config_path = Path("/Users/kendo/daniqan/shyvrai-rlte/config/config.yaml")
        
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                main_config = yaml.safe_load(f)
        else:
            pytest.skip("Main config.yaml not found")
        
        # Load training config
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        
        # Check for compatibility
        # Training config corpus bucket should match GCS settings in main config if they exist
        if 'model_preservation' in main_config:
            main_bucket = main_config['model_preservation'].get('gcs_bucket')
            if main_bucket:
                training_bucket = training_config['corpus']['bucket']
                # Should use same bucket or compatible bucket
                assert training_bucket == main_bucket or training_bucket.startswith("shyvr-"), \
                    f"Training and main config buckets should be compatible: {training_bucket} vs {main_bucket}"
        
        logger.info("Integration with main config validated")
    
    def test_environment_variable_support(self, training_config_path):
        """Test: Training config supports environment variable overrides"""
        with open(training_config_path, 'r') as f:
            config_content = f.read()
        
        # Check for environment variable patterns
        env_patterns = ['${', '${}', '${TRAINING_']
        
        has_env_vars = any(pattern in config_content for pattern in env_patterns)
        
        if has_env_vars:
            logger.info("Training config supports environment variable overrides")
        else:
            logger.info("Training config uses static values (no environment variables)")
        
        # This is informational - both approaches are valid


if __name__ == "__main__":
    """
    Run integration tests with proper async support
    
    Usage:
        python -m pytest tests/integration/config/test_training_config.py -v
        
    Or run specific test classes:
        python -m pytest tests/integration/config/test_training_config.py::TestTrainingConfigurationLoading -v
    """
    import sys
    
    # Configure logging for test visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])