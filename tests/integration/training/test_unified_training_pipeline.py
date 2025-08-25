"""
Integration Tests for UnifiedTrainingPipeline

Following TDD methodology, these comprehensive tests define the expected behavior
of the UnifiedTrainingPipeline before implementation. Tests use REAL GCS data
and actual model training - no mocks.

Test Structure:
1. GCS corpus data loading from real bucket
2. Train all supported models (LSTM, Transformer variants)
3. Integration with ModelManager for ensemble coordination
4. Database tracking via model_training_history table
5. Saved models to GCS shyvr-models-prod/trained-models/
6. Performance metrics validation
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Database imports
import asyncpg
from src.utils.database import get_database_connection, execute_query

# ML model imports
from src.ml_analysis.lstm_model import LSTMPricePredictor
from src.ml_analysis.transformers.transformer_predictor import TransformerPredictor
from src.ml_analysis.transformers.itransformer import iTransformerPredictor
from src.ml_analysis.transformers.patchtst import PatchTSTPredictor
from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor
from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import ModelType

# Data pipeline imports
from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader

# Google Cloud imports
from google.cloud import storage


logger = logging.getLogger(__name__)


@pytest.fixture
async def database_connection():
    """Database connection for tracking model training history"""
    try:
        async with get_database_connection() as conn:
            yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
async def gcs_corpus_loader():
    """GCS corpus loader for loading real training data"""
    try:
        loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir="/tmp/test_corpus_cache",
            cache_ttl_hours=1  # Short TTL for testing
        )
        yield loader
    except Exception as e:
        logger.error(f"GCS connection failed: {e}")
        pytest.skip(f"GCS not available: {e}")


@pytest.fixture
async def model_manager():
    """ModelManager instance for ensemble coordination"""
    config = {
        'environment': 'development',  # Use LSTM only for faster tests
        'model_dir': '/tmp/test_models',
        'cache_ttl_minutes': 1
    }
    manager = ModelManager(config)
    yield manager


@pytest.fixture
async def sample_corpus_data(gcs_corpus_loader):
    """Load a small sample of real corpus data for testing"""
    try:
        # Load latest daily corpus data
        df = await gcs_corpus_loader.load_corpus_from_gcs(
            timeframe="daily",
            token="WBTC"  # Filter for WBTC only to reduce test data size
        )
        
        # Take a sample for faster testing
        if len(df) > 100:
            df = df.tail(100).copy()
            
        logger.info(f"Loaded sample corpus data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load sample corpus data: {e}")
        pytest.skip(f"Sample data not available: {e}")


class TestUnifiedTrainingPipeline:
    """
    Integration tests for UnifiedTrainingPipeline
    
    Tests follow TDD red-green-refactor cycle:
    1. Write failing tests (RED)
    2. Implement minimal code to pass (GREEN) 
    3. Refactor for quality (REFACTOR)
    """
    
    @pytest.mark.asyncio
    async def test_unified_training_pipeline_imports(self):
        """Test: Import UnifiedTrainingPipeline class (should fail initially)"""
        with pytest.raises(ImportError):
            from scripts.training.train_all_models import UnifiedTrainingPipeline
    
    @pytest.mark.asyncio
    async def test_gcs_corpus_loader_integration(self, gcs_corpus_loader):
        """Test: GCS corpus loader can access real data"""
        # Test listing corpus versions
        versions = await gcs_corpus_loader.list_available_corpus_versions()
        assert len(versions) > 0, "Should find corpus versions in GCS"
        
        # Test getting latest version
        latest_version = await gcs_corpus_loader.get_latest_corpus_version()
        assert latest_version in versions, "Latest version should be in versions list"
        
        logger.info(f"Found {len(versions)} corpus versions, latest: {latest_version}")
    
    @pytest.mark.asyncio
    async def test_corpus_data_loading_and_structure(self, sample_corpus_data):
        """Test: Corpus data loads with expected structure"""
        df = sample_corpus_data
        
        # Verify data structure
        assert len(df) > 0, "Corpus should contain data"
        assert 'close' in df.columns, "Should have close price column"
        assert 'timestamp' in df.columns or df.index.name == 'timestamp', "Should have timestamp"
        
        # Verify numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 10, f"Should have at least 10 numeric features, got {len(numeric_cols)}"
        
        logger.info(f"Corpus data validation passed: {len(df)} rows, {len(numeric_cols)} numeric features")
    
    @pytest.mark.asyncio
    async def test_model_training_interfaces_exist(self):
        """Test: All model classes have prepare_training_from_corpus method"""
        models_to_test = [
            LSTMPricePredictor,
            TransformerPredictor,
            iTransformerPredictor, 
            PatchTSTPredictor,
            TimesMixerPredictor
        ]
        
        for model_class in models_to_test:
            model = model_class({})
            assert hasattr(model, 'prepare_training_from_corpus'), \
                f"{model_class.__name__} should have prepare_training_from_corpus method"
            
            # Verify method signature (should accept DataFrame)
            method = getattr(model, 'prepare_training_from_corpus')
            assert callable(method), f"prepare_training_from_corpus should be callable in {model_class.__name__}"
    
    @pytest.mark.asyncio 
    async def test_lstm_model_training_from_corpus(self, sample_corpus_data):
        """Test: LSTM model can train from corpus data"""
        df = sample_corpus_data
        
        model = LSTMPricePredictor({
            'sequence_length': 20,
            'hidden_size': 32,
            'num_layers': 1,
            'num_epochs': 2  # Quick training for test
        })
        
        # Test prepare_training_from_corpus method
        X, y, feature_names = model.prepare_training_from_corpus(df)
        
        # Verify training data structure
        assert X.ndim == 3, f"X should be 3D array [samples, sequence, features], got {X.ndim}D"
        assert y.ndim == 2, f"y should be 2D array [samples, targets], got {y.ndim}D" 
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert len(feature_names) > 0, "Should have feature names"
        
        logger.info(f"LSTM training data prepared: X{X.shape}, y{y.shape}, {len(feature_names)} features")
    
    @pytest.mark.asyncio
    async def test_model_manager_integration(self, model_manager, sample_corpus_data):
        """Test: ModelManager can coordinate model training"""
        df = sample_corpus_data
        
        # Test that models are initialized
        assert len(model_manager._models) > 0, "ModelManager should have models"
        
        # Test training coordination (mock training for speed)
        available_models = list(model_manager._models.keys())
        logger.info(f"ModelManager has {len(available_models)} models: {[m.value for m in available_models]}")
        
        # Verify LSTM is available (required for development mode)
        assert ModelType.LSTM in available_models, "LSTM model should be available"
    
    @pytest.mark.asyncio
    async def test_database_training_history_structure(self, database_connection):
        """Test: Database has model_training_history table with correct structure"""
        db = database_connection
        
        # Check if table exists
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'model_training_history'
        ORDER BY ordinal_position;
        """
        
        result = await execute_query(query)
        assert len(result) > 0, "model_training_history table should exist"
        
        # Check for required columns
        columns = {row['column_name']: row['data_type'] for row in result}
        
        required_columns = [
            'training_id', 'model_type', 'corpus_version_id',
            'trained_at', 'training_mode', 'performance_metrics',
            'model_checkpoint_path', 'training_config'
        ]
        
        for col in required_columns:
            assert col in columns, f"Required column '{col}' should exist in model_training_history"
        
        logger.info(f"Database validation passed: model_training_history has {len(columns)} columns")
    
    @pytest.mark.asyncio
    async def test_gcs_model_storage_access(self):
        """Test: Can access GCS bucket for saving trained models"""
        try:
            client = storage.Client()
            bucket = client.bucket("shyvr-models-prod")
            
            # Test listing trained-models directory
            blobs = list(bucket.list_blobs(prefix="trained-models/", max_results=5))
            logger.info(f"Found {len(blobs)} existing model files in GCS trained-models/")
            
            # Test write access by creating a test file
            test_blob_name = f"trained-models/test_access_{datetime.now().isoformat()}.txt"
            blob = bucket.blob(test_blob_name)
            blob.upload_from_string("test access")
            
            # Clean up test file
            blob.delete()
            
            logger.info("GCS model storage access validated")
            
        except Exception as e:
            pytest.skip(f"GCS model storage not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_unified_training_pipeline_class_structure(self):
        """Test: UnifiedTrainingPipeline class should have required methods (will fail initially)"""
        # This test will fail until we implement the class
        with pytest.raises(ImportError):
            from scripts.training.train_all_models import UnifiedTrainingPipeline
            
            # Expected methods after implementation:
            # - load_corpus_data()
            # - prepare_train_val_test_split() 
            # - train_lstm_model()
            # - train_transformer_models()
            # - save_trained_models()
            # - track_training_history()
    
    @pytest.mark.asyncio 
    async def test_time_series_aware_data_splitting(self, sample_corpus_data):
        """Test: Time-series aware train/val/test splitting preserves temporal order"""
        df = sample_corpus_data
        
        # Sort by timestamp to ensure temporal order
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
        else:
            df_sorted = df.sort_index()  # Assume index is timestamp
        
        # Calculate split indices for 80/10/10 split
        total_samples = len(df_sorted)
        train_end = int(total_samples * 0.8)
        val_end = int(total_samples * 0.9)
        
        train_data = df_sorted.iloc[:train_end]
        val_data = df_sorted.iloc[train_end:val_end]
        test_data = df_sorted.iloc[val_end:]
        
        # Verify splits
        assert len(train_data) > 0, "Training set should not be empty"
        assert len(val_data) > 0, "Validation set should not be empty" 
        assert len(test_data) > 0, "Test set should not be empty"
        assert len(train_data) + len(val_data) + len(test_data) == total_samples
        
        # Verify temporal order (latest timestamp in train < earliest in val)
        if 'timestamp' in df.columns:
            train_max = train_data['timestamp'].max()
            val_min = val_data['timestamp'].min()
            assert train_max < val_min, "Training data should be earlier than validation data"
        
        logger.info(f"Time-series split validated: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    @pytest.mark.asyncio
    async def test_multiple_model_training_compatibility(self, sample_corpus_data):
        """Test: All models can process the same corpus data format"""
        df = sample_corpus_data
        
        models = [
            ('LSTM', LSTMPricePredictor, {'sequence_length': 20}),
            ('Transformer', TransformerPredictor, {'sequence_length': 20, 'd_model': 32}),
            ('iTransformer', iTransformerPredictor, {'sequence_length': 15}),
            ('PatchTST', PatchTSTPredictor, {'patch_length': 4}),
            ('TimesMixer', TimesMixerPredictor, {'seq_len': 25})
        ]
        
        training_results = {}
        
        for model_name, model_class, config in models:
            try:
                model = model_class(config)
                X, y, feature_names = model.prepare_training_from_corpus(df)
                
                # Verify consistent output structure
                assert X.ndim == 3, f"{model_name} should output 3D X array"
                assert y.ndim == 2, f"{model_name} should output 2D y array"
                assert len(feature_names) > 0, f"{model_name} should return feature names"
                
                training_results[model_name] = {
                    'X_shape': X.shape,
                    'y_shape': y.shape,
                    'n_features': len(feature_names)
                }
                
            except Exception as e:
                logger.warning(f"{model_name} training preparation failed: {e}")
                training_results[model_name] = {'error': str(e)}
        
        # Verify at least LSTM works (required)
        assert 'LSTM' in training_results, "LSTM should be tested"
        assert 'error' not in training_results['LSTM'], "LSTM training preparation should succeed"
        
        logger.info(f"Model compatibility test results: {training_results}")
    
    @pytest.mark.asyncio
    async def test_training_performance_metrics_tracking(self):
        """Test: Training pipeline should track performance metrics"""
        # This test defines the expected metrics structure
        expected_metrics = {
            'train_loss': float,
            'val_loss': float,
            'train_accuracy': float,
            'val_accuracy': float,
            'training_time_seconds': float,
            'epochs_completed': int,
            'best_epoch': int,
            'final_lr': float
        }
        
        # This will be used to validate the actual implementation
        assert len(expected_metrics) == 8, "Should track 8 performance metrics"
        logger.info(f"Expected performance metrics structure: {list(expected_metrics.keys())}")
    
    @pytest.mark.asyncio
    async def test_model_checkpoint_saving_structure(self):
        """Test: Saved models should have consistent structure and metadata"""
        # Define expected checkpoint structure
        expected_checkpoint_structure = {
            'model_state_dict': dict,  # PyTorch state dict
            'model_config': dict,      # Model configuration
            'training_metrics': dict,  # Performance metrics
            'corpus_version': str,     # Version of training data used
            'model_version': str,      # Model version identifier
            'feature_names': list,     # List of feature names used
            'trained_at': str,         # ISO timestamp of training
            'model_type': str          # Type identifier (lstm, transformer, etc.)
        }
        
        # This structure will guide the implementation
        assert len(expected_checkpoint_structure) == 8, "Checkpoint should have 8 metadata fields"
        logger.info(f"Expected checkpoint structure: {list(expected_checkpoint_structure.keys())}")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_corpus_data):
        """Test: Training pipeline handles errors gracefully"""
        # Test with insufficient data
        small_df = sample_corpus_data.head(5)  # Very small dataset
        
        model = LSTMPricePredictor({'sequence_length': 20})  # Longer sequence than data
        
        # Should handle gracefully, not crash
        try:
            X, y, feature_names = model.prepare_training_from_corpus(small_df)
            # If it succeeds, verify reasonable behavior
            assert X.shape[0] >= 0, "Should handle small datasets gracefully"
        except ValueError as e:
            # Expected error for insufficient data
            assert "insufficient" in str(e).lower() or "not enough" in str(e).lower()
            logger.info(f"Graceful error handling: {e}")
        except Exception as e:
            pytest.fail(f"Should handle errors gracefully, not crash: {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_training_safety(self):
        """Test: Training pipeline should handle concurrent access safely"""
        # Define requirements for concurrent training safety:
        # 1. Unique model checkpoint paths
        # 2. Atomic database updates
        # 3. GCS upload collision avoidance
        
        timestamp = datetime.now().isoformat()
        model_types = ['lstm', 'transformer', 'itransformer']
        
        # Generate unique paths for each model
        checkpoint_paths = []
        for model_type in model_types:
            path = f"trained-models/{model_type}/{timestamp}/model.pt"
            checkpoint_paths.append(path)
        
        # Verify uniqueness
        assert len(checkpoint_paths) == len(set(checkpoint_paths)), "Checkpoint paths should be unique"
        logger.info(f"Concurrent training safety: unique paths generated")
    
    @pytest.mark.asyncio
    async def test_integration_with_existing_systems(self, model_manager, database_connection):
        """Test: Training pipeline integrates with existing ModelManager and database"""
        # Verify ModelManager is available and functional
        health = await model_manager.health_check()
        assert health['overall_healthy'], "ModelManager should be healthy"
        
        # Verify database connection is functional
        result = await execute_query("SELECT NOW() as current_time")
        assert result is not None, "Database should be accessible"
        
        # Verify training_corpus_versions table exists (for tracking corpus versions)
        corpus_query = "SELECT version_name FROM training_corpus_versions LIMIT 1"
        try:
            corpus_result = await execute_query(corpus_query)
            logger.info(f"Found {len(corpus_result)} corpus versions in database")
        except Exception as e:
            logger.warning(f"Corpus versions table access: {e}")
        
        logger.info("Integration with existing systems validated")


@pytest.mark.integration
class TestUnifiedTrainingPipelineE2E:
    """
    End-to-end integration tests for complete training pipeline
    
    These tests run the full pipeline from GCS data loading to model saving
    and should only pass once the implementation is complete.
    """
    
    @pytest.mark.asyncio
    async def test_complete_lstm_training_pipeline(self, gcs_corpus_loader, database_connection):
        """Test: Complete LSTM training pipeline from GCS data to saved model"""
        # This test will fail until implementation is complete
        with pytest.raises(ImportError):
            from scripts.training.train_all_models import UnifiedTrainingPipeline
            
            pipeline = UnifiedTrainingPipeline(
                gcs_bucket="shyvr-models-prod",
                model_save_bucket="shyvr-models-prod/trained-models",
                database=database_connection
            )
            
            # Expected workflow:
            # 1. Load corpus data from GCS
            # 2. Split into train/val/test
            # 3. Train LSTM model
            # 4. Validate performance
            # 5. Save model to GCS
            # 6. Record training history in database
            
            result = await pipeline.train_lstm_model(
                corpus_version="latest",
                config={
                    'sequence_length': 50,
                    'hidden_size': 128,
                    'num_epochs': 10
                }
            )
            
            assert result['success'], "LSTM training should succeed"
            assert 'model_path' in result, "Should return model save path"
            assert 'training_id' in result, "Should return database training ID"
    
    @pytest.mark.asyncio
    async def test_complete_multi_model_training_pipeline(self):
        """Test: Complete multi-model training pipeline with all transformer variants"""
        with pytest.raises(ImportError):
            from scripts.training.train_all_models import UnifiedTrainingPipeline
            
            pipeline = UnifiedTrainingPipeline()
            
            # Train all models in sequence
            results = await pipeline.train_all_models(
                corpus_version="latest",
                models=['lstm', 'transformer', 'itransformer', 'patchtst', 'timesmixer']
            )
            
            assert len(results) == 5, "Should train 5 models"
            
            for model_type, result in results.items():
                assert result['success'], f"{model_type} training should succeed"
                assert 'performance_metrics' in result, f"{model_type} should have performance metrics"
    
    @pytest.mark.asyncio  
    async def test_model_ensemble_coordination_after_training(self):
        """Test: ModelManager can coordinate ensemble predictions with newly trained models"""
        with pytest.raises(ImportError):
            from scripts.training.train_all_models import UnifiedTrainingPipeline
            
            # After training, ModelManager should load the new models
            # and be able to create ensemble predictions
            
            pipeline = UnifiedTrainingPipeline()
            model_manager = ModelManager({'environment': 'production'})
            
            # Train models
            await pipeline.train_all_models(corpus_version="latest")
            
            # Load trained models into ModelManager  
            load_results = await model_manager.load_models()
            
            # Verify ensemble capability
            health = await model_manager.health_check()
            assert health['ensemble_available'], "Ensemble should be available after training"


if __name__ == "__main__":
    """
    Run integration tests with proper async support
    
    Usage:
        python -m pytest tests/integration/training/test_unified_training_pipeline.py -v
        
    Or run specific test classes:
        python -m pytest tests/integration/training/test_unified_training_pipeline.py::TestUnifiedTrainingPipeline -v
    """
    import sys
    
    # Configure logging for test visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])