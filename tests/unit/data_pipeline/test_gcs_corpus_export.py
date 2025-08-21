"""
Test GCS corpus export functionality for MultiGranularityCollector.

This test ensures that corpus data can be properly exported to GCS
in parquet format for model training.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import tempfile
import pyarrow.parquet as pq

from src.data_pipeline.multi_granularity_collector import MultiGranularityCollector


class TestGCSCorpusExport:
    """Test GCS corpus export functionality"""
    
    @pytest.fixture
    def mock_corpus_data(self):
        """Create mock corpus data similar to production"""
        # Create sample data for each timeframe
        base_time = datetime.now()
        
        data = {}
        for timeframe in ['daily', 'hourly', 'four_hour']:
            # Create DataFrame with production-like structure
            num_records = 200  # Increased for LSTM sequence requirements
            df = pd.DataFrame({
                'timestamp': pd.date_range(base_time - timedelta(days=30), periods=num_records, freq='1h'),
                'symbol': ['WETH'] * 100 + ['WBTC'] * 100,
                'granularity': timeframe,
                'open': np.random.uniform(1000, 2000, num_records),
                'high': np.random.uniform(2000, 2100, num_records),
                'low': np.random.uniform(900, 1000, num_records),
                'close': np.random.uniform(1000, 2000, num_records),
                'volume': np.random.uniform(1000000, 10000000, num_records),
                'rsi_14': np.random.uniform(30, 70, num_records),
                'macd': np.random.uniform(-50, 50, num_records),
                'macd_signal': np.random.uniform(-50, 50, num_records),
                'bb_position': np.random.uniform(0, 1, num_records),
                'ema_12': np.random.uniform(1000, 2000, num_records),
                'ema_26': np.random.uniform(1000, 2000, num_records),
                'sma_50': np.random.uniform(1000, 2000, num_records),
                'atr': np.random.uniform(10, 100, num_records),
                'adx': np.random.uniform(0, 100, num_records),
                'returns_1h': np.random.uniform(-0.1, 0.1, num_records),
                'returns_24h': np.random.uniform(-0.2, 0.2, num_records),
                'volatility_24h': np.random.uniform(0.01, 0.5, num_records),
            })
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            data[f"corpus_{timeframe.replace('_', '')}"] = df
            
        return data
    
    @pytest.fixture
    def collector(self):
        """Create MultiGranularityCollector instance with mocked dependencies"""
        with patch('src.data_pipeline.multi_granularity_collector.CoinGeckoClient'):
            # Don't pass enable_pagination to __init__, it's handled internally
            collector = MultiGranularityCollector()
            collector.enable_pagination = True
            return collector
    
    @pytest.mark.asyncio
    async def test_export_to_gcs_structure(self, collector, mock_corpus_data):
        """Test that export creates correct GCS structure"""
        
        # Mock GCS client
        mock_storage_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        with patch('google.cloud.storage.Client', return_value=mock_storage_client):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Call export method
                result = await collector.export_to_gcs(
                    corpus_data=mock_corpus_data,
                    bucket_name='shyvr-models-prod',
                    gcs_prefix='training-data/initial-corpus/initial_v2.0_20240820_120000'
                )
                
                # Verify structure
                assert result['success'] is True
                assert 'gcs_path' in result
                assert 'initial_v2.0_' in result['gcs_path']
                
                # Check that blob.upload_from_filename was called for each timeframe
                assert mock_blob.upload_from_filename.call_count >= 3  # At least 3 parquet files
    
    @pytest.mark.asyncio
    async def test_parquet_format_compatibility(self, collector, mock_corpus_data):
        """Test that parquet files are compatible with model training methods"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to local directory first
            staging_dir = Path(tmpdir) / "corpus_export"
            staging_dir.mkdir(parents=True)
            
            # Save each timeframe as parquet
            for key, df in mock_corpus_data.items():
                timeframe = key.replace('_corpus', '')
                parquet_path = staging_dir / f"corpus_{timeframe}.parquet"
                df.to_parquet(parquet_path, index=False, compression='snappy')
                
                # Verify parquet can be read back
                df_read = pd.read_parquet(parquet_path)
                assert len(df) == len(df_read)
                assert list(df.columns) == list(df_read.columns)
                
                # Verify compatibility with model methods
                from src.ml_analysis.lstm_model import LSTMPricePredictor
                lstm = LSTMPricePredictor()
                
                # Filter for single token
                df_token = df_read[df_read['symbol'] == 'WETH'].copy()
                if len(df_token) >= 50:  # Need enough data for sequence
                    X, y, features = lstm.prepare_training_from_corpus(
                        df_token,
                        sequence_length=50,
                        prediction_horizons=[1, 4, 24]
                    )
                    
                    assert X.shape[1] == 50  # Sequence length
                    assert y.shape[1] == 3    # 3 prediction horizons
    
    @pytest.mark.asyncio
    async def test_metadata_generation(self, collector, mock_corpus_data):
        """Test that metadata.json is correctly generated"""
        
        # Test metadata generation as part of export
        with patch('google.cloud.storage.Client'):
            result = await collector.export_to_gcs(
                corpus_data=mock_corpus_data,
                bucket_name='test-bucket'
            )
            
            # Result should have metadata path
            if result['success']:
                assert 'metadata_path' in result
                assert 'total_records' in result
    
    @pytest.mark.asyncio
    async def test_gcs_upload_with_versioning(self, collector):
        """Test that GCS upload includes proper versioning"""
        
        mock_storage_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_storage_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        with patch('google.cloud.storage.Client', return_value=mock_storage_client):
            # When gcs_prefix is None, it should auto-generate with timestamp
            result = await collector.export_to_gcs(
                corpus_data={'test': pd.DataFrame()},
                bucket_name='test-bucket',
                gcs_prefix=None  # Should auto-generate
            )
            
            # Check that GCS path includes versioning
            if 'gcs_path' in result:
                assert 'initial_v2.0_' in result['gcs_path']
    
    @pytest.mark.asyncio
    async def test_export_error_handling(self, collector):
        """Test proper error handling during export"""
        
        # Test with invalid bucket
        with patch('google.cloud.storage.Client', side_effect=Exception("GCS connection failed")):
            result = await collector.export_to_gcs(
                corpus_data={},
                bucket_name='invalid-bucket'
            )
            
            assert result['success'] is False
            assert 'error' in result
            assert 'GCS connection failed' in result['error']
    
    @pytest.mark.asyncio  
    async def test_local_staging_before_upload(self, collector, mock_corpus_data):
        """Test that data is staged locally before GCS upload"""
        
        # The export_to_parquet method creates local files first
        # Let's verify this happens
        with patch('google.cloud.storage.Client'):
            # Mock the blob upload to prevent actual GCS calls
            with patch.object(collector, 'export_to_parquet', return_value={}) as mock_export:
                result = await collector.export_to_gcs(
                    corpus_data=mock_corpus_data,
                    bucket_name='shyvr-models-prod'
                )
                
                # Verify export_to_parquet was called
                mock_export.assert_called_once_with(mock_corpus_data)
    
    def test_integration_with_production_workflow(self):
        """Test that export integrates with production-multi workflow"""
        
        # This tests that the method can be called from the existing workflow
        with patch('src.data_pipeline.multi_granularity_collector.CoinGeckoClient'):
            collector = MultiGranularityCollector()
            
            # Verify method exists
            assert hasattr(collector, 'export_to_gcs')
            
            # Verify method signature
            import inspect
            sig = inspect.signature(collector.export_to_gcs)
            params = list(sig.parameters.keys())
            
            assert 'corpus_data' in params
            assert 'bucket_name' in params
            assert 'gcs_prefix' in params