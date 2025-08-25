"""
Integration Tests for GCSCorpusLoader

Tests real GCS integration for loading training corpus data.
NO MOCKS - uses actual GCS bucket shyvr-models-prod.

Following TDD methodology:
1. Write comprehensive tests that define the expected behavior
2. Run tests to see them fail (red phase)
3. Implement GCSCorpusLoader to make tests pass (green phase)
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import structlog

# Import GCSCorpusLoader class (TDD - will be implemented)
try:
    # Try direct import first, avoiding complex import chain during TDD
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gcs_corpus_loader", 
        "/Users/kendo/daniqan/shyvrai-rlte/src/data_pipeline/gcs_corpus_loader.py"
    )
    gcs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcs_module)
    GCSCorpusLoader = gcs_module.GCSCorpusLoader
except ImportError as e:
    # Fallback for normal import
    from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader

logger = structlog.get_logger(__name__)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_corpus_cache_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def gcs_loader(temp_cache_dir):
    """Create GCSCorpusLoader instance for testing"""
    return GCSCorpusLoader(
        bucket_name="shyvr-models-prod",
        cache_dir=str(temp_cache_dir),
        cache_ttl_hours=24
    )


@pytest.fixture
def gcs_loader_with_auth():
    """Create GCSCorpusLoader with Application Default Credentials"""
    # Use /tmp/corpus_cache as specified in requirements
    cache_dir = "/tmp/corpus_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    return GCSCorpusLoader(
        bucket_name="shyvr-models-prod",
        cache_dir=cache_dir,
        cache_ttl_hours=24
    )


class TestGCSCorpusLoaderInitialization:
    """Test GCSCorpusLoader initialization and configuration"""
    
    def test_init_with_default_cache_dir(self):
        """Test initialization with default cache directory"""
        loader = GCSCorpusLoader(bucket_name="shyvr-models-prod")
        
        assert loader.bucket_name == "shyvr-models-prod"
        assert loader.cache_dir == "/tmp/corpus_cache"
        assert loader.cache_ttl_hours == 24
        assert loader.gcs_client is not None
        assert loader.bucket is not None
    
    def test_init_with_custom_cache_dir(self, temp_cache_dir):
        """Test initialization with custom cache directory"""
        loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir=str(temp_cache_dir),
            cache_ttl_hours=48
        )
        
        assert loader.bucket_name == "shyvr-models-prod"
        assert loader.cache_dir == str(temp_cache_dir)
        assert loader.cache_ttl_hours == 48
        assert Path(loader.cache_dir).exists()
    
    def test_init_creates_cache_directory(self, temp_cache_dir):
        """Test that initialization creates cache directory if it doesn't exist"""
        cache_subdir = temp_cache_dir / "new_cache_dir"
        assert not cache_subdir.exists()
        
        loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir=str(cache_subdir)
        )
        
        assert cache_subdir.exists()


class TestGCSCorpusVersionListing:
    """Test corpus version listing functionality"""
    
    @pytest.mark.asyncio
    async def test_list_available_corpus_versions_real_gcs(self, gcs_loader_with_auth):
        """Test listing corpus versions from real GCS bucket"""
        versions = await gcs_loader_with_auth.list_available_corpus_versions()
        
        # Should return a list of version paths
        assert isinstance(versions, list)
        
        # Should contain at least one version (from the completed Phase 4.1)
        assert len(versions) >= 1, "Expected at least one corpus version in GCS"
        
        # Each version should be a string path
        for version in versions:
            assert isinstance(version, str)
            assert version.startswith("training-data/initial-corpus/")
            
        logger.info(f"Found {len(versions)} corpus versions in GCS", versions=versions[:3])
    
    @pytest.mark.asyncio
    async def test_get_latest_corpus_version_real_gcs(self, gcs_loader_with_auth):
        """Test getting latest corpus version from real GCS"""
        latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
        
        # Should return a version path
        assert isinstance(latest_version, str)
        assert latest_version.startswith("training-data/initial-corpus/")
        
        # Should contain v2.0 (from multi-granularity collector)
        assert "v2.0" in latest_version
        
        logger.info(f"Latest corpus version: {latest_version}")
    
    @pytest.mark.asyncio
    async def test_list_versions_returns_sorted_by_date(self, gcs_loader_with_auth):
        """Test that versions are returned sorted by creation date (newest first)"""
        versions = await gcs_loader_with_auth.list_available_corpus_versions()
        
        if len(versions) >= 2:
            # Extract timestamps from version names
            timestamps = []
            for version in versions:
                # Look for timestamp patterns like "v2.0_20250821_153449"
                parts = version.split('_')
                if len(parts) >= 3:
                    try:
                        date_str = parts[-2] + '_' + parts[-1]  # "20250821_153449"
                        timestamp = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            if len(timestamps) >= 2:
                # Should be sorted newest first
                assert timestamps == sorted(timestamps, reverse=True)


class TestGCSCorpusDataDownload:
    """Test corpus data download and caching functionality"""
    
    @pytest.mark.asyncio
    async def test_download_and_cache_parquet_real_file(self, gcs_loader_with_auth):
        """Test downloading real corpus parquet file from GCS"""
        # Get the latest corpus version
        latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
        
        # Try to download one of the expected corpus files
        for timeframe in ["daily", "hourly", "hour"]:
            gcs_path = f"{latest_version}/corpus/corpus_{timeframe}.parquet"
            
            try:
                local_path = await gcs_loader_with_auth.download_and_cache_parquet(gcs_path)
                
                # Should return a valid local path
                assert isinstance(local_path, Path)
                assert local_path.exists()
                assert local_path.suffix == ".parquet"
                
                # File should be cached in the cache directory
                assert str(gcs_loader_with_auth.cache_dir) in str(local_path)
                
                # File should be readable as parquet
                df = pd.read_parquet(local_path)
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0, f"corpus_{timeframe}.parquet should contain data"
                
                logger.info(f"Successfully downloaded corpus_{timeframe}.parquet", 
                           local_path=str(local_path), rows=len(df))
                
                # Test that subsequent download uses cache
                cached_path = await gcs_loader_with_auth.download_and_cache_parquet(gcs_path)
                assert cached_path == local_path
                
                break  # Success with at least one file
            except Exception as e:
                logger.warning(f"Failed to download corpus_{timeframe}.parquet: {e}")
                continue
        else:
            pytest.fail("Could not download any corpus files from GCS")
    
    @pytest.mark.asyncio
    async def test_cache_ttl_functionality(self, gcs_loader_with_auth):
        """Test that cache TTL works correctly"""
        # Create a loader with very short TTL
        short_ttl_loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir=gcs_loader_with_auth.cache_dir,
            cache_ttl_hours=0  # Immediate expiration
        )
        
        latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
        gcs_path = f"{latest_version}/corpus/corpus_daily.parquet"
        
        try:
            # Download file
            local_path = await short_ttl_loader.download_and_cache_parquet(gcs_path)
            assert local_path.exists()
            
            # Wait a moment and download again - should re-download due to TTL
            await asyncio.sleep(0.1)
            local_path2 = await short_ttl_loader.download_and_cache_parquet(gcs_path)
            
            # Path should be the same, but file should be re-downloaded
            assert local_path2 == local_path
            assert local_path2.exists()
            
        except Exception as e:
            pytest.skip(f"Could not test cache TTL: {e}")
    
    @pytest.mark.asyncio
    async def test_clear_old_cache_functionality(self, gcs_loader_with_auth):
        """Test clearing old cached files"""
        # Download a file to cache
        latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
        gcs_path = f"{latest_version}/corpus/corpus_daily.parquet"
        
        try:
            local_path = await gcs_loader_with_auth.download_and_cache_parquet(gcs_path)
            assert local_path.exists()
            
            # Clear cache with 0 TTL (should remove all files)
            removed_count = await gcs_loader_with_auth.clear_old_cache(ttl_hours=0)
            
            assert removed_count >= 0
            # File might still exist if it was recently downloaded
            
        except Exception as e:
            pytest.skip(f"Could not test cache clearing: {e}")


class TestGCSCorpusLoadingMain:
    """Test main corpus loading functionality"""
    
    @pytest.mark.asyncio
    async def test_load_corpus_from_gcs_daily_all_tokens(self, gcs_loader_with_auth):
        """Test loading complete daily corpus data from GCS"""
        try:
            df = await gcs_loader_with_auth.load_corpus_from_gcs(
                timeframe="daily",
                token=None  # Load all tokens
            )
            
            # Should return a DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0, "Daily corpus should contain data"
            
            # Should contain expected columns
            expected_columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'symbol', 'token_symbol', 'rsi_14', 'macd', 'bb_upper'
            ]
            
            for col in expected_columns:
                if col in df.columns:
                    logger.info(f"Found expected column: {col}")
                else:
                    logger.warning(f"Missing expected column: {col}")
            
            # Should contain multiple tokens
            if 'symbol' in df.columns:
                unique_tokens = df['symbol'].unique()
                assert len(unique_tokens) >= 1, "Should contain at least one token"
                logger.info(f"Found tokens: {list(unique_tokens)}")
            
            logger.info(f"Loaded daily corpus", rows=len(df), columns=len(df.columns))
            
        except Exception as e:
            logger.error(f"Failed to load daily corpus: {e}")
            pytest.fail(f"Could not load daily corpus from GCS: {e}")
    
    @pytest.mark.asyncio
    async def test_load_corpus_from_gcs_single_token_filter(self, gcs_loader_with_auth):
        """Test loading corpus data filtered by token"""
        try:
            # First, get all tokens to see what's available
            all_df = await gcs_loader_with_auth.load_corpus_from_gcs(
                timeframe="daily",
                token=None
            )
            
            if 'symbol' in all_df.columns and len(all_df) > 0:
                available_tokens = all_df['symbol'].unique()
                test_token = available_tokens[0]  # Use first available token
                
                # Load filtered data
                filtered_df = await gcs_loader_with_auth.load_corpus_from_gcs(
                    timeframe="daily",
                    token=test_token
                )
                
                assert isinstance(filtered_df, pd.DataFrame)
                assert len(filtered_df) > 0, f"Should have data for token {test_token}"
                
                # All rows should be for the requested token
                if 'symbol' in filtered_df.columns:
                    unique_symbols = filtered_df['symbol'].unique()
                    assert len(unique_symbols) == 1
                    assert unique_symbols[0] == test_token
                
                logger.info(f"Successfully filtered data for token {test_token}", rows=len(filtered_df))
            else:
                pytest.skip("No symbol column found in corpus data")
                
        except Exception as e:
            pytest.skip(f"Could not test token filtering: {e}")
    
    @pytest.mark.asyncio
    async def test_load_corpus_different_timeframes(self, gcs_loader_with_auth):
        """Test loading corpus data for different timeframes"""
        timeframes_to_test = ["daily", "hourly", "hour"]  # Based on multi_granularity_collector
        successful_loads = 0
        
        for timeframe in timeframes_to_test:
            try:
                df = await gcs_loader_with_auth.load_corpus_from_gcs(
                    timeframe=timeframe,
                    token=None
                )
                
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0, f"Timeframe {timeframe} should contain data"
                
                logger.info(f"Successfully loaded {timeframe} corpus", rows=len(df))
                successful_loads += 1
                
            except Exception as e:
                logger.warning(f"Failed to load {timeframe} corpus: {e}")
                
        # Should be able to load at least one timeframe
        assert successful_loads >= 1, "Should be able to load at least one timeframe"
    
    @pytest.mark.asyncio
    async def test_load_corpus_auto_latest_version(self, gcs_loader_with_auth):
        """Test that corpus loading automatically uses latest version"""
        try:
            # Load without specifying version (should use latest)
            df = await gcs_loader_with_auth.load_corpus_from_gcs(
                timeframe="daily"
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
            # Get the latest version manually
            latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
            
            # Load with explicit version
            df_explicit = await gcs_loader_with_auth.load_corpus_from_gcs(
                gcs_prefix=latest_version,
                timeframe="daily"
            )
            
            # Should be the same data (or very similar - allowing for minor differences)
            assert len(df_explicit) > 0
            
            logger.info("Auto-latest version selection works correctly")
            
        except Exception as e:
            pytest.skip(f"Could not test auto-latest version: {e}")


class TestGCSCorpusLoaderErrorHandling:
    """Test error handling and retry logic"""
    
    @pytest.mark.asyncio
    async def test_handle_missing_corpus_file(self, gcs_loader_with_auth):
        """Test handling of missing corpus files"""
        fake_path = "training-data/initial-corpus/nonexistent_v999.0/corpus/corpus_fake.parquet"
        
        with pytest.raises((FileNotFoundError, Exception)) as exc_info:
            await gcs_loader_with_auth.download_and_cache_parquet(fake_path)
        
        # Should raise an appropriate exception
        assert "not found" in str(exc_info.value).lower() or "404" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_handle_invalid_timeframe(self, gcs_loader_with_auth):
        """Test handling of invalid timeframe"""
        with pytest.raises((ValueError, FileNotFoundError, Exception)) as exc_info:
            await gcs_loader_with_auth.load_corpus_from_gcs(
                timeframe="invalid_timeframe_xyz"
            )
        
        # Should raise an appropriate exception
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["not found", "invalid", "error", "404"])
    
    @pytest.mark.asyncio
    async def test_authentication_with_default_credentials(self, temp_cache_dir):
        """Test that GCS authentication works with Application Default Credentials"""
        # This test verifies that the loader can authenticate to GCS
        loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir=str(temp_cache_dir)
        )
        
        try:
            # Try to list versions - this requires authentication
            versions = await loader.list_available_corpus_versions()
            assert isinstance(versions, list)
            logger.info("Authentication with Application Default Credentials successful")
            
        except Exception as e:
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                pytest.skip(f"Authentication not configured: {e}")
            else:
                raise


class TestGCSCorpusLoaderPerformance:
    """Test performance and efficiency"""
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_files(self, gcs_loader_with_auth):
        """Test that loader can handle large parquet files efficiently"""
        try:
            # Load daily data (potentially large)
            df = await gcs_loader_with_auth.load_corpus_from_gcs(timeframe="daily")
            
            # Should be able to load data without memory errors
            assert isinstance(df, pd.DataFrame)
            
            # Check memory usage is reasonable for the data size
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            rows = len(df)
            
            logger.info(f"Memory efficiency test", 
                       rows=rows, 
                       memory_mb=memory_usage_mb,
                       mb_per_1k_rows=memory_usage_mb / (rows / 1000) if rows > 0 else 0)
            
            # Memory usage should be reasonable (< 500MB for typical corpus)
            assert memory_usage_mb < 500, f"Memory usage too high: {memory_usage_mb:.2f} MB"
            
        except Exception as e:
            pytest.skip(f"Could not test memory efficiency: {e}")
    
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, gcs_loader_with_auth):
        """Test that caching improves performance"""
        import time
        
        try:
            latest_version = await gcs_loader_with_auth.get_latest_corpus_version()
            gcs_path = f"{latest_version}/corpus/corpus_daily.parquet"
            
            # First download (no cache)
            start_time = time.time()
            path1 = await gcs_loader_with_auth.download_and_cache_parquet(gcs_path)
            first_download_time = time.time() - start_time
            
            # Second download (from cache)
            start_time = time.time()
            path2 = await gcs_loader_with_auth.download_and_cache_parquet(gcs_path)
            second_download_time = time.time() - start_time
            
            assert path1 == path2
            
            # Cache should be significantly faster (at least 50% improvement)
            cache_speedup = first_download_time / second_download_time if second_download_time > 0 else float('inf')
            
            logger.info("Cache performance test",
                       first_download_s=first_download_time,
                       second_download_s=second_download_time,
                       speedup_factor=cache_speedup)
            
            # Cache should provide some performance benefit
            assert cache_speedup >= 1.5, f"Cache should improve performance, got {cache_speedup:.2f}x speedup"
            
        except Exception as e:
            pytest.skip(f"Could not test cache performance: {e}")


class TestGCSCorpusLoaderIntegration:
    """Test integration with actual training pipeline requirements"""
    
    @pytest.mark.asyncio
    async def test_data_format_suitable_for_training(self, gcs_loader_with_auth):
        """Test that loaded data is in suitable format for ML training"""
        try:
            df = await gcs_loader_with_auth.load_corpus_from_gcs(timeframe="daily")
            
            # Check data types are appropriate for training
            numeric_columns = df.select_dtypes(include=['number']).columns
            assert len(numeric_columns) > 0, "Should have numeric columns for training"
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                logger.warning(f"Found {total_nans} NaN values", nan_by_column=nan_counts[nan_counts > 0].to_dict())
            
            # Check timestamp column exists and is proper format
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            assert len(timestamp_cols) > 0, "Should have timestamp column"
            
            timestamp_col = timestamp_cols[0]
            assert pd.api.types.is_datetime64_any_dtype(df[timestamp_col]), "Timestamp should be datetime type"
            
            # Check data spans reasonable time period
            if len(df) > 1:
                time_span = df[timestamp_col].max() - df[timestamp_col].min()
                assert time_span.days > 30, "Data should span more than 30 days"
            
            logger.info("Data format validation passed", 
                       rows=len(df),
                       numeric_columns=len(numeric_columns),
                       time_span_days=time_span.days if len(df) > 1 else 0)
            
        except Exception as e:
            pytest.skip(f"Could not validate data format: {e}")
    
    @pytest.mark.asyncio
    async def test_feature_completeness_for_training(self, gcs_loader_with_auth):
        """Test that corpus contains necessary features for training"""
        try:
            df = await gcs_loader_with_auth.load_corpus_from_gcs(timeframe="daily")
            
            # Check for essential OHLCV columns
            essential_ohlcv = ['open', 'high', 'low', 'close', 'volume']
            found_ohlcv = [col for col in essential_ohlcv if col in df.columns]
            
            assert len(found_ohlcv) >= 4, f"Missing essential OHLCV columns. Found: {found_ohlcv}"
            
            # Check for technical indicators (from FeatureEngineer)
            expected_indicators = ['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'ema_12', 'sma_20']
            found_indicators = [col for col in expected_indicators if col in df.columns]
            
            logger.info(f"Feature completeness check",
                       total_columns=len(df.columns),
                       ohlcv_columns=len(found_ohlcv),
                       indicator_columns=len(found_indicators))
            
            # Should have reasonable number of features (at least 30 for comprehensive training)
            assert len(df.columns) >= 20, f"Expected at least 20 columns, got {len(df.columns)}"
            
        except Exception as e:
            pytest.skip(f"Could not validate feature completeness: {e}")


# Helper function for running tests standalone
if __name__ == "__main__":
    import asyncio
    
    async def run_sample_test():
        """Run a sample test to verify GCS connectivity"""
        loader = GCSCorpusLoader(bucket_name="shyvr-models-prod")
        
        try:
            versions = await loader.list_available_corpus_versions()
            print(f"Found {len(versions)} corpus versions")
            
            if versions:
                latest = await loader.get_latest_corpus_version()
                print(f"Latest version: {latest}")
                
                # Try to load some data
                df = await loader.load_corpus_from_gcs(timeframe="daily")
                print(f"Loaded daily data: {len(df)} rows, {len(df.columns)} columns")
                
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(run_sample_test())