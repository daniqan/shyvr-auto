"""
GCS Corpus Loader for Training Data

This module provides functionality to load training corpus data from Google Cloud Storage.
Built following TDD methodology with real GCS integration (no mocks).

Key features:
- Load corpus parquet files from gs://shyvr-models-prod/training-data/initial-corpus/
- Local caching in /tmp/corpus_cache/ with TTL support
- Token filtering from combined parquet files
- Automatic version discovery and latest version selection
- Application Default Credentials authentication
- Retry logic for network failures
"""

import os
import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError

# Use standard logging to avoid import chain issues during TDD
logger = logging.getLogger(__name__)


class GCSCorpusLoader:
    """
    Loads training corpus data from Google Cloud Storage
    
    Provides functionality to:
    1. List available corpus versions in GCS
    2. Download and cache parquet files locally
    3. Load corpus data with optional token filtering
    4. Manage cache with TTL-based cleanup
    """
    
    def __init__(self, 
                 bucket_name: str = "shyvr-models-prod",
                 cache_dir: str = "/tmp/corpus_cache",
                 cache_ttl_hours: int = 24):
        """
        Initialize GCS Corpus Loader
        
        Args:
            bucket_name: GCS bucket containing corpus data
            cache_dir: Local directory for caching downloaded files  
            cache_ttl_hours: Hours before cached files are considered stale
        """
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client with Application Default Credentials
        self.gcs_client = storage.Client()
        self.bucket = self.gcs_client.bucket(bucket_name)
        
        logger.info(
            f"GCS Corpus Loader initialized - bucket: {bucket_name}, "
            f"cache_dir: {cache_dir}, cache_ttl_hours: {cache_ttl_hours}"
        )
    
    async def list_available_corpus_versions(self) -> List[str]:
        """
        List available corpus versions in GCS bucket
        
        Returns:
            List of corpus version paths, sorted by creation date (newest first)
        """
        try:
            # Search for corpus versions in the training-data/initial-corpus/ prefix
            prefix = "training-data/initial-corpus/"
            
            # List all blobs and extract unique version directories
            blobs = self.bucket.list_blobs(prefix=prefix)
            version_paths = set()
            
            for blob in blobs:
                # Extract the version directory from the full path
                # e.g., "training-data/initial-corpus/initial_v2.0_20250821_153449/corpus/corpus_daily.parquet"
                # should extract "training-data/initial-corpus/initial_v2.0_20250821_153449"
                blob_path = blob.name
                if blob_path.count('/') >= 2:  # Has at least training-data/initial-corpus/version/...
                    parts = blob_path.split('/')
                    if len(parts) >= 3:
                        version_path = '/'.join(parts[:3])  # training-data/initial-corpus/version_name
                        version_paths.add(version_path)
            
            # Convert to list and sort by name (newest first)
            version_paths = list(version_paths)
            version_paths.sort(reverse=True)
            
            logger.info(f"Found {len(version_paths)} corpus versions in GCS")
            return version_paths
            
        except Exception as e:
            logger.error(f"Failed to list corpus versions: {e}")
            raise
    
    async def get_latest_corpus_version(self) -> str:
        """
        Get the path to the latest corpus version
        
        Returns:
            GCS path to the most recent corpus version
        """
        versions = await self.list_available_corpus_versions()
        if not versions:
            raise FileNotFoundError("No corpus versions found in GCS bucket")
        
        latest_version = versions[0]  # First item is newest due to sorting
        logger.info(f"Latest corpus version: {latest_version}")
        return latest_version
    
    async def download_and_cache_parquet(self, 
                                       gcs_path: str,
                                       local_path: Optional[Path] = None) -> Path:
        """
        Download parquet file from GCS and cache locally
        
        Args:
            gcs_path: GCS path to parquet file (without gs:// prefix)
            local_path: Optional local path override
            
        Returns:
            Path to local cached file
        """
        try:
            # Generate local cache path if not provided
            if local_path is None:
                # Create a safe filename from the GCS path
                safe_filename = gcs_path.replace('/', '_').replace('\\', '_')
                local_path = Path(self.cache_dir) / safe_filename
            
            # Check if file exists and is within TTL
            if local_path.exists():
                file_age = datetime.now() - datetime.fromtimestamp(local_path.stat().st_mtime)
                ttl_delta = timedelta(hours=self.cache_ttl_hours)
                
                if file_age < ttl_delta:
                    logger.info(f"Using cached file: {local_path}")
                    return local_path
                else:
                    logger.info(f"Cached file expired, re-downloading: {local_path}")
            
            # Download from GCS
            logger.info(f"Downloading from GCS: gs://{self.bucket_name}/{gcs_path}")
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"File not found in GCS: gs://{self.bucket_name}/{gcs_path}")
            
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with retry logic
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    blob.download_to_filename(str(local_path))
                    logger.info(f"Successfully downloaded: {local_path}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Download attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise e
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download and cache parquet file: {e}")
            raise
    
    async def load_corpus_from_gcs(self,
                                 gcs_prefix: Optional[str] = None,
                                 timeframe: str = "daily",
                                 token: Optional[str] = None) -> pd.DataFrame:
        """
        Load corpus data from GCS with optional filtering
        
        Args:
            gcs_prefix: GCS prefix for corpus version (auto-detect latest if None)
            timeframe: Timeframe to load (daily, hourly, hour)
            token: Specific token to filter (load all if None)
            
        Returns:
            DataFrame with corpus data
        """
        try:
            # Get corpus version path
            if gcs_prefix is None:
                gcs_prefix = await self.get_latest_corpus_version()
            
            # Construct path to parquet file
            parquet_filename = f"corpus_{timeframe}.parquet"
            gcs_path = f"{gcs_prefix}/corpus/{parquet_filename}"
            
            # Download and cache the file
            local_path = await self.download_and_cache_parquet(gcs_path)
            
            # Load parquet file
            logger.info(f"Loading parquet file: {local_path}")
            df = pd.read_parquet(local_path)
            
            logger.info(f"Loaded corpus data - rows: {len(df)}, "
                       f"columns: {len(df.columns)}, timeframe: {timeframe}")
            
            # Filter by token if specified
            if token is not None:
                # Try different column names that might contain token symbol
                token_columns = ['symbol', 'token_symbol', 'token', 'asset']
                filter_column = None
                
                for col in token_columns:
                    if col in df.columns:
                        filter_column = col
                        break
                
                if filter_column is not None:
                    # Filter case-insensitive
                    mask = df[filter_column].str.upper() == token.upper()
                    df = df[mask].copy()
                    
                    logger.info(f"Filtered data for token {token} - "
                               f"rows: {len(df)}, filter_column: {filter_column}")
                else:
                    logger.warning(f"No token column found for filtering. Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load corpus from GCS: {e}")
            raise
    
    async def clear_old_cache(self, ttl_hours: Optional[int] = None) -> int:
        """
        Clear old cached files based on TTL
        
        Args:
            ttl_hours: Custom TTL in hours (uses instance default if None)
            
        Returns:
            Number of files removed
        """
        if ttl_hours is None:
            ttl_hours = self.cache_ttl_hours
        
        ttl_delta = timedelta(hours=ttl_hours)
        current_time = datetime.now()
        removed_count = 0
        
        try:
            cache_path = Path(self.cache_dir)
            if not cache_path.exists():
                return 0
            
            # Find and remove old files
            for file_path in cache_path.rglob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_age > ttl_delta:
                        try:
                            file_path.unlink()  # Remove file
                            removed_count += 1
                            logger.debug(f"Removed old cached file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove cached file {file_path}: {e}")
            
            # Clean up empty directories
            for dir_path in cache_path.rglob('*'):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        logger.debug(f"Removed empty cache directory: {dir_path}")
                    except Exception as e:
                        logger.debug(f"Could not remove empty directory {dir_path}: {e}")
            
            logger.info(f"Cache cleanup completed", 
                       removed_files=removed_count, 
                       ttl_hours=ttl_hours)
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to clear old cache: {e}")
            return removed_count