#!/usr/bin/env python3
"""
GCS Corpus Exporter

Handles exporting initial corpus data to Google Cloud Storage
following the established bucket structure from the deploy/ directory.

Exports to: gs://shyvr-models-prod/training-data/initial-corpus/v1.0/
Structure:
  raw/             # Raw OHLCV data in parquet format
  processed/       # Feature-engineered data
  metadata.json    # Corpus metadata and statistics
  checksum.md5     # Data integrity verification
"""

import os
import json
import hashlib
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import structlog

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Import existing infrastructure
from src.utils.database import get_database_connection


try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, Forbidden
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


logger = structlog.get_logger(__name__)


class GCSCorpusExporter:
    """
    Export initial corpus data to GCS with proper structure and metadata
    
    Integrates with existing GCS bucket infrastructure and follows
    established patterns from the deploy/ directory.
    """
    
    def __init__(self, 
                 project_id: str,
                 bucket_name: str = "shyvr-models-prod",
                 base_path: str = "training-data/initial-corpus"):
        """
        Initialize GCS corpus exporter
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name
            base_path: base path within bucket for corpus data
        """
        if not GCS_AVAILABLE:
            raise ImportError("Google Cloud Storage library not available")
        
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.base_path = base_path
        
        # Initialize GCS client
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        
        self.logger = structlog.get_logger().bind(
            component="GCSCorpusExporter",
            bucket=bucket_name,
            project=project_id
        )
        
        # Track export statistics
        self.export_stats = {
            'start_time': None,
            'end_time': None,
            'files_exported': 0,
            'bytes_exported': 0,
            'export_path': None
        }
    
    async def export_corpus(self, 
                          version_id: int,
                          version_name: str,
                          corpus_path: Optional[str] = None,
                          export_format: str = 'parquet',
                          include_metadata: bool = True,
                          compress: bool = True) -> Dict[str, Any]:
        """
        Export corpus data to GCS
        
        Args:
            version_id: corpus version ID from database
            version_name: version name (e.g., 'initial_v1.0_20250108')
            corpus_path: custom GCS path (optional)
            export_format: export format ('parquet', 'csv', 'json')
            include_metadata: include metadata files
            compress: compress data files
            
        Returns:
            Export results dictionary
        """
        self.export_stats['start_time'] = datetime.now().isoformat()
        
        # Determine export path
        if corpus_path is None:
            corpus_path = f"{self.base_path}/{version_name}/"
        
        self.export_stats['export_path'] = f"gs://{self.bucket_name}/{corpus_path}"
        
        self.logger.info("Starting corpus export to GCS",
                        version_id=version_id,
                        version_name=version_name,
                        export_path=self.export_stats['export_path'],
                        export_format=export_format)
        
        try:
            # Verify bucket access
            await self._verify_bucket_access()
            
            # Export data tables
            export_results = {}
            
            # Export OHLCV data
            ohlcv_result = await self._export_ohlcv_data(
                version_id, corpus_path, export_format, compress
            )
            export_results['ohlcv'] = ohlcv_result
            
            # Export features data
            features_result = await self._export_features_data(
                version_id, corpus_path, export_format, compress
            )
            export_results['features'] = features_result
            
            # Export market data
            market_result = await self._export_market_data(
                version_id, corpus_path, export_format, compress
            )
            export_results['market_data'] = market_result
            
            # Export metadata
            if include_metadata:
                metadata_result = await self._export_metadata(
                    version_id, version_name, corpus_path, export_results
                )
                export_results['metadata'] = metadata_result
            
            # Generate checksums
            checksum_result = await self._generate_checksums(
                corpus_path, export_results
            )
            export_results['checksums'] = checksum_result
            
            self.export_stats['end_time'] = datetime.now().isoformat()
            
            # Update database with storage path
            await self._update_storage_path(version_id, self.export_stats['export_path'])
            
            self.logger.info("Corpus export completed successfully",
                           files_exported=self.export_stats['files_exported'],
                           bytes_exported=self.export_stats['bytes_exported'],
                           duration_seconds=(
                               datetime.fromisoformat(self.export_stats['end_time']) -
                               datetime.fromisoformat(self.export_stats['start_time'])
                           ).total_seconds())
            
            return {
                'success': True,
                'export_path': self.export_stats['export_path'],
                'export_results': export_results,
                'export_stats': self.export_stats
            }
            
        except Exception as e:
            self.export_stats['end_time'] = datetime.now().isoformat()
            self.logger.error("Corpus export failed", error=str(e))
            
            return {
                'success': False,
                'error': str(e),
                'export_path': self.export_stats.get('export_path'),
                'export_stats': self.export_stats
            }
    
    async def _verify_bucket_access(self) -> None:
        """Verify bucket exists and we have write access"""
        try:
            # Check if bucket exists
            if not self.bucket.exists():
                raise NotFound(f"Bucket {self.bucket_name} not found")
            
            # Test write access by creating a test object
            test_blob = self.bucket.blob(f"{self.base_path}/.access_test")
            test_blob.upload_from_string("access_test", content_type="text/plain")
            test_blob.delete()
            
            self.logger.info("GCS bucket access verified", bucket=self.bucket_name)
            
        except Forbidden as e:
            raise PermissionError(f"Insufficient permissions for bucket {self.bucket_name}: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to verify bucket access: {e}")
    
    async def _export_ohlcv_data(self, 
                               version_id: int,
                               corpus_path: str,
                               export_format: str,
                               compress: bool) -> Dict[str, Any]:
        """Export OHLCV data to GCS"""
        self.logger.info("Exporting OHLCV data")
        
        # Query OHLCV data for this corpus
        query = """
            SELECT 
                token_symbol, token_address, chain, timestamp,
                open, high, low, close, volume,
                exchange, market_cap, circulating_supply,
                collection_timestamp, training_status
            FROM crypto_ohlcv 
            WHERE data_source = 'initial'
            ORDER BY token_symbol, timestamp
        """
        
        async with get_database_connection() as conn:
            rows = await conn.fetch(query)
        
        if not rows:
            self.logger.warning("No OHLCV data found for initial corpus")
            return {'success': False, 'error': 'No OHLCV data found'}
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Export by format
        file_name = f"ohlcv_data.{export_format}"
        if compress and export_format == 'parquet':
            file_name = f"ohlcv_data.parquet.gz"
        
        blob_path = f"{corpus_path}raw/{file_name}"
        blob = self.bucket.blob(blob_path)
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile() as temp_file:
            if export_format == 'parquet':
                if compress:
                    df.to_parquet(temp_file.name, compression='gzip', index=False)
                else:
                    df.to_parquet(temp_file.name, index=False)
            elif export_format == 'csv':
                df.to_csv(temp_file.name, index=False)
            elif export_format == 'json':
                df.to_json(temp_file.name, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            # Upload to GCS
            temp_file.seek(0)
            blob.upload_from_filename(temp_file.name)
            
            file_size = temp_file.tell()
            self.export_stats['files_exported'] += 1
            self.export_stats['bytes_exported'] += file_size
        
        self.logger.info("OHLCV data exported",
                        records=len(df),
                        file_size_bytes=file_size,
                        blob_path=blob_path)
        
        return {
            'success': True,
            'file_path': f"gs://{self.bucket_name}/{blob_path}",
            'records_exported': len(df),
            'file_size_bytes': file_size,
            'columns': list(df.columns)
        }
    
    async def _export_features_data(self,
                                  version_id: int, 
                                  corpus_path: str,
                                  export_format: str,
                                  compress: bool) -> Dict[str, Any]:
        """Export features data to GCS"""
        self.logger.info("Exporting features data")
        
        # Query features data
        query = """
            SELECT 
                token_symbol, timestamp,
                rsi, macd, macd_signal, macd_histogram,
                bollinger_upper, bollinger_lower, bollinger_middle,
                ema_12, ema_26, ema_50, ema_200,
                sma_20, sma_50, sma_200,
                volume_sma, volume_ema, atr, adx, cci,
                stoch_k, stoch_d, williams_r, obv, vwap,
                price_change_1h, price_change_4h, price_change_24h, price_change_7d,
                volatility_1h, volatility_24h, high_low_ratio, close_open_ratio,
                collection_timestamp
            FROM crypto_features
            WHERE data_source = 'initial'
            ORDER BY token_symbol, timestamp
        """
        
        async with get_database_connection() as conn:
            rows = await conn.fetch(query)
        
        if not rows:
            self.logger.warning("No features data found for initial corpus")
            return {'success': False, 'error': 'No features data found'}
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Export processed features
        file_name = f"features_data.{export_format}"
        if compress and export_format == 'parquet':
            file_name = f"features_data.parquet.gz"
        
        blob_path = f"{corpus_path}processed/{file_name}"
        blob = self.bucket.blob(blob_path)
        
        with tempfile.NamedTemporaryFile() as temp_file:
            if export_format == 'parquet':
                if compress:
                    df.to_parquet(temp_file.name, compression='gzip', index=False)
                else:
                    df.to_parquet(temp_file.name, index=False)
            elif export_format == 'csv':
                df.to_csv(temp_file.name, index=False)
            elif export_format == 'json':
                df.to_json(temp_file.name, orient='records', date_format='iso')
            
            temp_file.seek(0)
            blob.upload_from_filename(temp_file.name)
            
            file_size = temp_file.tell()
            self.export_stats['files_exported'] += 1
            self.export_stats['bytes_exported'] += file_size
        
        self.logger.info("Features data exported",
                        records=len(df),
                        file_size_bytes=file_size,
                        blob_path=blob_path)
        
        return {
            'success': True,
            'file_path': f"gs://{self.bucket_name}/{blob_path}",
            'records_exported': len(df),
            'file_size_bytes': file_size,
            'feature_columns': [col for col in df.columns if col not in ['token_symbol', 'timestamp', 'collection_timestamp']]
        }
    
    async def _export_market_data(self,
                                version_id: int,
                                corpus_path: str, 
                                export_format: str,
                                compress: bool) -> Dict[str, Any]:
        """Export market-wide data (sentiment, DeFi, social, on-chain)"""
        self.logger.info("Exporting market data")
        
        market_data_exports = {}
        
        # Export market sentiment
        sentiment_query = """
            SELECT timestamp, fear_greed_index, sentiment_classification,
                   btc_dominance, eth_dominance, total_market_cap, total_volume_24h,
                   collection_timestamp
            FROM market_sentiment
            WHERE data_source = 'initial'
            ORDER BY timestamp
        """
        
        async with get_database_connection() as conn:
            sentiment_rows = await conn.fetch(sentiment_query)
            
            if sentiment_rows:
                df = pd.DataFrame([dict(row) for row in sentiment_rows])
                result = await self._export_dataframe_to_gcs(
                    df, f"{corpus_path}processed/market_sentiment.{export_format}", 
                    export_format, compress
                )
                market_data_exports['sentiment'] = result
        
        # Export DeFi metrics
        defi_query = """
            SELECT token_address, chain, timestamp, tvl, tvl_change_24h, tvl_change_7d,
                   protocol_count, avg_apy, max_apy, staking_ratio, collection_timestamp
            FROM defi_metrics
            WHERE data_source = 'initial'
            ORDER BY timestamp
        """
        
        async with get_database_connection() as conn:
            defi_rows = await conn.fetch(defi_query)
            
            if defi_rows:
                df = pd.DataFrame([dict(row) for row in defi_rows])
                result = await self._export_dataframe_to_gcs(
                    df, f"{corpus_path}processed/defi_metrics.{export_format}",
                    export_format, compress
                )
                market_data_exports['defi'] = result
        
        # Export social sentiment (if available)
        social_query = """
            SELECT token_symbol, timestamp, social_volume, social_engagement,
                   social_contributors, social_dominance, sentiment_score,
                   sentiment_absolute, galaxy_score, alt_rank, collection_timestamp
            FROM social_sentiment
            WHERE data_source = 'initial'
            ORDER BY token_symbol, timestamp
        """
        
        async with get_database_connection() as conn:
            social_rows = await conn.fetch(social_query)
            
            if social_rows:
                df = pd.DataFrame([dict(row) for row in social_rows])
                result = await self._export_dataframe_to_gcs(
                    df, f"{corpus_path}processed/social_sentiment.{export_format}",
                    export_format, compress  
                )
                market_data_exports['social'] = result
        
        # Export on-chain metrics (if available)
        onchain_query = """
            SELECT token_address, chain, timestamp, transaction_count,
                   unique_addresses, active_addresses_24h, whale_activity,
                   whale_transactions, exchange_inflow, exchange_outflow, collection_timestamp
            FROM onchain_metrics
            WHERE data_source = 'initial'
            ORDER BY chain, timestamp
        """
        
        async with get_database_connection() as conn:
            onchain_rows = await conn.fetch(onchain_query)
            
            if onchain_rows:
                df = pd.DataFrame([dict(row) for row in onchain_rows])
                result = await self._export_dataframe_to_gcs(
                    df, f"{corpus_path}processed/onchain_metrics.{export_format}",
                    export_format, compress
                )
                market_data_exports['onchain'] = result
        
        return {
            'success': True,
            'exports': market_data_exports,
            'total_datasets': len(market_data_exports)
        }
    
    async def _export_dataframe_to_gcs(self,
                                     df: pd.DataFrame,
                                     blob_path: str,
                                     export_format: str,
                                     compress: bool) -> Dict[str, Any]:
        """Helper method to export DataFrame to GCS"""
        blob = self.bucket.blob(blob_path)
        
        with tempfile.NamedTemporaryFile() as temp_file:
            if export_format == 'parquet':
                if compress:
                    df.to_parquet(temp_file.name, compression='gzip', index=False)
                else:
                    df.to_parquet(temp_file.name, index=False)
            elif export_format == 'csv':
                df.to_csv(temp_file.name, index=False)
            elif export_format == 'json':
                df.to_json(temp_file.name, orient='records', date_format='iso')
            
            temp_file.seek(0)
            blob.upload_from_filename(temp_file.name)
            
            file_size = temp_file.tell()
            self.export_stats['files_exported'] += 1
            self.export_stats['bytes_exported'] += file_size
        
        return {
            'success': True,
            'file_path': f"gs://{self.bucket_name}/{blob_path}",
            'records_exported': len(df),
            'file_size_bytes': file_size
        }
    
    async def _export_metadata(self,
                             version_id: int,
                             version_name: str,
                             corpus_path: str,
                             export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Export corpus metadata"""
        self.logger.info("Exporting metadata")
        
        # Get corpus version information
        query = """
            SELECT version_name, created_at, data_source, sample_count, feature_count,
                   tokens, start_timestamp, end_timestamp, is_active, is_immutable,
                   statistics, created_by, notes
            FROM training_corpus_versions
            WHERE version_id = $1
        """
        
        async with get_database_connection() as conn:
            row = await conn.fetchrow(query, version_id)
        
        if not row:
            return {'success': False, 'error': f'Version {version_id} not found'}
        
        # Create comprehensive metadata
        metadata = {
            'corpus_info': {
                'version_id': version_id,
                'version_name': version_name,
                'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                'data_source': row['data_source'],
                'sample_count': row['sample_count'],
                'feature_count': row['feature_count'],
                'tokens': row['tokens'],
                'start_timestamp': row['start_timestamp'].isoformat() if row['start_timestamp'] else None,
                'end_timestamp': row['end_timestamp'].isoformat() if row['end_timestamp'] else None,
                'is_active': row['is_active'],
                'is_immutable': row['is_immutable'],
                'created_by': row['created_by'],
                'notes': row['notes']
            },
            'export_info': {
                'export_timestamp': datetime.now().isoformat(),
                'export_path': f"gs://{self.bucket_name}/{corpus_path}",
                'exporter': 'GCSCorpusExporter',
                'export_stats': self.export_stats
            },
            'data_structure': {
                'directories': {
                    'raw/': 'Raw OHLCV data in original format',
                    'processed/': 'Feature-engineered and processed data'
                },
                'files': {}
            },
            'schema': {
                'ohlcv_columns': [
                    'token_symbol', 'token_address', 'chain', 'timestamp',
                    'open', 'high', 'low', 'close', 'volume',
                    'exchange', 'market_cap', 'circulating_supply'
                ],
                'feature_categories': {
                    'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'moving_averages'],
                    'price_action': ['price_changes', 'volatility', 'ratios'],
                    'volume_indicators': ['volume_sma', 'volume_ema', 'obv'],
                    'momentum': ['adx', 'cci', 'stochastic', 'williams_r']
                }
            }
        }
        
        # Add file information from export results
        for data_type, result in export_results.items():
            if isinstance(result, dict) and result.get('success'):
                if 'file_path' in result:
                    metadata['data_structure']['files'][data_type] = {
                        'path': result['file_path'],
                        'records': result.get('records_exported', 0),
                        'size_bytes': result.get('file_size_bytes', 0)
                    }
                elif 'exports' in result:  # Market data with multiple files
                    for sub_type, sub_result in result['exports'].items():
                        metadata['data_structure']['files'][f"{data_type}_{sub_type}"] = {
                            'path': sub_result['file_path'],
                            'records': sub_result.get('records_exported', 0),
                            'size_bytes': sub_result.get('file_size_bytes', 0)
                        }
        
        # Export metadata as JSON
        blob_path = f"{corpus_path}metadata.json"
        blob = self.bucket.blob(blob_path)
        
        metadata_json = json.dumps(metadata, indent=2, default=str)
        blob.upload_from_string(metadata_json, content_type='application/json')
        
        self.export_stats['files_exported'] += 1
        self.export_stats['bytes_exported'] += len(metadata_json.encode('utf-8'))
        
        self.logger.info("Metadata exported", blob_path=blob_path)
        
        return {
            'success': True,
            'file_path': f"gs://{self.bucket_name}/{blob_path}",
            'metadata': metadata
        }
    
    async def _generate_checksums(self,
                                corpus_path: str,
                                export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate checksums for data integrity verification"""
        self.logger.info("Generating checksums")
        
        checksums = {
            'generated_at': datetime.now().isoformat(),
            'algorithm': 'md5',
            'files': {}
        }
        
        # Generate checksums for each exported file
        for data_type, result in export_results.items():
            if isinstance(result, dict) and result.get('success'):
                if 'file_path' in result:
                    file_path = result['file_path']
                    blob_path = file_path.replace(f"gs://{self.bucket_name}/", "")
                    blob = self.bucket.blob(blob_path)
                    
                    # Download and compute checksum
                    with tempfile.NamedTemporaryFile() as temp_file:
                        blob.download_to_filename(temp_file.name)
                        
                        with open(temp_file.name, 'rb') as f:
                            checksum = hashlib.md5(f.read()).hexdigest()
                        
                        checksums['files'][blob_path] = checksum
                
                elif 'exports' in result:  # Market data with multiple files
                    for sub_type, sub_result in result['exports'].items():
                        file_path = sub_result['file_path']
                        blob_path = file_path.replace(f"gs://{self.bucket_name}/", "")
                        blob = self.bucket.blob(blob_path)
                        
                        with tempfile.NamedTemporaryFile() as temp_file:
                            blob.download_to_filename(temp_file.name)
                            
                            with open(temp_file.name, 'rb') as f:
                                checksum = hashlib.md5(f.read()).hexdigest()
                            
                            checksums['files'][blob_path] = checksum
        
        # Export checksums file
        blob_path = f"{corpus_path}checksums.json"
        blob = self.bucket.blob(blob_path)
        
        checksums_json = json.dumps(checksums, indent=2)
        blob.upload_from_string(checksums_json, content_type='application/json')
        
        self.export_stats['files_exported'] += 1
        self.export_stats['bytes_exported'] += len(checksums_json.encode('utf-8'))
        
        self.logger.info("Checksums generated", 
                        file_count=len(checksums['files']),
                        checksum_file=blob_path)
        
        return {
            'success': True,
            'file_path': f"gs://{self.bucket_name}/{blob_path}",
            'checksums': checksums
        }
    
    async def _update_storage_path(self, version_id: int, storage_path: str) -> None:
        """Update database with storage path"""
        query = """
            UPDATE training_corpus_versions 
            SET storage_path = $1, updated_at = NOW()
            WHERE version_id = $2
        """
        
        async with get_database_connection() as conn:
            await conn.execute(query, storage_path, version_id)
        
        self.logger.info("Storage path updated in database",
                        version_id=version_id,
                        storage_path=storage_path)
    
    async def verify_export(self, export_path: str) -> Dict[str, Any]:
        """Verify exported corpus data integrity"""
        self.logger.info("Verifying export integrity", export_path=export_path)
        
        # Extract path components
        path_parts = export_path.replace(f"gs://{self.bucket_name}/", "")
        
        verification_results = {
            'verified_at': datetime.now().isoformat(),
            'export_path': export_path,
            'checks': {}
        }
        
        try:
            # Check metadata file exists
            metadata_blob = self.bucket.blob(f"{path_parts}metadata.json")
            if metadata_blob.exists():
                verification_results['checks']['metadata'] = {'exists': True}
                # Could validate metadata structure here
            else:
                verification_results['checks']['metadata'] = {'exists': False}
            
            # Check checksums file exists
            checksums_blob = self.bucket.blob(f"{path_parts}checksums.json") 
            if checksums_blob.exists():
                verification_results['checks']['checksums'] = {'exists': True}
                # Could verify checksums here
            else:
                verification_results['checks']['checksums'] = {'exists': False}
            
            # Check main data files exist
            required_files = ['raw/ohlcv_data.parquet', 'processed/features_data.parquet']
            for file_path in required_files:
                blob = self.bucket.blob(f"{path_parts}{file_path}")
                verification_results['checks'][file_path] = {'exists': blob.exists()}
            
            # Overall success
            all_checks_passed = all(
                check.get('exists', False) 
                for check in verification_results['checks'].values()
            )
            verification_results['success'] = all_checks_passed
            
            self.logger.info("Export verification completed",
                           success=all_checks_passed,
                           checks_passed=sum(1 for c in verification_results['checks'].values() if c.get('exists')),
                           total_checks=len(verification_results['checks']))
            
            return verification_results
            
        except Exception as e:
            verification_results['success'] = False
            verification_results['error'] = str(e)
            self.logger.error("Export verification failed", error=str(e))
            return verification_results