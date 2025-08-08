#!/usr/bin/env python3
"""
Direct Initial Corpus Collection Script

A streamlined script that executes initial corpus collection directly
without container overhead. Uses existing infrastructure and is optimized
for production execution with batching and progress tracking.

Key Features:
- Loads secrets directly from GCP Secret Manager
- Uses existing InitialCorpusCollector
- Batched execution to avoid timeouts
- Progress tracking and resumability
- Real API calls to production endpoints
- Stores in CloudSQL with data_source='initial'
- Exports to GCS bucket gs://shyvr-models-prod/training-data/initial-corpus/v1.0/
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing infrastructure
from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
from src.utils.database import get_database_connection
import structlog

# GCP imports
try:
    from google.cloud import secretmanager
    GCP_AVAILABLE = True
except ImportError:
    print("WARNING: google-cloud-secret-manager not available. Set API keys manually.")
    GCP_AVAILABLE = False


class DirectCorpusCollector:
    """Direct executor for initial corpus collection"""
    
    def __init__(self, project_id: str = "shvyr-ai-bots", batch_size: int = 3):
        self.project_id = project_id
        self.batch_size = batch_size  # Process tokens in batches
        self.logger = structlog.get_logger().bind(component="DirectCorpusCollector")
        
        # Default tokens (can be overridden)
        self.tokens = [
            'bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano',
            'matic-network', 'avalanche-2', 'polkadot', 'chainlink', 'uniswap'
        ]
        
        # Collection parameters
        self.collection_days = 180  # 6 months
        self.rate_limit_delay = 2.1  # Conservative rate limiting
        
    def load_secrets_from_gcp(self) -> Dict[str, str]:
        """Load API keys from GCP Secret Manager"""
        secrets = {}
        
        if not GCP_AVAILABLE:
            self.logger.warning("GCP libraries not available, checking environment variables")
            return {
                'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY'),
                'LUNARCRUSH_API_KEY': os.getenv('LUNARCRUSH_API_KEY'), 
                'HELIUS_API_KEY': os.getenv('HELIUS_API_KEY'),
                'DATABASE_URL': os.getenv('DATABASE_URL')
            }
        
        try:
            client = secretmanager.SecretManagerServiceClient()
            
            secret_names = [
                'COINGECKO_API_KEY',
                'LUNARCRUSH_API_KEY', 
                'HELIUS_API_KEY',
                'DATABASE_URL'
            ]
            
            for secret_name in secret_names:
                try:
                    name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
                    response = client.access_secret_version(request={"name": name})
                    secret_value = response.payload.data.decode("UTF-8")
                    secrets[secret_name] = secret_value
                    self.logger.info(f"Loaded secret: {secret_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load secret {secret_name}: {e}")
                    # Fallback to environment variable
                    secrets[secret_name] = os.getenv(secret_name)
            
            return secrets
            
        except Exception as e:
            self.logger.error(f"Failed to access Secret Manager: {e}")
            self.logger.info("Falling back to environment variables")
            return {
                'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY'),
                'LUNARCRUSH_API_KEY': os.getenv('LUNARCRUSH_API_KEY'),
                'HELIUS_API_KEY': os.getenv('HELIUS_API_KEY'),
                'DATABASE_URL': os.getenv('DATABASE_URL')
            }
    
    def set_environment_variables(self, secrets: Dict[str, str]) -> None:
        """Set environment variables from loaded secrets"""
        for key, value in secrets.items():
            if value:
                os.environ[key] = value
                self.logger.info(f"Set environment variable: {key}")
            else:
                self.logger.warning(f"No value for secret: {key}")
    
    async def check_existing_data(self) -> Dict[str, Any]:
        """Check if initial corpus data already exists"""
        try:
            async with get_database_connection() as conn:
                # Check for existing initial corpus data
                existing_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM crypto_ohlcv 
                    WHERE data_source = 'initial'
                """)
                
                # Check for existing corpus version
                existing_version = await conn.fetchrow("""
                    SELECT version_id, version_name, created_at, sample_count
                    FROM training_corpus_versions 
                    WHERE data_source = 'initial' AND is_active = true
                    ORDER BY created_at DESC LIMIT 1
                """)
                
                return {
                    'existing_records': existing_count or 0,
                    'existing_version': dict(existing_version) if existing_version else None,
                    'has_initial_data': (existing_count or 0) > 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to check existing data: {e}")
            return {
                'existing_records': 0,
                'existing_version': None,
                'has_initial_data': False
            }
    
    async def collect_batch(self, token_batch: List[str]) -> Dict[str, Any]:
        """Collect data for a batch of tokens"""
        self.logger.info(f"Starting collection for batch: {token_batch}")
        
        try:
            # Create collector instance for this batch
            collector = InitialCorpusCollector(
                collection_days=self.collection_days,
                rate_limit_delay=self.rate_limit_delay,
                retry_attempts=3,
                batch_size=100
            )
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.collection_days)
            
            # Collect data for this batch
            result = await collector.collect_standardized_corpus(
                tokens=token_batch,
                start_date=start_date,
                end_date=end_date
            )
            
            await collector.close()
            
            self.logger.info(f"Batch collection completed", 
                           batch=token_batch,
                           success=result.get('success', False),
                           records=result.get('total_records', 0))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch collection failed", batch=token_batch, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'batch': token_batch
            }
    
    async def execute_collection(self, resume_from_batch: int = 0) -> Dict[str, Any]:
        """Execute the full collection process in batches"""
        self.logger.info("Starting direct corpus collection execution")
        
        # Check existing data
        existing_data = await self.check_existing_data()
        
        if existing_data['has_initial_data'] and resume_from_batch == 0:
            self.logger.warning(f"Initial corpus data already exists: {existing_data['existing_records']} records")
            user_input = input("Continue anyway? This will add more data. (y/N): ").lower()
            if user_input != 'y':
                return {
                    'success': False,
                    'message': 'Collection cancelled by user',
                    'existing_data': existing_data
                }
        
        # Split tokens into batches
        token_batches = []
        for i in range(0, len(self.tokens), self.batch_size):
            batch = self.tokens[i:i + self.batch_size]
            token_batches.append(batch)
        
        self.logger.info(f"Created {len(token_batches)} batches of tokens")
        
        # Execute batches
        results = []
        successful_batches = 0
        total_records = 0
        
        for batch_idx, token_batch in enumerate(token_batches):
            if batch_idx < resume_from_batch:
                self.logger.info(f"Skipping batch {batch_idx} (resuming from {resume_from_batch})")
                continue
                
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(token_batches)}: {token_batch}")
            
            try:
                batch_result = await self.collect_batch(token_batch)
                results.append(batch_result)
                
                if batch_result.get('success', False):
                    successful_batches += 1
                    total_records += batch_result.get('total_records', 0)
                    self.logger.info(f"Batch {batch_idx + 1} completed successfully")
                else:
                    self.logger.error(f"Batch {batch_idx + 1} failed: {batch_result.get('error')}")
                
                # Progress update
                progress = ((batch_idx + 1) / len(token_batches)) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({batch_idx + 1}/{len(token_batches)} batches)")
                
                # Brief pause between batches to avoid overwhelming APIs
                if batch_idx < len(token_batches) - 1:
                    await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                self.logger.info("Collection interrupted by user")
                return {
                    'success': False,
                    'message': f'Collection interrupted after batch {batch_idx}',
                    'completed_batches': batch_idx,
                    'resume_from': batch_idx + 1,
                    'results': results
                }
            except Exception as e:
                self.logger.error(f"Unexpected error in batch {batch_idx + 1}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'batch': token_batch
                })
        
        # Summary
        summary = {
            'success': successful_batches > 0,
            'total_batches': len(token_batches),
            'successful_batches': successful_batches,
            'failed_batches': len(token_batches) - successful_batches,
            'total_records': total_records,
            'results': results
        }
        
        self.logger.info("Collection execution completed", 
                        total_batches=len(token_batches),
                        successful=successful_batches,
                        total_records=total_records)
        
        return summary
    
    async def export_to_gcs(self) -> Dict[str, Any]:
        """Export collected data to GCS"""
        try:
            # Get the latest corpus version
            async with get_database_connection() as conn:
                version_row = await conn.fetchrow("""
                    SELECT version_id, version_name 
                    FROM training_corpus_versions 
                    WHERE data_source = 'initial' AND is_active = true
                    ORDER BY created_at DESC LIMIT 1
                """)
            
            if not version_row:
                return {'success': False, 'error': 'No active initial corpus version found'}
            
            version_id = version_row['version_id']
            version_name = version_row['version_name']
            
            # Create collector to handle export
            collector = InitialCorpusCollector(
                collection_days=self.collection_days,
                rate_limit_delay=self.rate_limit_delay
            )
            
            # Export to GCS
            gcs_path = "gs://shyvr-models-prod/training-data/initial-corpus/v1.0/"
            export_result = await collector.create_corpus_snapshot(
                version_id=version_id,
                snapshot_path=gcs_path,
                export_format='parquet',
                project_id=self.project_id
            )
            
            await collector.close()
            
            return export_result
            
        except Exception as e:
            self.logger.error(f"GCS export failed: {e}")
            return {'success': False, 'error': str(e)}


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Direct Initial Corpus Collection')
    parser.add_argument('--project-id', default='shvyr-ai-bots', help='GCP Project ID')
    parser.add_argument('--batch-size', type=int, default=3, help='Tokens per batch')
    parser.add_argument('--resume-from', type=int, default=0, help='Resume from batch number')
    parser.add_argument('--skip-export', action='store_true', help='Skip GCS export')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (check only)')
    
    args = parser.parse_args()
    
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create collector
    collector = DirectCorpusCollector(
        project_id=args.project_id,
        batch_size=args.batch_size
    )
    
    try:
        # Load secrets from GCP
        print("Loading secrets from GCP Secret Manager...")
        secrets = collector.load_secrets_from_gcp()
        collector.set_environment_variables(secrets)
        
        # Verify database connection
        print("Testing database connection...")
        try:
            async with get_database_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    print("✓ Database connection successful")
                else:
                    print("✗ Database connection test failed")
                    return
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return
        
        # Check existing data
        print("Checking existing data...")
        existing_data = await collector.check_existing_data()
        print(f"Existing initial corpus records: {existing_data['existing_records']}")
        
        if existing_data['existing_version']:
            print(f"Active version: {existing_data['existing_version']['version_name']}")
        
        if args.dry_run:
            print("Dry run complete - no collection performed")
            return
        
        # Execute collection
        print(f"Starting collection with {len(collector.tokens)} tokens in batches of {args.batch_size}")
        print(f"Collection period: {collector.collection_days} days")
        
        if args.resume_from > 0:
            print(f"Resuming from batch {args.resume_from}")
        
        collection_result = await collector.execute_collection(args.resume_from)
        
        # Print results
        print("\n" + "="*50)
        print("COLLECTION RESULTS")
        print("="*50)
        print(f"Success: {collection_result['success']}")
        print(f"Total batches: {collection_result.get('total_batches', 0)}")
        print(f"Successful batches: {collection_result.get('successful_batches', 0)}")
        print(f"Failed batches: {collection_result.get('failed_batches', 0)}")
        print(f"Total records: {collection_result.get('total_records', 0)}")
        
        if not collection_result['success']:
            print(f"Collection failed or incomplete: {collection_result.get('message', 'Unknown error')}")
            if 'resume_from' in collection_result:
                print(f"To resume, use: --resume-from {collection_result['resume_from']}")
            return
        
        # Export to GCS
        if not args.skip_export:
            print("\nExporting to GCS...")
            export_result = await collector.export_to_gcs()
            
            if export_result['success']:
                print("✓ GCS export completed successfully")
                if 'export_result' in export_result:
                    stats = export_result['export_result'].get('export_stats', {})
                    print(f"Files exported: {stats.get('files_exported', 'N/A')}")
            else:
                print(f"✗ GCS export failed: {export_result.get('error', 'Unknown error')}")
        
        print("\n" + "="*50)
        print("COLLECTION COMPLETE")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Collection failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())