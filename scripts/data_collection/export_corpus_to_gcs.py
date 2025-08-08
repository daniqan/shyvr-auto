#!/usr/bin/env python
"""
Export Existing Corpus to GCS
Re-exports an existing corpus version to Google Cloud Storage with all features

Usage:
    # Export the latest corpus version
    python scripts/data_collection/export_corpus_to_gcs.py
    
    # Export a specific corpus version
    python scripts/data_collection/export_corpus_to_gcs.py --version-id 4
    
    # Export to a different bucket
    python scripts/data_collection/export_corpus_to_gcs.py --bucket my-bucket
"""

import asyncio
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
from src.utils.database import get_database_connection


async def get_latest_corpus_version():
    """Get the latest corpus version ID from database"""
    async with get_database_connection() as conn:
        version_id = await conn.fetchval("""
            SELECT version_id 
            FROM training_corpus_versions 
            WHERE data_source = 'initial'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        return version_id


async def export_corpus(version_id: int = None, bucket_name: str = 'shyvr-models-prod'):
    """
    Export corpus data to GCS
    
    Args:
        version_id: Corpus version ID to export (None for latest)
        bucket_name: GCS bucket name
    """
    print('📤 Corpus Export to GCS')
    print('')
    
    try:
        # Get version ID if not specified
        if version_id is None:
            version_id = await get_latest_corpus_version()
            if version_id is None:
                print('❌ No corpus versions found in database')
                return False
            print(f'   Using latest version: {version_id}')
        else:
            print(f'   Using specified version: {version_id}')
        
        print(f'   Target bucket: gs://{bucket_name}/')
        print('')
        
        # Get corpus info
        async with get_database_connection() as conn:
            version_info = await conn.fetchrow("""
                SELECT * FROM training_corpus_versions 
                WHERE version_id = $1
            """, version_id)
            
            if not version_info:
                print(f'❌ Corpus version {version_id} not found')
                return False
            
            print(f'📊 Corpus Information:')
            print(f'   - Version Name: {version_info["version_name"]}')
            print(f'   - Created: {version_info["created_at"]}')
            print(f'   - Date Range: {version_info["start_date"]} to {version_info["end_date"]}')
            print(f'   - Tokens: {", ".join(version_info["tokens"])}')
            print('')
        
        # Initialize collector and export
        print('☁️  Starting export to GCS...')
        collector = InitialCorpusCollector()
        
        result = await collector.export_to_gcs(
            version_id=version_id,
            bucket_name=bucket_name
        )
        
        if result['success']:
            print('')
            print('✅ Export successful!')
            print('')
            print('📁 Exported files:')
            for key, path in result["gcs_paths"].items():
                if path:
                    print(f'   • {key}: {path}')
            
            print('')
            print('📊 Records exported:')
            if isinstance(result["records_exported"], dict):
                for key, count in result["records_exported"].items():
                    if key != 'total' and count > 0:
                        print(f'   • {key}: {count:,}')
                print(f'   Total: {result["records_exported"]["total"]:,} records')
            else:
                print(f'   Total: {result["records_exported"]:,} records')
            
            print('')
            print('Next steps:')
            print(f'1. Verify files: gsutil ls -l gs://{bucket_name}/training-data/initial-corpus/{result["version_name"]}/')
            print(f'2. Download locally: gsutil -m cp -r gs://{bucket_name}/training-data/initial-corpus/{result["version_name"]}/ .')
            
            return True
        else:
            print('')
            print(f'❌ Export failed: {result.get("error", "Unknown error")}')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        await collector.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Export corpus data to Google Cloud Storage'
    )
    
    parser.add_argument(
        '--version-id',
        type=int,
        help='Corpus version ID to export (default: latest)'
    )
    
    parser.add_argument(
        '--bucket',
        default='shyvr-models-prod',
        help='GCS bucket name (default: shyvr-models-prod)'
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get('DATABASE_URL'):
        print('❌ DATABASE_URL environment variable not set')
        print('Please run with the deployment script:')
        print('   ./scripts/data_collection/collect_initial_corpus.sh export')
        sys.exit(1)
    
    # Run export
    success = asyncio.run(export_corpus(
        version_id=args.version_id,
        bucket_name=args.bucket
    ))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()