#!/usr/bin/env python
"""
Initial Corpus Collection Script
Collects historical training data for transformer models

This script performs the complete initial corpus collection:
- Collects 6 months (180 days) of historical OHLCV data
- Fetches market sentiment, DeFi metrics, and on-chain data
- Calculates technical indicators and features
- Stores in production database with data_source='initial'
- Creates versioned corpus for reproducible training

Usage:
    # Test run (2 tokens, 7 days)
    python scripts/data_collection/run_initial_corpus_collection.py --test
    
    # Full production run (10 tokens, 180 days)
    python scripts/data_collection/run_initial_corpus_collection.py --production
    
    # Custom configuration
    python scripts/data_collection/run_initial_corpus_collection.py \
        --tokens bitcoin ethereum solana \
        --days 30
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector


# Default token lists
TEST_TOKENS = ['bitcoin', 'ethereum']
PRODUCTION_TOKENS = [
    'bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano',
    'matic-network', 'avalanche-2', 'polkadot', 'chainlink', 'uniswap'
]


async def run_collection(
    tokens: List[str],
    days: int,
    rate_limit_delay: float = 2.1,
    retry_attempts: int = 3
) -> bool:
    """
    Run initial corpus collection
    
    Args:
        tokens: List of token IDs to collect
        days: Number of days of historical data
        rate_limit_delay: Delay between API calls in seconds
        retry_attempts: Number of retry attempts for failed requests
        
    Returns:
        True if collection successful, False otherwise
    """
    print('📡 Starting Initial Corpus Collection')
    print(f'   Database: {os.environ.get("DB_NAME", "Not set")}')
    print(f'   Environment: {os.environ.get("ENVIRONMENT", "Not set")}')
    print('')
    
    print(f'📊 Configuration:')
    print(f'   - Tokens: {len(tokens)} tokens')
    for token in tokens[:3]:  # Show first 3 tokens
        print(f'     • {token}')
    if len(tokens) > 3:
        print(f'     • ... and {len(tokens) - 3} more')
    print(f'   - Period: {days} days')
    print(f'   - Expected records: ~{len(tokens) * days * 4:,} (4-hour candles)')
    print('')
    
    try:
        # Initialize collector
        collector = InitialCorpusCollector(
            collection_days=days,
            rate_limit_delay=rate_limit_delay,
            retry_attempts=retry_attempts
        )
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        print(f'🔄 Collecting data from {start_date.date()} to {end_date.date()}...')
        print('   This may take 30-60 minutes for large datasets due to rate limiting')
        print('')
        
        # Run collection
        result = await collector.collect_standardized_corpus(
            tokens=tokens,
            start_date=start_date,
            end_date=end_date
        )
        
        if result['success']:
            print('')
            print('✅ COLLECTION SUCCESSFUL!')
            print(f'   - Version: {result.get("version_name", "N/A")}')
            print(f'   - Version ID: {result.get("corpus_version_id", "N/A")}')
            print(f'   - Total Records: {result.get("total_records", 0):,}')
            print(f'   - Tokens Collected: {result.get("tokens_collected", 0)}/{len(tokens)}')
            print(f'   - Features Calculated: {result.get("feature_count", 0)}')
            print(f'   - Duration: {result.get("collection_duration_seconds", 0):.1f} seconds')
            
            if result.get('failed_tokens'):
                print('')
                print('⚠️  Failed tokens:')
                for token, error in result['failed_tokens'].items():
                    print(f'   • {token}: {error}')
            
            print('')
            print('📊 Data successfully stored in production database!')
            
            # Export to GCS
            print('')
            print('☁️  Exporting to Google Cloud Storage...')
            export_result = await collector.export_to_gcs(
                version_id=result['corpus_version_id'],
                bucket_name='shyvr-models-prod'
            )
            
            if export_result['success']:
                print('✅ Export to GCS successful!')
                print('   Exported files:')
                for key, path in export_result["gcs_paths"].items():
                    if path:
                        print(f'     • {key}: {path}')
                print('')
                print('   Records exported:')
                if isinstance(export_result["records_exported"], dict):
                    for key, count in export_result["records_exported"].items():
                        if key != 'total' and count > 0:
                            print(f'     • {key}: {count:,}')
                    print(f'   Total: {export_result["records_exported"]["total"]:,} records')
                else:
                    print(f'   Total: {export_result["records_exported"]:,} records')
            else:
                print(f'⚠️  GCS export failed: {export_result.get("error", "Unknown error")}')
                print('   Data remains in database and can be exported later')
            
            print('')
            print('Next steps:')
            print('1. Verify data in database:')
            print('   SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source=\'initial\';')
            print('2. Train transformer models:')
            print('   python scripts/training/train_transformers.py --corpus-version ' + 
                  str(result.get("corpus_version_id", "1")))
            print('3. Start continuous learning:')
            print('   python scripts/services/start_continuous_learning.py')
            
            return True
        else:
            print('')
            print(f'❌ Collection failed: {result.get("error", "Unknown error")}')
            if result.get('failed_tokens'):
                print(f'   Failed tokens: {list(result["failed_tokens"].keys())}')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False


async def clean_existing_corpus():
    """Clean existing initial corpus data from database"""
    from src.utils.database import get_database_connection
    
    print('🧹 Cleaning existing initial corpus data...')
    print('')
    
    async with get_database_connection() as conn:
        # Check existing data
        counts = {}
        counts['ohlcv'] = await conn.fetchval("SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source='initial'")
        counts['sentiment'] = await conn.fetchval("SELECT COUNT(*) FROM market_sentiment WHERE data_source='initial'")
        counts['defi'] = await conn.fetchval("SELECT COUNT(*) FROM defi_metrics WHERE data_source='initial'")
        counts['social'] = await conn.fetchval("SELECT COUNT(*) FROM social_sentiment WHERE data_source='initial'")
        counts['onchain'] = await conn.fetchval("SELECT COUNT(*) FROM onchain_metrics WHERE data_source='initial'")
        counts['versions'] = await conn.fetchval("SELECT COUNT(*) FROM training_corpus_versions WHERE data_source='initial'")
        
        total = sum(counts.values())
        
        if total == 0:
            print('   No existing data found - database is clean')
            return True
        
        print(f'   Found existing records:')
        for table, count in counts.items():
            if count > 0:
                print(f'     • {table}: {count:,} records')
        print(f'   Total: {total:,} records')
        print('')
        
        # Confirm deletion
        response = input('Delete all existing initial corpus data? (y/N): ')
        if response.lower() != 'y':
            print('   Cleanup cancelled')
            return False
        
        # Delete all initial data
        print('   Deleting data...')
        await conn.execute("DELETE FROM crypto_ohlcv WHERE data_source='initial'")
        await conn.execute("DELETE FROM market_sentiment WHERE data_source='initial'")
        await conn.execute("DELETE FROM defi_metrics WHERE data_source='initial'")
        await conn.execute("DELETE FROM social_sentiment WHERE data_source='initial'")
        await conn.execute("DELETE FROM onchain_metrics WHERE data_source='initial'")
        await conn.execute("DELETE FROM training_corpus_versions WHERE data_source='initial'")
        
        print('✅ Database cleaned successfully')
        print('')
        return True


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run initial corpus collection for transformer training'
    )
    
    # Mutually exclusive group for preset configurations
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--test',
        action='store_true',
        help='Run test collection (2 tokens, 7 days)'
    )
    group.add_argument(
        '--production',
        action='store_true',
        help='Run full production collection (10 tokens, 180 days)'
    )
    group.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing initial corpus data from database'
    )
    
    # Custom configuration options
    parser.add_argument(
        '--tokens',
        nargs='+',
        help='Custom list of token IDs to collect'
    )
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days of historical data to collect'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=2.1,
        help='Rate limit delay between API calls (default: 2.1 seconds)'
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Number of retry attempts for failed requests (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Handle clean option first
    if args.clean:
        print('🧹 Database Cleanup Mode')
        print('')
        
        # Check database connection
        if not os.environ.get('DATABASE_URL'):
            print('❌ DATABASE_URL environment variable not set')
            print('Please run with the deployment script:')
            print('   ./scripts/data_collection/collect_initial_corpus.sh clean')
            sys.exit(1)
        
        # Run cleanup
        success = asyncio.run(clean_existing_corpus())
        sys.exit(0 if success else 1)
    
    # Determine configuration for collection
    if args.test:
        tokens = TEST_TOKENS
        days = 7
        print('🧪 Running TEST collection...')
    elif args.production:
        tokens = PRODUCTION_TOKENS
        days = 180
        print('🚀 Running PRODUCTION collection...')
    else:
        # Use custom configuration or defaults
        tokens = args.tokens if args.tokens else TEST_TOKENS
        days = args.days if args.days else 7
        print('⚙️  Running CUSTOM collection...')
    
    print('')
    
    # Check required environment variables
    required_vars = ['DATABASE_URL', 'COINGECKO_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print('❌ Missing required environment variables:')
        for var in missing_vars:
            print(f'   - {var}')
        print('')
        print('Please set these variables or run with the deployment script:')
        print('   ./scripts/data_collection/collect_initial_corpus.sh')
        sys.exit(1)
    
    # Run async collection
    success = asyncio.run(run_collection(
        tokens=tokens,
        days=days,
        rate_limit_delay=args.rate_limit,
        retry_attempts=args.retries
    ))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()