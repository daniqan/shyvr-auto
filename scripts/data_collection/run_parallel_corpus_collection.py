#!/usr/bin/env python3
"""
Run parallel corpus collection with improved rate limiting
Optimized for production data collection
"""

import asyncio
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.parallel_corpus_collector import ParallelCorpusCollector
from src.utils.config import DatabaseConfig
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Production tokens list (top 10 by market cap)
PRODUCTION_TOKENS = [
    'bitcoin',
    'ethereum', 
    'tether',
    'binancecoin',
    'solana',
    'ripple',
    'dogecoin',
    'cardano',
    'avalanche-2',
    'chainlink'
]

# Test tokens (smaller set for testing)
TEST_TOKENS = [
    'bitcoin',
    'ethereum'
]


async def main():
    parser = argparse.ArgumentParser(description='Run parallel corpus collection')
    parser.add_argument(
        '--mode',
        choices=['test', 'production', 'custom'],
        default='test',
        help='Collection mode'
    )
    parser.add_argument(
        '--tokens',
        nargs='+',
        help='Custom list of tokens (for custom mode)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data to collect'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing data before collection'
    )
    parser.add_argument(
        '--max-concurrent-tokens',
        type=int,
        default=3,
        help='Maximum tokens to process concurrently'
    )
    parser.add_argument(
        '--max-concurrent-api',
        type=int,
        default=5,
        help='Maximum concurrent API calls per token'
    )
    
    args = parser.parse_args()
    
    # Determine tokens and timeframe based on mode
    if args.mode == 'test':
        tokens = TEST_TOKENS
        days = 7
        logger.info("Running in TEST mode", tokens=tokens, days=days)
    elif args.mode == 'production':
        tokens = PRODUCTION_TOKENS
        days = 180
        logger.info("Running in PRODUCTION mode", tokens=tokens, days=days)
    elif args.mode == 'custom':
        if not args.tokens:
            print("Error: --tokens required for custom mode")
            sys.exit(1)
        tokens = args.tokens
        days = args.days
        logger.info("Running in CUSTOM mode", tokens=tokens, days=days)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    # Set up dates
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Initialize database config
    try:
        db_config = DatabaseConfig()
        logger.info("Database configured", host=db_config.host, port=db_config.port)
    except Exception as e:
        logger.error("Failed to configure database", error=str(e))
        sys.exit(1)
    
    # Create collector with optimized settings
    collector = ParallelCorpusCollector(
        database_config=db_config,
        rate_limit_delay=0.2,  # 200ms between sequential calls
        max_concurrent_tokens=args.max_concurrent_tokens,
        max_concurrent_api_calls=args.max_concurrent_api,
        chunk_size=30,  # 30-day chunks for hourly granularity
        retry_attempts=3,
        timeout_seconds=30
    )
    
    # Run collection
    try:
        logger.info("Starting parallel corpus collection...")
        stats = await collector.collect_corpus(
            tokens=tokens,
            start_date=start_date,
            end_date=end_date,
            clean_existing=args.clean
        )
        
        # Print summary
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Duration: {stats.get('duration_seconds', 0):.1f} seconds")
        print(f"Successful tokens: {stats.get('successful_tokens', 0)}/{len(tokens)}")
        print(f"Failed tokens: {stats.get('failed_tokens', 0)}")
        print(f"Total records: {stats.get('total_records', 0):,}")
        print(f"Total features: {stats.get('total_features', 0):,}")
        print(f"API calls made: {stats.get('api_calls_made', 0):,}")
        print(f"Errors encountered: {stats.get('errors_encountered', 0)}")
        
        if collector.failed_tokens:
            print("\nFailed tokens:")
            for token, error in collector.failed_tokens.items():
                print(f"  - {token}: {error}")
        
        # Export to GCS if in production mode
        if args.mode == 'production' and stats.get('total_records', 0) > 0:
            print("\n" + "="*60)
            print("EXPORTING TO GCS")
            print("="*60)
            
            try:
                export_stats = await collector.export_to_gcs(
                    bucket_name='shyvr-rlte-corpus',
                    prefix='initial_corpus'
                )
                print(f"✅ Exported {export_stats['files_exported']} files to GCS")
                print(f"   Total size: {export_stats['total_size_mb']:.2f} MB")
            except Exception as e:
                print(f"❌ Failed to export to GCS: {e}")
        
        print("="*60)
        
        # Return success if we collected data
        return 0 if stats.get('total_records', 0) > 0 else 1
        
    except Exception as e:
        logger.error("Collection failed", error=str(e), exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)