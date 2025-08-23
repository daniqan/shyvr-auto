#!/usr/bin/env python
"""
Multi-Granularity Corpus Collection Script
Collects training data at multiple timeframes for comprehensive market analysis

Usage:
    python scripts/collect_multi_granularity_corpus.py --config config/corpus_collection.yaml
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.multi_granularity_collector import MultiGranularityCollector
# from src.data_pipeline.gcs_corpus_exporter import GCSCorpusExporter  # TODO: Fix GCS export
from src.utils.system_secrets import get_system_secrets
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

# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)


async def main():
    """Main collection function"""
    logger.info("Starting main() function")
    parser = argparse.ArgumentParser(description='Collect multi-granularity training corpus')
    parser.add_argument(
        '--config',
        type=str,
        default='config/corpus_collection.yaml',
        help='Path to corpus collection configuration'
    )
    parser.add_argument(
        '--export-to-gcs',
        action='store_true',
        help='Export collected data to Google Cloud Storage as Parquet files'
    )
    parser.add_argument(
        '--export-to-parquet',
        action='store_true',
        help='Export collected data to local Parquet files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual data collection'
    )
    parser.add_argument(
        '--tokens',
        nargs='+',
        help='Specific tokens to collect (overrides config)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        choices=['daily', 'four_hour', 'hourly', 'fifteen_minute'],
        help='Specific timeframes to collect (overrides config)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Use parallel collection (default: True)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for saving collection checkpoints'
    )
    parser.add_argument(
        '--enable-pagination',
        action='store_true',
        help='Enable dynamic pagination for full data collection (365 days)'
    )
    
    args = parser.parse_args()
    logger.info(f"Arguments parsed: config={args.config}, dry_run={args.dry_run}")
    
    # Load secrets
    try:
        logger.info("Loading system secrets...")
        system_secrets = get_system_secrets()
        logger.info("System secrets loaded successfully")
        coingecko_api_key = system_secrets.coingecko_pro_api_key
        
        if not coingecko_api_key:
            logger.error("CoinGecko Pro API key not found in secrets")
            return 1
        else:
            logger.info("CoinGecko Pro API key found")
            
    except Exception as e:
        logger.error(f"Failed to load secrets: {e}")
        return 1
    
    # Initialize collector
    try:
        logger.info(f"Initializing MultiGranularityCollector with config: {args.config}")
        collector = MultiGranularityCollector(
            config_path=args.config,
            rate_limit_delay=2.1,  # Conservative rate limiting
            retry_attempts=3
        )
        logger.info("MultiGranularityCollector initialized successfully")
        
        # Override tokens if specified
        if args.tokens:
            logger.info(f"Overriding tokens from command line: {args.tokens}")
            # Filter collector.tokens to only include specified tokens
            collector.tokens = {
                symbol: config 
                for symbol, config in collector.tokens.items() 
                if symbol in args.tokens
            }
        
        # Override timeframes if specified
        if args.timeframes:
            logger.info(f"Overriding timeframes from command line: {args.timeframes}")
            # Disable timeframes not in the list
            for tf_enum, tf_config in collector.timeframes.items():
                if tf_enum.value not in args.timeframes:
                    tf_config.enabled = False
        
        # Set parallel collection
        collector.collection_strategy['parallel_collection'] = args.parallel
        
        # Enable dynamic pagination if requested
        if args.enable_pagination:
            logger.info("Enabling dynamic pagination for full data collection")
            # Update pagination settings in config
            if 'pagination' not in collector.config:
                collector.config['pagination'] = {}
            collector.config['pagination']['enabled'] = True
            collector.config['pagination']['strategy'] = {
                'direction': 'backward',
                'overlap_periods': 1,
                'max_retries': 3,
                'retry_delay': 2,
                'fill_gaps': True
            }
            # Force reinitialization of paginated collector with new settings
            collector.paginated_collector = None
        
    except Exception as e:
        logger.error(f"Failed to initialize collector: {e}")
        return 1
    
    # Perform collection
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Showing collection plan")
        print("="*60)
        print(f"Tokens to collect: {list(collector.tokens.keys())}")
        enabled_timeframes = [tf.value for tf, config in collector.timeframes.items() if config.enabled]
        print(f"Timeframes to collect: {enabled_timeframes}")
        print("")
        
        total_expected_candles = 0
        for token in collector.tokens.keys():
            for tf_enum, tf_config in collector.timeframes.items():
                if tf_config.enabled:
                    print(f"  {token} @ {tf_enum.value}: ~{tf_config.expected_candles:,} candles")
                    total_expected_candles += tf_config.expected_candles
        
        print("")
        print(f"Total expected candles: {total_expected_candles:,}")
        api_calls = total_expected_candles // 1000 + len(collector.tokens) * len(enabled_timeframes)
        print(f"Estimated API calls: {api_calls}")
        print(f"Estimated time: ~{api_calls * 2.1 / 60:.1f} minutes with rate limiting")
        print("="*60)
        
        return 0
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint_file = checkpoint_dir / 'collection_checkpoint.json'
    completed_keys = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            completed_keys = set(checkpoint.get('completed', []))
            logger.info(f"Resuming from checkpoint, {len(completed_keys)} datasets already collected")
    
    try:
        # Collect corpus data
        logger.info("Starting multi-granularity corpus collection...")
        logger.info(f"Tokens to collect: {list(collector.tokens.keys())}")
        enabled_timeframes = [tf.value for tf, config in collector.timeframes.items() if config.enabled]
        logger.info(f"Enabled timeframes: {enabled_timeframes}")
        start_time = datetime.now()
        
        corpus_data = await collector.collect_multi_granularity_corpus()
        logger.info(f"Collection returned {len(corpus_data)} datasets")
        
        # Save checkpoint after successful collection
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'completed': list(corpus_data.keys()),
                'timestamp': datetime.now().isoformat(),
                'total_candles': sum(len(df) for df in corpus_data.values())
            }, f)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Collection completed in {duration:.2f} seconds",
            datasets=len(corpus_data),
            total_candles=sum(len(df) for df in corpus_data.values())
        )
        
        # Store to database
        logger.info("Storing corpus to database...")
        await collector.store_corpus_to_database(corpus_data)
        
        # Export to GCS if requested
        if args.export_to_gcs:
            logger.info("Exporting corpus to Google Cloud Storage as Parquet files...")
            
            try:
                # Generate timestamp-based folder name following v1.0 convention
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gcs_prefix = f"training-data/initial-corpus/initial_{collector.corpus_version}_{timestamp}"
                
                gcs_paths = await collector.export_to_gcs(
                    corpus_data=corpus_data,
                    bucket_name="shyvr-models-prod",
                    gcs_prefix=gcs_prefix
                )
                
                logger.info(
                    "Corpus exported to GCS",
                    total_files=len(gcs_paths),
                    sample_path=next(iter(gcs_paths.values())) if gcs_paths else None
                )
            except Exception as e:
                logger.error(f"Failed to export to GCS: {e}")
                logger.info("Falling back to local Parquet export...")
                
                # Export to local Parquet files as fallback
                local_paths = await collector.export_to_parquet(corpus_data)
                logger.info(
                    "Corpus exported locally as Parquet",
                    total_files=len(local_paths),
                    directory="data/corpus/v2.0"
                )
        
        # Export to local Parquet if requested (without GCS)
        elif args.export_to_parquet:
            logger.info("Exporting corpus to local Parquet files...")
            
            local_paths = await collector.export_to_parquet(corpus_data)
            logger.info(
                "Corpus exported as Parquet",
                total_files=len(local_paths),
                directory="data/corpus/v2.0"
            )
            # logger.info("Exporting corpus to Google Cloud Storage...")
            # 
            # exporter = GCSCorpusExporter(
            #     bucket_name="shyvr-models-prod",
            #     corpus_version="v2.0"
            # )
            # 
            # export_result = await exporter.export_corpus(
            #     corpus_data=corpus_data,
            #     metadata={
            #         'collection_date': datetime.now().isoformat(),
            #         'tokens': list(collector.tokens.keys()),
            #         'timeframes': [tf.value for tf, config in collector.timeframes.items() if config.enabled],
            #         'total_candles': sum(len(df) for df in corpus_data.values()),
            #         'collection_duration_seconds': duration
            #     }
            # )
            # 
            # logger.info(f"Corpus exported to GCS: {export_result['gcs_path']}")
        
        # Print summary statistics
        logger.info("\n" + "="*50)
        logger.info("COLLECTION SUMMARY")
        logger.info("="*50)
        
        for key, df in corpus_data.items():
            token, timeframe = key.rsplit('_', 1)
            logger.info(
                f"{token} @ {timeframe}:",
                candles=len(df),
                date_range=f"{df.iloc[0]['timestamp'] if 'timestamp' in df.columns else 'N/A'} to {df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else 'N/A'}",
                features=len(df.columns)
            )
        
        logger.info("="*50)
        logger.info(
            "TOTAL:",
            datasets=len(corpus_data),
            candles=sum(len(df) for df in corpus_data.values()),
            features=len(next(iter(corpus_data.values())).columns) if corpus_data else 0
        )
        
        # Clean up checkpoint on success
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint file removed after successful completion")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Collection interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Clean up resources
        if hasattr(collector, 'coingecko_client') and collector.coingecko_client:
            await collector.coingecko_client.close()
        
        logger.info("Collection script completed")


if __name__ == "__main__":
    logger.info("Script started")
    try:
        exit_code = asyncio.run(main())
        logger.info(f"Script completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Script failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)