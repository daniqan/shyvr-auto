#!/usr/bin/env python
"""
Clean corpus data from the database

This script provides utilities to clean corpus data with different granularities
from the training database.
"""

import asyncio
import argparse
import sys
import logging
from typing import List, Optional
import structlog

from src.utils.database import get_database_connection

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stderr
)

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
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def clean_all_corpus_data() -> int:
    """
    Clean all corpus data from the database
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Starting clean_all_corpus_data()")
    try:
        logger.info("Getting database connection...")
        async with get_database_connection() as conn:
            logger.info("Database connection established")
            # Clean corpus tables
            tables = [
                'crypto_ohlcv', 
                'crypto_features', 
                'market_sentiment',
                'defi_metrics', 
                'onchain_metrics', 
                'social_sentiment',
                'training_corpus_versions'
            ]
            
            total_deleted = 0
            for table in tables:
                logger.info(f"Cleaning table {table}...")
                query = f"DELETE FROM {table} WHERE data_source = 'initial'"
                logger.debug(f"Executing: {query}")
                result = await conn.execute(query)
                
                # Extract row count from result string (e.g., "DELETE 42")
                if result and ' ' in result:
                    count = int(result.split(' ')[1]) if result.split(' ')[1].isdigit() else 0
                else:
                    count = 0
                    
                total_deleted += count
                logger.info(f"Cleaned {table}", rows_deleted=count)
            
            logger.info("✅ Corpus data cleaned", total_rows_deleted=total_deleted)
            return 0
            
    except Exception as e:
        logger.error(f"Failed to clean corpus data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


async def clean_multi_granularity_data() -> int:
    """
    Clean only multi-granularity corpus data from the database
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        async with get_database_connection() as conn:
            # Check if granularity column exists
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'crypto_ohlcv' 
                    AND column_name = 'granularity'
                )
            """
            has_granularity = await conn.fetchval(check_query)
            
            if not has_granularity:
                logger.warning("Granularity column does not exist yet")
                return 0
            
            # Clean multi-granularity OHLCV data
            result = await conn.execute("""
                DELETE FROM crypto_ohlcv 
                WHERE data_source = 'initial' 
                AND granularity IS NOT NULL 
                AND granularity != ''
            """)
            
            ohlcv_count = int(result.split(' ')[1]) if result and ' ' in result else 0
            logger.info(f"Cleaned multi-granularity OHLCV data", rows_deleted=ohlcv_count)
            
            # Clean multi-granularity features data
            result = await conn.execute("""
                DELETE FROM crypto_features 
                WHERE data_source = 'initial' 
                AND granularity IS NOT NULL 
                AND granularity != ''
            """)
            
            features_count = int(result.split(' ')[1]) if result and ' ' in result else 0
            logger.info(f"Cleaned multi-granularity feature data", rows_deleted=features_count)
            
            # Clean corpus versions with multi-granularity metadata
            result = await conn.execute("""
                DELETE FROM training_corpus_versions 
                WHERE data_source = 'initial' 
                AND (
                    metadata::text LIKE '%multi_granularity%'
                    OR granularities IS NOT NULL
                )
            """)
            
            versions_count = int(result.split(' ')[1]) if result and ' ' in result else 0
            logger.info(f"Cleaned multi-granularity corpus versions", rows_deleted=versions_count)
            
            total_deleted = ohlcv_count + features_count + versions_count
            logger.info("✅ Multi-granularity corpus data cleaned", total_rows_deleted=total_deleted)
            return 0
            
    except Exception as e:
        logger.error(f"Failed to clean multi-granularity data: {e}")
        return 1


async def clean_specific_tokens(tokens: List[str]) -> int:
    """
    Clean corpus data for specific tokens
    
    Args:
        tokens: List of token symbols to clean
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        async with get_database_connection() as conn:
            # Build token list for query
            token_list = "', '".join(tokens)
            
            # Clean OHLCV data for specific tokens
            result = await conn.execute(f"""
                DELETE FROM crypto_ohlcv 
                WHERE data_source = 'initial' 
                AND symbol IN ('{token_list}')
            """)
            
            ohlcv_count = int(result.split(' ')[1]) if result and ' ' in result else 0
            logger.info(f"Cleaned OHLCV data for {tokens}", rows_deleted=ohlcv_count)
            
            # Clean features data for specific tokens
            result = await conn.execute(f"""
                DELETE FROM crypto_features 
                WHERE data_source = 'initial' 
                AND token_id IN (
                    SELECT DISTINCT token_id FROM crypto_ohlcv 
                    WHERE symbol IN ('{token_list}')
                )
            """)
            
            features_count = int(result.split(' ')[1]) if result and ' ' in result else 0
            logger.info(f"Cleaned feature data for {tokens}", rows_deleted=features_count)
            
            total_deleted = ohlcv_count + features_count
            logger.info(f"✅ Cleaned data for tokens: {tokens}", total_rows_deleted=total_deleted)
            return 0
            
    except Exception as e:
        logger.error(f"Failed to clean token data: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Clean corpus data from database')
    parser.add_argument(
        '--mode',
        choices=['all', 'multi', 'tokens'],
        default='all',
        help='Cleaning mode: all (default), multi (multi-granularity only), tokens (specific tokens)'
    )
    parser.add_argument(
        '--tokens',
        nargs='+',
        help='Specific tokens to clean (used with --mode tokens)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Confirmation prompt unless --yes is provided
    if not args.yes:
        if args.mode == 'all':
            print("⚠️  WARNING: This will delete ALL corpus data!")
        elif args.mode == 'multi':
            print("⚠️  WARNING: This will delete all multi-granularity corpus data!")
        elif args.mode == 'tokens' and args.tokens:
            print(f"⚠️  WARNING: This will delete corpus data for tokens: {args.tokens}")
        else:
            print("Error: --tokens required when using --mode tokens")
            sys.exit(1)
            
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)
    
    # Run the appropriate cleaning function
    if args.mode == 'all':
        exit_code = asyncio.run(clean_all_corpus_data())
    elif args.mode == 'multi':
        exit_code = asyncio.run(clean_multi_granularity_data())
    elif args.mode == 'tokens':
        if not args.tokens:
            print("Error: --tokens required when using --mode tokens")
            sys.exit(1)
        exit_code = asyncio.run(clean_specific_tokens(args.tokens))
    else:
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()