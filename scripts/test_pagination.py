#!/usr/bin/env python
"""
Test script for the new pagination system
Verifies that the PaginatedDataCollector can retrieve full 365 days of data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.paginated_collector import PaginatedDataCollector
from src.ml_analysis.market_data import CoinGeckoClient
from src.utils.system_secrets import get_system_secrets
import structlog

# Configure logging
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


async def test_single_token_pagination():
    """Test pagination for a single token"""
    
    # Load API key
    system_secrets = get_system_secrets()
    api_key = system_secrets.coingecko_pro_api_key
    
    if not api_key:
        logger.error("CoinGecko Pro API key not found")
        return False
    
    # Initialize CoinGecko client
    coingecko_client = CoinGeckoClient(api_key=api_key, use_pro=True)
    
    # Initialize paginated collector
    paginated_collector = PaginatedDataCollector(
        coingecko_client=coingecko_client,
        checkpoint_dir="checkpoints/test_pagination"
    )
    
    # Test with WETH for 365 days of daily data
    token_config = {
        'symbol': 'WETH',
        'contract_address': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
        'network': 'eth'
    }
    
    logger.info("Starting pagination test for WETH")
    logger.info("Target: 365 days of daily data")
    
    try:
        # Collect data with pagination
        df = await paginated_collector.collect_with_pagination(
            contract_address=token_config['contract_address'],
            network=token_config['network'],
            token_symbol=token_config['symbol'],
            timeframe='daily',
            days=365,
            resume=False  # Start fresh for test
        )
        
        logger.info(f"Collection completed!")
        logger.info(f"Total records collected: {len(df)}")
        
        if not df.empty:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            logger.info(f"Date range covered: {date_range} days")
            logger.info(f"From: {df['timestamp'].min()}")
            logger.info(f"To: {df['timestamp'].max()}")
            
            # Check completeness
            expected_candles = 365
            completeness = (len(df) / expected_candles) * 100
            logger.info(f"Data completeness: {completeness:.1f}%")
            
            if completeness >= 85:
                logger.info("✅ Test PASSED: Sufficient data collected")
                return True
            else:
                logger.warning(f"⚠️ Test WARNING: Only {completeness:.1f}% data collected")
                return False
        else:
            logger.error("❌ Test FAILED: No data collected")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_timeframes():
    """Test pagination for multiple timeframes"""
    
    # Load API key
    system_secrets = get_system_secrets()
    api_key = system_secrets.coingecko_pro_api_key
    
    if not api_key:
        logger.error("CoinGecko Pro API key not found")
        return False
    
    # Initialize CoinGecko client
    coingecko_client = CoinGeckoClient(api_key=api_key, use_pro=True)
    
    # Initialize paginated collector
    paginated_collector = PaginatedDataCollector(
        coingecko_client=coingecko_client,
        checkpoint_dir="checkpoints/test_pagination"
    )
    
    # Test configurations
    test_configs = [
        {'timeframe': 'daily', 'days': 365, 'expected_min': 300},
        {'timeframe': 'hourly', 'days': 30, 'expected_min': 600},
    ]
    
    token_config = {
        'symbol': 'USDC',
        'contract_address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
        'network': 'eth'
    }
    
    all_passed = True
    
    for test in test_configs:
        logger.info(f"Testing {test['timeframe']} for {test['days']} days")
        
        try:
            df = await paginated_collector.collect_with_pagination(
                contract_address=token_config['contract_address'],
                network=token_config['network'],
                token_symbol=token_config['symbol'],
                timeframe=test['timeframe'],
                days=test['days'],
                resume=False
            )
            
            if len(df) >= test['expected_min']:
                logger.info(f"✅ {test['timeframe']}: Collected {len(df)} candles (expected >= {test['expected_min']})")
            else:
                logger.warning(f"⚠️ {test['timeframe']}: Only {len(df)} candles (expected >= {test['expected_min']})")
                all_passed = False
                
        except Exception as e:
            logger.error(f"❌ {test['timeframe']} failed: {e}")
            all_passed = False
    
    return all_passed


async def main():
    """Run all tests"""
    
    print("\n" + "="*60)
    print("PAGINATION SYSTEM TEST")
    print("="*60)
    
    # Test 1: Single token with 365 days
    print("\n📊 Test 1: Single token pagination (365 days)")
    print("-"*40)
    test1_result = await test_single_token_pagination()
    
    # Test 2: Multiple timeframes
    print("\n📊 Test 2: Multiple timeframes")
    print("-"*40)
    test2_result = await test_multiple_timeframes()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Test 1 (Single token, 365 days): {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Test 2 (Multiple timeframes): {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)