#!/usr/bin/env python3
"""
Quick test script to verify SOL data collection with wrapped contract address
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_analysis.market_data import CoinGeckoClient
from src.utils.system_secrets import SystemSecrets
import structlog

logger = structlog.get_logger(__name__)

async def test_sol_collection():
    """Test SOL data collection with both methods"""
    
    # Initialize secrets and client
    secrets = SystemSecrets()
    api_key = secrets.get_secret("COINGECKO_API_KEY")
    
    client = CoinGeckoClient(api_key=api_key)
    
    print("\n" + "="*60)
    print("Testing SOL Data Collection")
    print("="*60)
    
    # Test 1: Coin ID endpoint (original method)
    print("\n1. Testing coin_id endpoint (original)...")
    try:
        coin_data = await client.get_ohlcv_data(
            coin_id="solana",
            days=365
        )
        print(f"   ✓ Coin ID endpoint returned {len(coin_data)} daily records")
        if len(coin_data) > 0:
            print(f"   Date range: {coin_data['timestamp'].min()} to {coin_data['timestamp'].max()}")
    except Exception as e:
        print(f"   ✗ Coin ID endpoint failed: {e}")
    
    # Test 2: Contract endpoint with wrapped SOL
    print("\n2. Testing contract endpoint (wrapped SOL)...")
    try:
        from datetime import datetime, timedelta, timezone
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)
        
        contract_data = await client._get_contract_ohlcv_with_pagination(
            contract_address="0xD31a59c85aE9D8edEFeC411D448f90841571b89c",
            network="eth",
            days=365,
            from_date=start_date,
            to_date=end_date,
            timeframe='day'
        )
        print(f"   ✓ Contract endpoint returned {len(contract_data)} daily records")
        if len(contract_data) > 0:
            print(f"   Date range: {contract_data['timestamp'].min()} to {contract_data['timestamp'].max()}")
    except Exception as e:
        print(f"   ✗ Contract endpoint failed: {e}")
    
    # Test 3: Compare hourly data
    print("\n3. Testing hourly data collection...")
    try:
        hourly_data = await client._get_contract_ohlcv_with_pagination(
            contract_address="0xD31a59c85aE9D8edEFeC411D448f90841571b89c",
            network="eth",
            days=90,
            from_date=end_date - timedelta(days=90),
            to_date=end_date,
            timeframe='hour'
        )
        print(f"   ✓ Contract endpoint returned {len(hourly_data)} hourly records")
        expected_hourly = 90 * 24  # 90 days * 24 hours
        completeness = (len(hourly_data) / expected_hourly) * 100
        print(f"   Data completeness: {completeness:.1f}% ({len(hourly_data)}/{expected_hourly})")
    except Exception as e:
        print(f"   ✗ Hourly data failed: {e}")
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("- Coin ID gives limited data (~92 days)")
    print("- Contract endpoint should give full 365 days")
    print("- Use wrapped SOL (0xD31a59c85aE9D8edEFeC411D448f90841571b89c)")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_sol_collection())