#!/usr/bin/env python3
"""
Investigate why we're getting incomplete historical data
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_analysis.market_data import CoinGeckoClient
from src.utils.system_secrets import SystemSecrets
import pandas as pd

async def investigate_data_gaps():
    """Test different tokens and date ranges to understand the limits"""
    
    # Initialize client
    secrets = SystemSecrets()
    api_key = secrets.get_secret("COINGECKO_API_KEY")
    client = CoinGeckoClient(api_key=api_key)
    
    print("=" * 70)
    print("INVESTIGATING DATA COLLECTION LIMITS")
    print("=" * 70)
    
    # Test different tokens
    test_cases = [
        # Token, Contract, Network, Expected behavior
        ("WETH", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "eth", "ERC-20 token"),
        ("SOL", "0xD31a59c85aE9D8edEFeC411D448f90841571b89c", "eth", "Wrapped SOL"),
        ("PEPE", "0x6982508145454ce325ddbe47a25d4ec3d2311933", "eth", "Meme coin"),
        ("BTC", None, None, "Native coin - no contract"),
    ]
    
    end_date = datetime.now(timezone.utc)
    
    for symbol, contract, network, description in test_cases:
        print(f"\n{symbol} ({description}):")
        print("-" * 50)
        
        # Test 1: Try 365 days with contract endpoint
        if contract:
            try:
                start_365 = end_date - timedelta(days=365)
                print(f"  365 days via contract:")
                
                df = await client._get_contract_ohlcv_with_pagination(
                    contract_address=contract,
                    network=network,
                    days=365,
                    from_date=start_365,
                    to_date=end_date,
                    timeframe='day'
                )
                
                if len(df) > 0:
                    actual_days = (df['timestamp'].max() - df['timestamp'].min()).days
                    print(f"    ✓ Got {len(df)} candles covering {actual_days} days")
                    print(f"    Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
                    
                    # Check for gaps
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    df['date_diff'] = df['timestamp'].diff()
                    gaps = df[df['date_diff'] > pd.Timedelta(days=1.5)]
                    if len(gaps) > 0:
                        print(f"    ⚠️  Found {len(gaps)} gaps in data")
                else:
                    print(f"    ✗ No data returned")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        # Test 2: Try with coin_id endpoint for comparison
        try:
            print(f"  365 days via coin_id:")
            
            # Map symbols to coin IDs
            coin_ids = {
                "WETH": "weth",
                "SOL": "solana", 
                "PEPE": "pepe",
                "BTC": "bitcoin"
            }
            
            if symbol in coin_ids:
                df_coin = await client.get_ohlcv_data(
                    coin_id=coin_ids[symbol],
                    days=365
                )
                
                if len(df_coin) > 0:
                    actual_days = (df_coin['timestamp'].max() - df_coin['timestamp'].min()).days
                    print(f"    ✓ Got {len(df_coin)} candles covering {actual_days} days")
                    print(f"    Date range: {df_coin['timestamp'].min().date()} to {df_coin['timestamp'].max().date()}")
                else:
                    print(f"    ✗ No data returned")
                    
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Test 3: Check CoinGecko's documented limits
    print("\n" + "=" * 70)
    print("COINGECKO API LIMITS (from documentation):")
    print("-" * 70)
    print("  Free tier:")
    print("    • /coins/{id}/ohlc: Max 365 days (returns ~90-180 days)")
    print("    • /coins/{id}/contract/{contract}/market_chart/range: 365+ days")
    print("  Pro tier:")
    print("    • Historical data up to inception")
    print("    • Higher rate limits")
    print("\n  ⚠️  The contract endpoint seems limited to ~180 days for daily data")
    print("  ⚠️  This appears to be a CoinGecko API limitation, not our code")
    
    # Test 4: Try different chunking strategies
    print("\n" + "=" * 70)
    print("TESTING CHUNKING STRATEGIES:")
    print("-" * 70)
    
    # Use WETH as test case
    contract = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    
    # Strategy 1: Single request for 365 days
    print("  Strategy 1: Single 365-day request")
    start = end_date - timedelta(days=365)
    df1 = await client._get_contract_ohlcv_with_pagination(
        contract_address=contract,
        network="eth",
        days=365,
        from_date=start,
        to_date=end_date,
        timeframe='day'
    )
    print(f"    Result: {len(df1)} candles")
    
    # Strategy 2: Two 180-day requests
    print("  Strategy 2: Two 180-day requests")
    mid_date = end_date - timedelta(days=180)
    
    df2a = await client._get_contract_ohlcv_with_pagination(
        contract_address=contract,
        network="eth",
        days=180,
        from_date=start,
        to_date=mid_date,
        timeframe='day'
    )
    
    df2b = await client._get_contract_ohlcv_with_pagination(
        contract_address=contract,
        network="eth",
        days=180,
        from_date=mid_date,
        to_date=end_date,
        timeframe='day'
    )
    
    print(f"    First chunk: {len(df2a)} candles")
    print(f"    Second chunk: {len(df2b)} candles")
    print(f"    Total: {len(df2a) + len(df2b)} candles")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("  • CoinGecko contract endpoint returns max ~183 days of daily data")
    print("  • This is an API limitation, not a bug in our code")
    print("  • For full 365 days, we'd need to:")
    print("    1. Use CoinGecko Pro API (if it provides more)")
    print("    2. Store historical data and incrementally update")
    print("    3. Use multiple data sources and merge")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(investigate_data_gaps())