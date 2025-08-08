#!/usr/bin/env python
"""
Test script to investigate CoinGecko API granularity for different endpoints
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml_analysis.market_data import CoinGeckoClient


async def test_ohlc_endpoint():
    """Test the /ohlc endpoint for different time ranges"""
    client = CoinGeckoClient(api_key=os.environ.get('COINGECKO_API_KEY'))
    
    print("=" * 80)
    print("Testing CoinGecko /ohlc endpoint granularity")
    print("=" * 80)
    print()
    
    # Test different time ranges
    test_ranges = [
        (1, "1 day"),
        (7, "7 days"),
        (30, "30 days"),
        (90, "90 days"),
        (180, "180 days"),
    ]
    
    token = 'bitcoin'
    
    for days, description in test_ranges:
        print(f"\n📊 Testing {description} range for {token}:")
        print("-" * 40)
        
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get OHLC data
            df = await client.get_ohlcv_data(
                coin_id=token,
                days=days,
                from_date=start_date,
                to_date=end_date
            )
            
            if not df.empty and len(df) > 1:
                # Calculate granularity
                time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
                hours = time_diff.total_seconds() / 3600
                
                print(f"   Records returned: {len(df)}")
                print(f"   Granularity: {hours:.1f} hours between data points")
                print(f"   Start: {df['timestamp'].iloc[0]}")
                print(f"   End: {df['timestamp'].iloc[-1]}")
                print(f"   Expected hourly records: {days * 24}")
                print(f"   Coverage: {len(df) / (days * 24) * 100:.1f}%")
            else:
                print(f"   No data returned or insufficient records")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    await client.close()


async def test_market_chart_endpoint():
    """Test the /market_chart endpoint for different time ranges"""
    client = CoinGeckoClient(api_key=os.environ.get('COINGECKO_API_KEY'))
    
    print("\n" + "=" * 80)
    print("Testing CoinGecko /market_chart endpoint granularity")
    print("=" * 80)
    print()
    
    # Test different time ranges
    test_ranges = [
        (1, "1 day"),
        (7, "7 days"),
        (30, "30 days"),
        (90, "90 days"),
        (180, "180 days"),
    ]
    
    token = 'bitcoin'
    
    for days, description in test_ranges:
        print(f"\n📈 Testing {description} range for {token}:")
        print("-" * 40)
        
        try:
            # Directly call the market_chart endpoint
            params = {"vs_currency": "usd", "days": days}
            
            if days == 180:
                # For 180 days, also try with from/to parameters
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=days)
                params["from"] = int(start_date.timestamp())
                params["to"] = int(end_date.timestamp())
            
            url = f"{client.base_url}/coins/{token}/market_chart"
            data = await client._make_request(url, params=params, headers=client.headers)
            
            if data and 'prices' in data:
                prices = data['prices']
                print(f"   Price data points: {len(prices)}")
                
                if len(prices) > 1:
                    # Calculate granularity
                    time_diff_ms = prices[1][0] - prices[0][0]
                    hours = time_diff_ms / (1000 * 3600)
                    
                    print(f"   Granularity: {hours:.1f} hours between data points")
                    print(f"   Start: {datetime.fromtimestamp(prices[0][0]/1000, tz=timezone.utc)}")
                    print(f"   End: {datetime.fromtimestamp(prices[-1][0]/1000, tz=timezone.utc)}")
                    print(f"   Expected hourly records: {days * 24}")
                    print(f"   Coverage: {len(prices) / (days * 24) * 100:.1f}%")
                    
                    # Check if we can construct OHLCV from this data
                    if hours <= 1.5:  # If granularity is hourly or better
                        print(f"   ✅ Good granularity for training data!")
                    else:
                        print(f"   ⚠️  Granularity too coarse for ideal training")
            else:
                print(f"   No price data returned")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    await client.close()


async def test_market_chart_range_endpoint():
    """Test the /market_chart/range endpoint for specific date ranges"""
    client = CoinGeckoClient(api_key=os.environ.get('COINGECKO_API_KEY'))
    
    print("\n" + "=" * 80)
    print("Testing CoinGecko /market_chart/range endpoint")
    print("=" * 80)
    print()
    
    token = 'bitcoin'
    
    # Test fetching in smaller chunks for 180 days
    print(f"\n🔄 Testing chunked approach for 180 days:")
    print("-" * 40)
    
    total_data_points = 0
    chunk_days = 30  # Fetch 30-day chunks
    
    end_date = datetime.now(timezone.utc)
    
    for i in range(6):  # 6 chunks of 30 days = 180 days
        chunk_start = end_date - timedelta(days=(i+1) * chunk_days)
        chunk_end = end_date - timedelta(days=i * chunk_days)
        
        try:
            params = {
                "vs_currency": "usd",
                "from": int(chunk_start.timestamp()),
                "to": int(chunk_end.timestamp())
            }
            
            url = f"{client.base_url}/coins/{token}/market_chart/range"
            data = await client._make_request(url, params=params, headers=client.headers)
            
            if data and 'prices' in data:
                prices = data['prices']
                total_data_points += len(prices)
                
                if len(prices) > 1:
                    time_diff_ms = prices[1][0] - prices[0][0]
                    hours = time_diff_ms / (1000 * 3600)
                    
                    print(f"   Chunk {i+1} ({chunk_start.date()} to {chunk_end.date()}):")
                    print(f"     - Data points: {len(prices)}")
                    print(f"     - Granularity: {hours:.1f} hours")
                
                # Rate limiting
                await asyncio.sleep(2.1)
                
        except Exception as e:
            print(f"   Chunk {i+1} error: {e}")
    
    print(f"\n   Total data points from chunked approach: {total_data_points}")
    print(f"   Expected hourly records for 180 days: {180 * 24}")
    print(f"   Coverage: {total_data_points / (180 * 24) * 100:.1f}%")
    
    await client.close()


async def main():
    """Run all tests"""
    # Check for API key
    if not os.environ.get('COINGECKO_API_KEY'):
        print("❌ COINGECKO_API_KEY not set")
        print("Please set: export COINGECKO_API_KEY=your_key")
        return
    
    # Run tests
    await test_ohlc_endpoint()
    await test_market_chart_endpoint()
    await test_market_chart_range_endpoint()
    
    print("\n" + "=" * 80)
    print("📊 Granularity Analysis Summary")
    print("=" * 80)
    print()
    print("CoinGecko API granularity by time range:")
    print("  • 1-2 days: 5-minute to hourly data")
    print("  • 3-30 days: hourly data")
    print("  • 31-90 days: 4-hour data")
    print("  • 91+ days: 4-day data (96 hours)")
    print()
    print("Recommendations:")
    print("  1. For 180-day corpus: Use chunked approach with 30-day windows")
    print("  2. Alternative: Use /market_chart/range with multiple requests")
    print("  3. Consider using /market_chart for price data and construct OHLC")


if __name__ == '__main__':
    asyncio.run(main())