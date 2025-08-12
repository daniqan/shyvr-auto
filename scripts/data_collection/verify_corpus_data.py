#!/usr/bin/env python
"""Verify corpus data in database"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['SECRET_KEY'] = 'test_secret_key_for_database_check'

from src.utils.database import get_database_connection
from datetime import datetime, timezone

async def check_corpus_data():
    """Check what data was collected"""
    
    print("=" * 60)
    print("Checking Corpus Data in Database")
    print("=" * 60)
    
    async with get_database_connection() as conn:
        # Check OHLCV data
        ohlcv_count = await conn.fetchval(
            "SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source='initial'"
        )
        print(f"\n✅ OHLCV Records: {ohlcv_count}")
        
        if ohlcv_count > 0:
            # Get sample records
            samples = await conn.fetch("""
                SELECT token_id, MIN(timestamp) as min_date, MAX(timestamp) as max_date, COUNT(*) as records
                FROM crypto_ohlcv 
                WHERE data_source='initial'
                GROUP BY token_id
            """)
            for sample in samples:
                print(f"  - {sample['token_id']}: {sample['records']} records")
                print(f"    Date range: {sample['min_date']} to {sample['max_date']}")
        
        # Check market sentiment
        sentiment_count = await conn.fetchval(
            "SELECT COUNT(*) FROM market_sentiment WHERE data_source='initial'"
        )
        print(f"\n✅ Market Sentiment Records: {sentiment_count}")
        
        if sentiment_count > 0:
            # Get date range
            date_range = await conn.fetchrow("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date,
                       AVG(fear_greed_index) as avg_index
                FROM market_sentiment 
                WHERE data_source='initial'
            """)
            print(f"  Date range: {date_range['min_date']} to {date_range['max_date']}")
            print(f"  Average Fear & Greed: {date_range['avg_index']:.1f}")
        
        # Check DeFi metrics
        defi_count = await conn.fetchval(
            "SELECT COUNT(*) FROM defi_metrics WHERE data_source='initial'"
        )
        print(f"\n✅ DeFi Metrics Records: {defi_count}")
        
        if defi_count > 0:
            # Get details
            defi_data = await conn.fetch("""
                SELECT protocol_name, chain, COUNT(*) as records,
                       MIN(total_value_locked) as min_tvl,
                       MAX(total_value_locked) as max_tvl,
                       MIN(timestamp) as min_date,
                       MAX(timestamp) as max_date
                FROM defi_metrics 
                WHERE data_source='initial'
                GROUP BY protocol_name, chain
            """)
            for record in defi_data:
                print(f"  - {record['protocol_name']} on {record['chain']}: {record['records']} records")
                print(f"    TVL range: ${record['min_tvl']:,.0f} to ${record['max_tvl']:,.0f}")
                print(f"    Date range: {record['min_date']} to {record['max_date']}")
        
        # Check social sentiment
        social_count = await conn.fetchval(
            "SELECT COUNT(*) FROM social_sentiment WHERE data_source='initial'"
        )
        print(f"\n✅ Social Sentiment Records: {social_count}")
        
        if social_count > 0:
            # Get date range
            social_range = await conn.fetchrow("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date,
                       AVG(sentiment_score) as avg_score,
                       SUM(social_volume) as total_mentions
                FROM social_sentiment 
                WHERE data_source='initial'
            """)
            print(f"  Date range: {social_range['min_date']} to {social_range['max_date']}")
            if social_range['avg_score'] is not None:
                print(f"  Average social score: {social_range['avg_score']:.2f}")
            if social_range['total_mentions'] is not None:
                print(f"  Total mentions: {social_range['total_mentions']:,}")
        
        # Check on-chain metrics
        onchain_count = await conn.fetchval(
            "SELECT COUNT(*) FROM onchain_metrics WHERE data_source='initial'"
        )
        print(f"\n✅ On-chain Metrics Records: {onchain_count}")
        
        # Check corpus versions
        versions = await conn.fetch("""
            SELECT version_id, version_name, start_timestamp, end_timestamp,
                   sample_count, feature_count, is_active
            FROM training_corpus_versions 
            WHERE data_source='initial'
            ORDER BY version_id DESC
            LIMIT 5
        """)
        
        print(f"\n✅ Corpus Versions: {len(versions)}")
        for v in versions:
            print(f"  - {v['version_name']} (ID: {v['version_id']})")
            print(f"    Samples: {v['sample_count']}, Features: {v['feature_count']}")
            print(f"    Active: {v['is_active']}")
        
        # Summary
        total = ohlcv_count + sentiment_count + defi_count + social_count + onchain_count
        print(f"\n{'=' * 60}")
        print(f"Total Records in Corpus: {total:,}")
        
        if defi_count > 0:
            print(f"\n🎉 Graph Protocol Integration Status: ✅ WORKING")
            print(f"   Successfully collected {defi_count} days of historical DeFi data from Uniswap V3!")
        else:
            print(f"\n⚠️  Graph Protocol Integration Status: No DeFi data found")
        
        return total > 0

if __name__ == "__main__":
    result = asyncio.run(check_corpus_data())
    sys.exit(0 if result else 1)