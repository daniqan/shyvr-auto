#!/usr/bin/env python3
"""
Collect only PEPE data to test the dual storage precision fix
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.multi_granularity_collector import MultiGranularityCollector
import structlog

logger = structlog.get_logger(__name__)

async def collect_pepe_only():
    """Collect and store only PEPE data"""
    
    print("=" * 60)
    print("Testing PEPE Collection with Dual Storage Precision")
    print("=" * 60)
    
    # Initialize collector
    collector = MultiGranularityCollector()
    
    # Override tokens to only collect PEPE
    collector.tokens = {
        'PEPE': collector.tokens['PEPE']
    }
    
    # Limit timeframes for testing
    enabled_timeframes = ['four_hour', 'hourly']
    for tf_enum, tf_config in collector.timeframes.items():
        if tf_enum.value not in enabled_timeframes:
            tf_config.enabled = False
    
    try:
        # Collect data
        print("\n1. Collecting PEPE data...")
        corpus_data = await collector.collect_multi_granularity_corpus()
        
        print(f"\n   ✓ Collected {len(corpus_data)} datasets:")
        for key, df in corpus_data.items():
            print(f"     - {key}: {len(df)} records")
            # Show sample prices to verify precision
            if len(df) > 0:
                sample_price = df['close'].iloc[-1]
                print(f"       Latest close price: {sample_price:.12f}")
        
        # Store to database
        print("\n2. Storing to database with DOUBLE PRECISION...")
        await collector.store_corpus_to_database(corpus_data)
        
        print("\n✅ Successfully stored PEPE data with full precision!")
        
        # Verify storage
        from src.utils.database import get_database_connection
        async with get_database_connection() as conn:
            result = await conn.fetch("""
                SELECT granularity, COUNT(*) as count, 
                       MIN(close) as min_price, MAX(close) as max_price,
                       AVG(price_precision) as avg_precision
                FROM crypto_ohlcv
                WHERE symbol = 'PEPE' AND data_source = 'initial'
                GROUP BY granularity
                ORDER BY granularity
            """)
            
            print("\n3. Verification from database:")
            for row in result:
                print(f"   {row['granularity']:10} : {row['count']:4} records")
                print(f"     Price range: {row['min_price']:.12f} - {row['max_price']:.12f}")
                print(f"     Avg precision: {row['avg_precision']:.1f} decimal places")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(collect_pepe_only())
    sys.exit(0 if success else 1)