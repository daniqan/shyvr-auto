#!/usr/bin/env python
"""
Simple test of multi-granularity collection
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables
os.environ['SECRET_KEY'] = 'test_secret_key_for_corpus_collection_16chars'

print("Loading modules...")

try:
    from src.data_pipeline.multi_granularity_collector import MultiGranularityCollector
    print("Modules loaded successfully")
except Exception as e:
    print(f"Failed to load modules: {e}")
    sys.exit(1)

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        collector = MultiGranularityCollector(
            config_path="config/corpus_collection.yaml",
            collection_days=7,  # Just 7 days for testing
            rate_limit_delay=2.1
        )
        
        print(f"Tokens configured: {list(collector.tokens.keys())}")
        print(f"Timeframes configured: {[tf.value for tf in collector.timeframes.keys()]}")
        
        # Filter to just USDC and WETH
        test_tokens = ['USDC', 'WETH']
        collector.tokens = {
            symbol: config 
            for symbol, config in collector.tokens.items() 
            if symbol in test_tokens
        }
        
        # Enable only daily and hourly
        for tf_enum, tf_config in collector.timeframes.items():
            if tf_enum.value not in ['daily', 'hourly']:
                tf_config.enabled = False
        
        print(f"\nTest configuration:")
        print(f"  Tokens: {list(collector.tokens.keys())}")
        print(f"  Enabled timeframes: {[tf.value for tf, config in collector.timeframes.items() if config.enabled]}")
        
        # Calculate expected data
        total_expected = 0
        for token in collector.tokens.keys():
            for tf_enum, tf_config in collector.timeframes.items():
                if tf_config.enabled:
                    print(f"  {token} @ {tf_enum.value}: ~{tf_config.expected_candles} candles expected")
                    total_expected += tf_config.expected_candles
        
        print(f"\nTotal expected candles: {total_expected}")
        
        return collector
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_collection(collector):
    """Test actual collection with small sample"""
    print("\nTesting actual collection (small sample)...")
    
    try:
        # Modify to collect just 1 day for testing
        for tf_config in collector.timeframes.values():
            tf_config.days_back = 1
            tf_config.expected_candles = 24 if tf_config.interval == 'hour' else 1
        
        # Disable parallel collection for debugging
        collector.collection_strategy['parallel_collection'] = False
        
        print("Starting collection...")
        corpus_data = await collector.collect_multi_granularity_corpus()
        
        print(f"\nCollection results:")
        for key, df in corpus_data.items():
            print(f"  {key}: {len(df)} candles, {len(df.columns)} features")
        
        return corpus_data
        
    except Exception as e:
        print(f"Collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("Multi-Granularity Collector Test")
    print("="*50)
    
    # Test configuration
    collector = test_config()
    if not collector:
        return 1
    
    # Test collection
    # corpus_data = await test_collection(collector)
    # if not corpus_data:
    #     return 1
    
    print("\n" + "="*50)
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)