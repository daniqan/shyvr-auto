#!/usr/bin/env python3
"""
Test Collection Setup

Quick test script to verify that all components needed for direct corpus
collection are working properly before running the full collection.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components to test
try:
    from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
    from src.utils.database import get_database_connection
    from src.ml_analysis.market_data import CoinGeckoClient
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# GCP imports
try:
    from google.cloud import secretmanager
    print("✓ GCP Secret Manager available")
    GCP_AVAILABLE = True
except ImportError:
    print("✗ GCP Secret Manager not available (will use environment variables)")
    GCP_AVAILABLE = False


async def test_database_connection():
    """Test database connectivity"""
    print("\nTesting database connection...")
    try:
        async with get_database_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                print("✓ Database connection successful")
                
                # Test schema
                tables = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('crypto_ohlcv', 'training_corpus_versions')
                """)
                
                table_names = [row['table_name'] for row in tables]
                if 'crypto_ohlcv' in table_names:
                    print("✓ crypto_ohlcv table exists")
                else:
                    print("✗ crypto_ohlcv table missing")
                    
                if 'training_corpus_versions' in table_names:
                    print("✓ training_corpus_versions table exists")
                else:
                    print("✗ training_corpus_versions table missing")
                
                # Check existing data
                existing_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source = 'initial'
                """)
                print(f"  Existing initial corpus records: {existing_count}")
                
                return True
            else:
                print("✗ Database connection test failed")
                return False
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


async def test_api_connectivity():
    """Test API connectivity"""
    print("\nTesting API connectivity...")
    
    # Test CoinGecko
    try:
        api_key = os.getenv('COINGECKO_API_KEY')
        client = CoinGeckoClient(api_key=api_key, rate_limit=30)
        
        # Simple test call
        data = await client.get_market_data('bitcoin')
        if data and hasattr(data, 'price_usd'):
            print(f"✓ CoinGecko API working (BTC price: ${data.price_usd:,.2f})")
        else:
            print("✗ CoinGecko API returned invalid data")
            
        await client.close()
        return True
        
    except Exception as e:
        print(f"✗ CoinGecko API test failed: {e}")
        return False


def test_secrets():
    """Test secret loading"""
    print("\nTesting secret loading...")
    
    secrets_to_check = [
        'DATABASE_URL',
        'COINGECKO_API_KEY', 
        'LUNARCRUSH_API_KEY',
        'HELIUS_API_KEY'
    ]
    
    secrets_found = 0
    
    for secret_name in secrets_to_check:
        value = os.getenv(secret_name)
        if value:
            print(f"✓ {secret_name}: {'*' * 20}...")
            secrets_found += 1
        else:
            print(f"✗ {secret_name}: Not set")
    
    if GCP_AVAILABLE:
        print(f"  {secrets_found}/{len(secrets_to_check)} secrets found in environment")
        print("  Note: Will attempt to load missing secrets from GCP Secret Manager")
    else:
        print(f"  {secrets_found}/{len(secrets_to_check)} secrets found in environment")
        if secrets_found < 2:  # Need at least DATABASE_URL and COINGECKO_API_KEY
            print("  ⚠️  Insufficient secrets for collection")
            return False
    
    return True


async def test_collector_initialization():
    """Test InitialCorpusCollector initialization"""
    print("\nTesting collector initialization...")
    
    try:
        collector = InitialCorpusCollector(
            collection_days=1,  # Just 1 day for test
            rate_limit_delay=1.0
        )
        print("✓ InitialCorpusCollector created successfully")
        
        # Test that it has the required methods
        required_methods = [
            'collect_standardized_corpus',
            'validate_corpus_completeness',
            'create_corpus_snapshot'
        ]
        
        for method_name in required_methods:
            if hasattr(collector, method_name):
                print(f"✓ Method {method_name} available")
            else:
                print(f"✗ Method {method_name} missing")
        
        await collector.close()
        return True
        
    except Exception as e:
        print(f"✗ Collector initialization failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("Initial Corpus Collection Setup Test")
    print("=" * 50)
    
    # Basic setup
    logging.basicConfig(level=logging.ERROR)  # Suppress noise
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection()),
        ("Secret Loading", test_secrets()),
        ("Collector Initialization", test_collector_initialization()),
        ("API Connectivity", test_api_connectivity())
    ]
    
    results = []
    for test_name, test_coro in tests:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n✓ All tests passed! Ready for corpus collection.")
        print("\nTo run the collection:")
        print("  ./scripts/run_direct_collection.sh")
        print("  # or for a dry run:")
        print("  ./scripts/run_direct_collection.sh --dry-run")
    else:
        print(f"\n✗ {len(results) - passed} test(s) failed. Please fix issues before running collection.")
        
        if not results[0][1]:  # Database test failed
            print("\n💡 Database connection tips:")
            print("  - Check DATABASE_URL environment variable")
            print("  - Ensure CloudSQL instance is accessible")
            print("  - Verify database migrations are up to date")
            
        if not results[1][1]:  # Secrets test failed
            print("\n💡 Secrets tips:")
            print("  - Run 'gcloud auth login' for GCP secret access")
            print("  - Or set environment variables manually:")
            print("    export DATABASE_URL='postgresql://...'")
            print("    export COINGECKO_API_KEY='your-key'")


if __name__ == "__main__":
    asyncio.run(main())