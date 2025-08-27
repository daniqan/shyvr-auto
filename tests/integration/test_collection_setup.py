#!/usr/bin/env python3
"""
Test Collection Setup

Integration tests to verify that all components needed for corpus collection 
are working properly. These are real tests that check actual database connectivity,
API access, and collector initialization - not mocks.
"""

import os
import sys
import asyncio
import pytest
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
from src.utils.database import get_database_connection
from src.ml_analysis.market_data import CoinGeckoClient

# Check GCP availability
try:
    from google.cloud import secretmanager
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_connection():
    """Test actual database connectivity - not mocked"""
    try:
        async with get_database_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1, "Database connection test failed"
            
            # Test schema existence
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('crypto_ohlcv', 'training_corpus_versions')
            """)
            
            table_names = [row['table_name'] for row in tables]
            assert 'crypto_ohlcv' in table_names, "crypto_ohlcv table missing"
            assert 'training_corpus_versions' in table_names, "training_corpus_versions table missing"
            
            # Check existing data
            existing_count = await conn.fetchval("""
                SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source = 'initial'
            """)
            print(f"Existing initial corpus records: {existing_count}")
            
            return True
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_connectivity():
    """Test actual CoinGecko API connectivity - not mocked"""
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        pytest.skip("COINGECKO_API_KEY not set - skipping API test")
    
    client = CoinGeckoClient(api_key=api_key, rate_limit=30)
    
    try:
        # Make real API call
        data = await client.get_market_data('bitcoin')
        assert data is not None, "CoinGecko API returned None"
        assert hasattr(data, 'price_usd'), "CoinGecko API returned invalid data structure"
        assert data.price_usd > 0, f"Invalid BTC price: {data.price_usd}"
        
        print(f"✓ CoinGecko API working (BTC price: ${data.price_usd:,.2f})")
        return True
        
    except Exception as e:
        pytest.fail(f"CoinGecko API test failed: {e}")
    finally:
        await client.close()


@pytest.mark.integration
def test_secrets():
    """Test that required secrets are actually available"""
    secrets_to_check = [
        'DATABASE_URL',
        'COINGECKO_API_KEY', 
        'LUNARCRUSH_API_KEY',
        'HELIUS_API_KEY'
    ]
    
    secrets_found = {}
    required_secrets = ['DATABASE_URL', 'COINGECKO_API_KEY']
    
    for secret_name in secrets_to_check:
        value = os.getenv(secret_name)
        secrets_found[secret_name] = value is not None
        
        if secret_name in required_secrets and not value:
            pytest.fail(f"Required secret {secret_name} not set")
    
    # Count and report optional secrets
    found_count = sum(secrets_found.values())
    print(f"{found_count}/{len(secrets_to_check)} secrets found in environment")
    
    if GCP_AVAILABLE:
        print("GCP Secret Manager available for missing secrets")
    
    return True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_collector_initialization():
    """Test actual InitialCorpusCollector initialization - not mocked"""
    try:
        collector = InitialCorpusCollector(
            collection_days=1,  # Just 1 day for test
            rate_limit_delay=1.0
        )
        
        # Test that it has the required methods
        required_methods = [
            'collect_standardized_corpus',
            'validate_corpus_completeness',
            'create_corpus_snapshot'
        ]
        
        for method_name in required_methods:
            assert hasattr(collector, method_name), f"Method {method_name} missing"
            method = getattr(collector, method_name)
            assert callable(method), f"{method_name} is not callable"
        
        # Test attributes
        assert collector.collection_days == 1
        assert collector.rate_limit_delay == 1.0
        
        await collector.close()
        return True
        
    except Exception as e:
        pytest.fail(f"Collector initialization failed: {e}")


@pytest.mark.integration
class TestCollectionSetupSuite:
    """
    Complete collection setup test suite
    Run with: pytest tests/integration/test_collection_setup.py -v -m integration
    """
    
    @pytest.mark.asyncio
    async def test_full_setup_validation(self):
        """Run all setup validations in sequence"""
        results = {}
        
        # Test database
        try:
            await test_database_connection()
            results['database'] = 'PASS'
        except (pytest.skip.Exception, pytest.xfail.Exception):
            raise
        except Exception as e:
            results['database'] = f'FAIL: {e}'
        
        # Test secrets
        try:
            test_secrets()
            results['secrets'] = 'PASS'
        except (pytest.skip.Exception, pytest.xfail.Exception):
            raise
        except Exception as e:
            results['secrets'] = f'FAIL: {e}'
        
        # Test collector
        try:
            await test_collector_initialization()
            results['collector'] = 'PASS'
        except (pytest.skip.Exception, pytest.xfail.Exception):
            raise
        except Exception as e:
            results['collector'] = f'FAIL: {e}'
        
        # Test API
        try:
            await test_api_connectivity()
            results['api'] = 'PASS'
        except (pytest.skip.Exception, pytest.xfail.Exception):
            results['api'] = 'SKIPPED'
        except Exception as e:
            results['api'] = f'FAIL: {e}'
        
        # Report results
        print("\n" + "=" * 50)
        print("SETUP VALIDATION RESULTS")
        print("=" * 50)
        
        for component, status in results.items():
            print(f"{component}: {status}")
        
        failed = [k for k, v in results.items() if v.startswith('FAIL')]
        if failed:
            pytest.fail(f"Setup validation failed for: {', '.join(failed)}")


# For standalone execution (preserving original functionality)
async def main():
    """Run all tests standalone (original behavior)"""
    print("Initial Corpus Collection Setup Test")
    print("=" * 50)
    
    logging.basicConfig(level=logging.ERROR)
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection()),
        ("Secret Loading", test_secrets()),
        ("Collector Initialization", test_collector_initialization()),
        ("API Connectivity", test_api_connectivity())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                await test_coro
            else:
                test_coro
            results.append((test_name, True))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
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
    # Run standalone for backward compatibility
    asyncio.run(main())