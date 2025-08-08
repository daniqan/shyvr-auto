#!/usr/bin/env python
"""
Check corpus features and identify issues with feature calculation
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.database import get_database_connection


async def check_features_status():
    """Check the status of features in the database"""
    
    print("=" * 80)
    print("📊 CORPUS FEATURES INVESTIGATION")
    print("=" * 80)
    print()
    
    async with get_database_connection() as conn:
        # 1. Check if crypto_features table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'crypto_features'
            )
        """)
        
        print("1️⃣  Feature Table Status:")
        print("-" * 40)
        
        if not table_exists:
            print("   ❌ crypto_features table does NOT exist!")
            print("   This explains why features are not being stored.")
            print()
            
            # Check what tables DO exist from migration 008
            corpus_tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND (
                    table_name LIKE '%corpus%' 
                    OR table_name LIKE '%ohlcv%'
                    OR table_name LIKE '%sentiment%'
                    OR table_name LIKE '%defi%'
                    OR table_name LIKE '%onchain%'
                )
                ORDER BY table_name
            """)
            
            print("   Tables that DO exist:")
            for table in corpus_tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
                print(f"     • {table['table_name']}: {count} records")
        else:
            # Table exists, check its structure and data
            print("   ✅ crypto_features table exists")
            
            # Get column info
            cols = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'crypto_features'
                ORDER BY ordinal_position
            """)
            
            print(f"   Columns: {len(cols)} total")
            for col in cols[:5]:
                print(f"     • {col['column_name']}: {col['data_type']}")
            if len(cols) > 5:
                print(f"     ... and {len(cols) - 5} more columns")
            
            # Check data
            total_records = await conn.fetchval("SELECT COUNT(*) FROM crypto_features")
            bitcoin_records = await conn.fetchval("""
                SELECT COUNT(*) FROM crypto_features 
                WHERE token_id = 'bitcoin'
            """)
            
            print(f"   Total records: {total_records}")
            print(f"   Bitcoin records: {bitcoin_records}")
        
        print()
        
        # 2. Check corpus validation error about 'data_source' column
        print("2️⃣  Data Source Column Investigation:")
        print("-" * 40)
        
        # Check which tables have data_source column
        tables_with_data_source = await conn.fetch("""
            SELECT DISTINCT table_name 
            FROM information_schema.columns 
            WHERE column_name = 'data_source' 
            AND table_schema = 'public'
            ORDER BY table_name
        """)
        
        if tables_with_data_source:
            print("   Tables WITH data_source column:")
            for table in tables_with_data_source:
                print(f"     • {table['table_name']}")
        else:
            print("   ❌ No tables have data_source column!")
        
        # Check if corpus validation is looking at wrong table
        validation_query_issue = await conn.fetch("""
            SELECT table_name, column_name
            FROM information_schema.columns 
            WHERE table_schema = 'public'
            AND table_name = 'training_corpus_versions'
            ORDER BY ordinal_position
        """)
        
        print()
        print("   training_corpus_versions columns:")
        for col in validation_query_issue[:10]:
            print(f"     • {col['column_name']}")
        
        print()
        
        # 3. Check feature calculation process
        print("3️⃣  Feature Calculation Process:")
        print("-" * 40)
        
        # Check if feature engineer is being called
        ohlcv_count = await conn.fetchval("""
            SELECT COUNT(*) FROM crypto_ohlcv 
            WHERE data_source = 'initial'
        """)
        
        print(f"   OHLCV records (source data): {ohlcv_count}")
        
        # Check if we're storing features elsewhere
        possible_feature_tables = [
            'crypto_technical_indicators',
            'market_features',
            'calculated_features',
            'feature_store'
        ]
        
        for table_name in possible_feature_tables:
            exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                )
            """)
            if exists:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                print(f"   Found {table_name}: {count} records")
        
        print()
        
        # 4. Check API data collection issues
        print("4️⃣  API Data Collection Status:")
        print("-" * 40)
        
        # Check sentiment data
        sentiment_count = await conn.fetchval("""
            SELECT COUNT(*) FROM market_sentiment 
            WHERE data_source = 'initial'
        """)
        print(f"   Market sentiment records: {sentiment_count}")
        
        # Check DeFi data
        defi_count = await conn.fetchval("""
            SELECT COUNT(*) FROM defi_metrics 
            WHERE data_source = 'initial'
        """)
        print(f"   DeFi metrics records: {defi_count}")
        
        # Check social data
        social_count = await conn.fetchval("""
            SELECT COUNT(*) FROM social_sentiment 
            WHERE data_source = 'initial'
        """)
        print(f"   Social sentiment records: {social_count}")
        
        # Check on-chain data
        onchain_count = await conn.fetchval("""
            SELECT COUNT(*) FROM onchain_metrics 
            WHERE data_source = 'initial'
        """)
        print(f"   On-chain metrics records: {onchain_count}")
        
        print()
        
        # 5. Recommendations
        print("5️⃣  Recommendations:")
        print("-" * 40)
        
        if not table_exists:
            print("   1. Create crypto_features table in migration")
            print("   2. Or modify feature storage to use existing tables")
            print("   3. Check InitialCorpusCollector._store_feature_data()")
        
        if defi_count == 0:
            print("   ⚠️  DeFiLlama API is failing - check API endpoint")
        
        print("   ⚠️  Feature validation query needs to be fixed")
        print("   ⚠️  Consider storing features inline with OHLCV data")
        
        print()
        print("=" * 80)
        print("Next steps:")
        print("1. Review migration 008 for crypto_features table")
        print("2. Check InitialCorpusCollector feature storage logic")
        print("3. Fix corpus validation query")
        print("4. Consider alternative DeFi data source")
        print("=" * 80)


async def check_feature_calculation():
    """Test feature calculation directly"""
    print()
    print("=" * 80)
    print("🧪 TESTING FEATURE CALCULATION")
    print("=" * 80)
    print()
    
    try:
        from src.ml_analysis.feature_engineer import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(50000, 60000, 100),
            'high': np.random.uniform(60000, 65000, 100),
            'low': np.random.uniform(45000, 50000, 100),
            'close': np.random.uniform(50000, 60000, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        print("Sample data created:")
        print(f"  Shape: {sample_data.shape}")
        print(f"  Columns: {list(sample_data.columns)}")
        print()
        
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Calculate features
        print("Calculating features...")
        features = fe.calculate_technical_indicators(sample_data, 'test_token')
        
        if features is not None and not features.empty:
            print("✅ Features calculated successfully!")
            print(f"  Shape: {features.shape}")
            print(f"  Feature columns: {len(features.columns)}")
            
            # Show some feature names
            feature_cols = [col for col in features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            print(f"  Example features: {feature_cols[:5]}")
            
            # Check for NaN values
            nan_counts = features.isna().sum()
            nan_features = nan_counts[nan_counts > 0]
            if len(nan_features) > 0:
                print(f"  ⚠️  Features with NaN: {len(nan_features)}")
            else:
                print("  ✅ No NaN values in features")
        else:
            print("❌ Feature calculation returned empty DataFrame")
            
    except Exception as e:
        print(f"❌ Error testing feature calculation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all checks"""
    await check_features_status()
    await check_feature_calculation()


if __name__ == '__main__':
    # Check environment
    required_vars = ['DATABASE_URL', 'SECRET_KEY']
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please run with: ./scripts/data_collection/collect_initial_corpus.sh")
        sys.exit(1)
    
    asyncio.run(main())