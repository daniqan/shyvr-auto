#!/usr/bin/env python3
"""
Run database migration for dual storage precision
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.database import get_database_connection
import structlog

logger = structlog.get_logger(__name__)

async def run_migration():
    """Execute the dual storage precision migration"""
    
    migration_file = Path("migrations/006_dual_storage_precision.sql")
    
    if not migration_file.exists():
        print(f"Migration file not found: {migration_file}")
        return False
    
    print("Running migration: Dual Storage Precision Strategy")
    print("=" * 60)
    
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    async with get_database_connection() as conn:
        try:
            # Run migration in a transaction
            async with conn.transaction():
                # Execute the entire migration as one script
                print("Executing migration script...")
                await conn.execute(migration_sql)
                
                print("\n✅ Migration completed successfully!")
                
                # Verify the changes
                print("\nVerifying schema changes...")
                
                # Check OHLCV table
                ohlcv_cols = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'crypto_ohlcv' 
                    AND column_name IN ('open', 'high', 'low', 'close', 'volume', 'price_precision')
                    ORDER BY column_name
                """)
                
                print("\nOHLCV table columns:")
                for col in ohlcv_cols:
                    print(f"  {col['column_name']:20} : {col['data_type']}")
                
                # Check features table sample
                feature_cols = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'crypto_features' 
                    AND column_name IN ('rsi_14', 'macd', 'volume_price_ratio', 'volatility_ratio')
                    ORDER BY column_name
                """)
                
                print("\nFeatures table (sample columns):")
                for col in feature_cols:
                    print(f"  {col['column_name']:20} : {col['data_type']}")
                
                return True
                
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)