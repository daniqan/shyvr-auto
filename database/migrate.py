#!/usr/bin/env python3
"""
Database migration runner for Shyvr RLTE
Runs database migrations in order and tracks progress
"""

import asyncio
import asyncpg
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.system_secrets import get_system_secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self, database_url: Optional[str] = None, use_local_proxy: Optional[bool] = None):
        """Initialize migration runner
        
        Args:
            database_url: Optional database URL. If not provided, uses SystemSecrets
            use_local_proxy: Whether to use local Cloud SQL proxy. If None, auto-detects
        """
        if database_url:
            self.database_url = database_url
        else:
            # Use SystemSecrets to get database configuration
            system_secrets = get_system_secrets()
            db_config = system_secrets.get_database_config(use_local_proxy=use_local_proxy)
            
            if not db_config.get('url'):
                raise ValueError("Could not get database configuration from SystemSecrets")
            
            self.database_url = db_config['url']
            logger.info(f"Using database: {db_config.get('host')}:{db_config.get('port')}/{db_config.get('database')}")
        
        self.migrations_dir = Path(__file__).parent / "migrations"
        
    async def create_migrations_table(self, conn):
        """Create migrations tracking table if it doesn't exist"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
    async def get_applied_migrations(self, conn) -> List[str]:
        """Get list of already applied migrations"""
        rows = await conn.fetch("SELECT filename FROM schema_migrations ORDER BY filename")
        return [row['filename'] for row in rows]
        
    async def get_migration_files(self) -> List[Path]:
        """Get sorted list of migration files"""
        if not self.migrations_dir.exists():
            logger.error(f"Migrations directory not found: {self.migrations_dir}")
            return []
            
        files = list(self.migrations_dir.glob("*.sql"))
        return sorted(files)
        
    async def run_migration(self, conn, migration_file: Path):
        """Run a single migration file"""
        logger.info(f"Running migration: {migration_file.name}")
        
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
                
            # Execute migration in a transaction
            async with conn.transaction():
                await conn.execute(migration_sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)",
                    migration_file.name
                )
                
            logger.info(f"✅ Migration completed: {migration_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {migration_file.name} - {e}")
            raise
            
    async def run_migrations(self) -> bool:
        """Run all pending migrations"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Create migrations table
            await self.create_migrations_table(conn)
            
            # Get migration files and applied migrations
            migration_files = await self.get_migration_files()
            applied_migrations = await self.get_applied_migrations(conn)
            
            if not migration_files:
                logger.warning("No migration files found")
                return True
                
            # Filter out already applied migrations
            pending_migrations = [
                f for f in migration_files 
                if f.name not in applied_migrations
            ]
            
            if not pending_migrations:
                logger.info("✅ All migrations are up to date")
                return True
                
            logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            # Run pending migrations
            for migration_file in pending_migrations:
                await self.run_migration(conn, migration_file)
                
            logger.info(f"✅ Successfully applied {len(pending_migrations)} migrations")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration process failed: {e}")
            return False
            
        finally:
            if 'conn' in locals():
                await conn.close()


async def main():
    """Main migration runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--drop-all', action='store_true', help='Drop all tables before running migrations')
    parser.add_argument('--use-local-proxy', action='store_true', help='Force use of local Cloud SQL proxy')
    parser.add_argument('--database-url', help='Override database URL')
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting database migrations...")
        
        # Initialize runner with SystemSecrets or provided URL
        runner = MigrationRunner(
            database_url=args.database_url,
            use_local_proxy=args.use_local_proxy if args.use_local_proxy else None
        )
        
        # Optionally drop all tables first
        if args.drop_all:
            logger.warning("⚠️  Dropping all tables...")
            conn = await asyncpg.connect(runner.database_url)
            
            # Get all table names (including schema_migrations when dropping all)
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            
            # Drop each table
            for table in tables:
                tablename = table['tablename']
                logger.info(f"  Dropping table: {tablename}")
                await conn.execute(f'DROP TABLE IF EXISTS {tablename} CASCADE')
            
            # Also drop custom types
            types = await conn.fetch("""
                SELECT typname FROM pg_type 
                WHERE typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                AND typtype = 'e'
            """)
            
            for typ in types:
                typename = typ['typname']
                logger.info(f"  Dropping type: {typename}")
                await conn.execute(f'DROP TYPE IF EXISTS {typename} CASCADE')
            
            await conn.close()
            logger.info("✅ All tables dropped")
        
        # Run migrations
        success = await runner.run_migrations()
        
        if success:
            logger.info("🎉 Database migrations completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Database migrations failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())