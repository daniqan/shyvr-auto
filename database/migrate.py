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
from typing import List, Tuple
from dotenv import load_dotenv

# Add src to path for config imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self, database_url: str):
        self.database_url = database_url
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
    try:
        # Get database URL from config or environment
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            try:
                config = get_config()
                database_url = config.database.url
            except Exception as e:
                logger.error(f"Failed to load database configuration: {e}")
                
        # Try loading from .env file if still not found
        if not database_url:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                database_url = os.getenv('DATABASE_URL')
                if database_url:
                    logger.info("Loaded DATABASE_URL from .env file")
                
        if not database_url:
            logger.error("DATABASE_URL not found in environment, config, or .env file")
            sys.exit(1)
            
        logger.info("🚀 Starting database migrations...")
        logger.info(f"Database: {database_url.split('@')[1] if '@' in database_url else 'localhost'}")
        
        # Run migrations
        runner = MigrationRunner(database_url)
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