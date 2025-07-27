#!/usr/bin/env python3
"""
Database initialization and migration runner for RLTE Activity Logging
Provides functions to initialize the database schema and run migrations
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import asyncpg
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Database initialization and migration runner"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config path"""
        self.config_path = config_path
        self.migration_dir = Path(__file__).parent / "migrations"
        
    async def create_database_if_not_exists(self) -> bool:
        """Create database if it doesn't exist"""
        try:
            config = get_config()
            db_config = config.database
            
            # Connect to postgres database to create our database
            conn = await asyncpg.connect(
                host=db_config.host,
                port=db_config.port,
                database="postgres",  # Connect to default postgres db
                user=db_config.username,
                password=db_config.password
            )
            
            try:
                # Check if database exists
                exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1",
                    db_config.database
                )
                
                if not exists:
                    # Create database
                    await conn.execute(f'CREATE DATABASE "{db_config.database}"')
                    logger.info(f"Created database: {db_config.database}")
                    return True
                else:
                    logger.info(f"Database already exists: {db_config.database}")
                    return False
                    
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def get_database_connection(self) -> asyncpg.Connection:
        """Get database connection"""
        config = get_config()
        db_config = config.database
        
        return await asyncpg.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.username,
            password=db_config.password
        )
    
    async def run_sql_file(self, conn: asyncpg.Connection, sql_file: Path) -> None:
        """Run SQL file against database"""
        logger.info(f"Running migration: {sql_file.name}")
        
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        try:
            await conn.execute(sql_content)
            logger.info(f"Successfully executed: {sql_file.name}")
        except Exception as e:
            logger.error(f"Failed to execute {sql_file.name}: {e}")
            raise
    
    async def get_applied_migrations(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of applied migrations"""
        try:
            # Create migrations tracking table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_name VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Get applied migrations
            rows = await conn.fetch("SELECT migration_name FROM schema_migrations ORDER BY migration_name")
            return [row['migration_name'] for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    async def mark_migration_applied(self, conn: asyncpg.Connection, migration_name: str) -> None:
        """Mark migration as applied"""
        await conn.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES ($1) ON CONFLICT DO NOTHING",
            migration_name
        )
        logger.info(f"Marked migration as applied: {migration_name}")
    
    async def run_migrations(self) -> None:
        """Run all pending migrations"""
        logger.info("Starting database migrations...")
        
        # Get migration files
        migration_files = sorted([
            f for f in self.migration_dir.glob("*.sql")
            if f.is_file()
        ])
        
        if not migration_files:
            logger.warning("No migration files found")
            return
        
        conn = await self.get_database_connection()
        try:
            applied_migrations = await self.get_applied_migrations(conn)
            
            for migration_file in migration_files:
                migration_name = migration_file.name
                
                if migration_name in applied_migrations:
                    logger.info(f"Migration already applied: {migration_name}")
                    continue
                
                # Run migration
                await self.run_sql_file(conn, migration_file)
                await self.mark_migration_applied(conn, migration_name)
                
            logger.info("All migrations completed successfully")
            
        finally:
            await conn.close()
    
    async def test_database_connection(self) -> bool:
        """Test database connectivity"""
        try:
            conn = await self.get_database_connection()
            try:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed: unexpected result")
                    return False
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def initialize_database(self) -> None:
        """Full database initialization"""
        logger.info("Starting database initialization...")
        
        # Create database if needed
        await self.create_database_if_not_exists()
        
        # Test connection
        if not await self.test_database_connection():
            raise RuntimeError("Database connection failed")
        
        # Run migrations
        await self.run_migrations()
        
        # Insert test user if in development
        config = get_config()
        if config.app.environment == "development":
            await self.create_test_data()
        
        logger.info("Database initialization completed successfully")
    
    async def create_test_data(self) -> None:
        """Create test data for development"""
        logger.info("Creating test data...")
        
        conn = await self.get_database_connection()
        try:
            # Insert test user
            await conn.execute("""
                INSERT INTO users (telegram_user_id, username, first_name, is_admin)
                VALUES (123456789, 'test_user', 'Test', true)
                ON CONFLICT (telegram_user_id) DO NOTHING
            """)
            
            logger.info("Test data created")
            
        finally:
            await conn.close()
    
    async def reset_database(self) -> None:
        """Reset database (development only)"""
        config = get_config()
        if config.app.environment == "production":
            raise RuntimeError("Cannot reset database in production")
        
        logger.warning("Resetting database...")
        
        conn = await self.get_database_connection()
        try:
            # Drop all tables
            await conn.execute("""
                DROP SCHEMA public CASCADE;
                CREATE SCHEMA public;
                GRANT ALL ON SCHEMA public TO rlte_user;
                GRANT ALL ON SCHEMA public TO public;
            """)
            
            logger.info("Database reset completed")
            
        finally:
            await conn.close()


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLTE Database Initialization")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--reset", action="store_true", help="Reset database (dev only)")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection")
    parser.add_argument("--migrate-only", action="store_true", help="Run migrations only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        initializer = DatabaseInitializer(args.config)
        
        if args.reset:
            await initializer.reset_database()
            await initializer.initialize_database()
        elif args.test_connection:
            success = await initializer.test_database_connection()
            sys.exit(0 if success else 1)
        elif args.migrate_only:
            await initializer.run_migrations()
        else:
            await initializer.initialize_database()
            
        logger.info("Database operations completed successfully")
        
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())