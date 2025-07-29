#!/usr/bin/env python3
"""
Production database migration runner with Cloud SQL Proxy support
Connects to production Cloud SQL instance and runs migrations safely
"""

import asyncio
import logging
import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional
import asyncpg
from google.cloud import secretmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class ProductionMigrationRunner:
    """Production migration runner with Cloud SQL Proxy"""
    
    def __init__(self):
        self.project_id = "shvyr-ai-bots"
        self.connection_name = "shvyr-ai-bots:us-central1:shyvr-rlte-db-prod"
        self.proxy_port = 5433
        self.proxy_process = None
        self.migration_dir = Path(__file__).parent.parent / "database" / "migrations"
        
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Secret Manager"""
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    
    def start_cloud_sql_proxy(self) -> bool:
        """Start Cloud SQL Proxy"""
        try:
            # Download proxy if not exists
            proxy_path = "/tmp/cloud_sql_proxy"
            if not os.path.exists(proxy_path):
                logger.info("Downloading Cloud SQL Proxy...")
                download_cmd = [
                    "curl", "-o", proxy_path,
                    "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.darwin.amd64"
                ]
                subprocess.run(download_cmd, check=True)
                os.chmod(proxy_path, 0o755)
            
            # Start proxy
            logger.info(f"Starting Cloud SQL Proxy on port {self.proxy_port}")
            proxy_cmd = [
                proxy_path,
                f"--port={self.proxy_port}",
                self.connection_name
            ]
            
            self.proxy_process = subprocess.Popen(
                proxy_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for proxy to start
            time.sleep(10)
            
            if self.proxy_process.poll() is not None:
                stdout, stderr = self.proxy_process.communicate()
                logger.error(f"Proxy failed to start: {stderr.decode()}")
                return False
            
            logger.info("Cloud SQL Proxy started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Cloud SQL Proxy: {e}")
            return False
    
    def stop_cloud_sql_proxy(self):
        """Stop Cloud SQL Proxy"""
        if self.proxy_process:
            logger.info("Stopping Cloud SQL Proxy")
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()
            self.proxy_process = None
    
    async def get_database_connection(self) -> asyncpg.Connection:
        """Get database connection through proxy"""
        db_password = self.get_secret("db-password-production")
        
        return await asyncpg.connect(
            host="localhost",
            port=self.proxy_port,
            database="shyvr_rlte_prod",
            user="rlte_prod_user",
            password=db_password
        )
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = await self.get_database_connection()
            try:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Connection test failed: unexpected result")
                    return False
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def run_sql_file(self, conn: asyncpg.Connection, sql_file: Path) -> None:
        """Run SQL migration file"""
        logger.info(f"Running migration: {sql_file.name}")
        
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        try:
            # Execute the entire file as a single transaction
            # This preserves BEGIN/COMMIT structure in migration files
            await conn.execute(sql_content)
            logger.info(f"Successfully executed: {sql_file.name}")
        except Exception as e:
            logger.error(f"Failed to execute {sql_file.name}: {e}")
            raise
    
    async def get_applied_migrations(self, conn: asyncpg.Connection) -> list:
        """Get list of applied migrations"""
        try:
            # Create migrations table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_name VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            rows = await conn.fetch("SELECT migration_name FROM schema_migrations ORDER BY migration_name")
            return [row['migration_name'] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    async def mark_migration_applied(self, conn: asyncpg.Connection, migration_name: str):
        """Mark migration as applied"""
        await conn.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES ($1) ON CONFLICT DO NOTHING",
            migration_name
        )
        logger.info(f"Marked migration as applied: {migration_name}")
    
    async def run_migrations(self) -> bool:
        """Run all pending migrations"""
        try:
            logger.info("Starting production database migrations...")
            
            # Get migration files
            migration_files = sorted([
                f for f in self.migration_dir.glob("*.sql")
                if f.is_file()
            ])
            
            if not migration_files:
                logger.warning("No migration files found")
                return True
            
            conn = await self.get_database_connection()
            try:
                applied_migrations = await self.get_applied_migrations(conn)
                
                migrations_run = 0
                for migration_file in migration_files:
                    migration_name = migration_file.name
                    
                    if migration_name in applied_migrations:
                        logger.info(f"Migration already applied: {migration_name}")
                        continue
                    
                    # Run migration
                    await self.run_sql_file(conn, migration_file)
                    await self.mark_migration_applied(conn, migration_name)
                    migrations_run += 1
                
                logger.info(f"Successfully ran {migrations_run} migrations")
                return True
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    async def validate_schema(self) -> bool:
        """Validate that all required tables exist"""
        try:
            conn = await self.get_database_connection()
            try:
                # Check for RL experience tables
                tables = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                """)
                
                table_names = [row['table_name'] for row in tables]
                required_tables = {'rl_experiences', 'rl_training_sessions', 'rl_performance_metrics'}
                
                if required_tables.issubset(set(table_names)):
                    logger.info("All required RL experience tables exist")
                    
                    # Check for sample data
                    experience_count = await conn.fetchval("SELECT COUNT(*) FROM rl_experiences")
                    session_count = await conn.fetchval("SELECT COUNT(*) FROM rl_training_sessions")
                    
                    logger.info(f"Database contains {experience_count} experiences and {session_count} sessions")
                    return True
                else:
                    missing = required_tables - set(table_names)
                    logger.error(f"Missing required tables: {missing}")
                    return False
                    
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method"""
        try:
            # Start Cloud SQL Proxy
            if not self.start_cloud_sql_proxy():
                return False
            
            # Set up signal handler for cleanup
            def cleanup_handler(signum, frame):
                logger.info("Received signal, cleaning up...")
                self.stop_cloud_sql_proxy()
                sys.exit(1)
            
            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)
            
            async def run_async():
                # Test connection
                if not await self.test_connection():
                    return False
                
                # Run migrations
                if not await self.run_migrations():
                    return False
                
                # Validate schema
                if not await self.validate_schema():
                    return False
                
                logger.info("Production migrations completed successfully")
                return True
            
            success = asyncio.run(run_async())
            return success
            
        except Exception as e:
            logger.error(f"Migration runner failed: {e}")
            return False
        finally:
            self.stop_cloud_sql_proxy()


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    runner = ProductionMigrationRunner()
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()