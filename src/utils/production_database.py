#!/usr/bin/env python3
"""
Production Database Configuration and Connection Management
Handles Cloud SQL connections, connection pooling, and production-specific optimizations
"""

import logging
import os
import asyncio
import ssl
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool

from .config import get_config
from .system_secrets import get_system_secrets

logger = logging.getLogger(__name__)


class ProductionDatabaseManager:
    """Production database connection manager with Cloud SQL integration"""
    
    def __init__(self, use_local_proxy: Optional[bool] = None):
        """Initialize production database manager
        
        Args:
            use_local_proxy: Whether to use local Cloud SQL proxy. If None, auto-detects.
        """
        self.config = get_config()
        self.system_secrets = get_system_secrets()
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'shvyr-ai-bots')
        self.use_local_proxy = use_local_proxy
        self._connection_pool = None
        self._async_engine = None
        self._session_factory = None
        
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration from SystemSecrets"""
        return self.system_secrets.get_database_config(use_local_proxy=self.use_local_proxy)
    
    def get_production_connection_string(self) -> str:
        """Get production database connection string with Cloud SQL proxy"""
        try:
            # Get database config from SystemSecrets
            db_config = self.get_database_config()
            
            # Check if we're using local proxy or Cloud SQL socket
            if db_config['host'] in ['localhost', '127.0.0.1']:
                # Local proxy connection
                connection_string = (
                    f"postgresql+asyncpg://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                )
                logger.info(f"Using local proxy connection: {db_config['host']}:{db_config['port']}")
            else:
                # Cloud SQL socket connection
                connection_name = os.getenv('CLOUDSQL_CONNECTION_NAME', 
                                          'shvyr-ai-bots:us-central1:shyvr-rlte-db-prod')
                connection_string = (
                    f"postgresql+asyncpg://{db_config['username']}:{db_config['password']}"
                    f"@/{db_config['database']}?host=/cloudsql/{connection_name}"
                )
                logger.info(f"Using Cloud SQL socket connection: {connection_name}")
            
            return connection_string
            
        except Exception as e:
            logger.error(f"Failed to build production connection string: {e}")
            raise
    
    def get_sync_connection_string(self) -> str:
        """Get synchronous connection string for migrations"""
        try:
            # Get database config from SystemSecrets
            db_config = self.get_database_config()
            
            # Check if we're using local proxy or Cloud SQL socket
            if db_config['host'] in ['localhost', '127.0.0.1']:
                # Local proxy connection
                connection_string = (
                    f"postgresql://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                )
            else:
                # Cloud SQL socket connection
                connection_name = os.getenv('CLOUDSQL_CONNECTION_NAME',
                                          'shvyr-ai-bots:us-central1:shyvr-rlte-db-prod')
                connection_string = (
                    f"postgresql://{db_config['username']}:{db_config['password']}"
                    f"@/{db_config['database']}?host=/cloudsql/{connection_name}"
                )
            
            return connection_string
        except Exception as e:
            logger.error(f"Failed to build sync connection string: {e}")
            raise
    
    async def create_connection_pool(self) -> asyncpg.Pool:
        """Create asyncpg connection pool for production"""
        try:
            # Get database config from SystemSecrets
            db_config = self.get_database_config()
            
            # Determine connection parameters based on environment
            if db_config['host'] in ['localhost', '127.0.0.1']:
                # Local proxy connection
                host = db_config['host']
                port = int(db_config['port'])
                logger.info(f"Creating pool for local proxy: {host}:{port}")
            else:
                # Cloud SQL Unix socket connection
                connection_name = os.getenv('CLOUDSQL_CONNECTION_NAME',
                                          'shvyr-ai-bots:us-central1:shyvr-rlte-db-prod')
                host = f"/cloudsql/{connection_name}"
                port = 5432
                logger.info(f"Creating pool for Cloud SQL socket: {host}")
            
            # Create the connection pool
            pool = await asyncpg.create_pool(
                host=host,
                port=port,
                database=db_config['database'],
                user=db_config['username'],
                password=db_config['password'],
                min_size=self.config.database.pool_size,
                max_size=self.config.database.pool_size + self.config.database.max_overflow,
                command_timeout=60,
                # Note: Server settings removed as they require superuser privileges
                # Consider setting these at the database level if needed
            )
            
            logger.info(f"Created production connection pool with {self.config.database.pool_size} connections")
            return pool
            
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def create_async_engine(self):
        """Create SQLAlchemy async engine for ORM operations"""
        try:
            connection_string = self.get_production_connection_string()
            
            # Production-optimized engine configuration
            engine = create_async_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_timeout=30,
                pool_recycle=3600,  # Recycle connections every hour
                pool_pre_ping=True,  # Validate connections before use
                echo=False,  # Disable SQL logging in production
                future=True,
                # Note: connect_args removed as server_settings require superuser privileges
            )
            
            self._async_engine = engine
            self._session_factory = async_sessionmaker(
                engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            logger.info("Created production async engine and session factory")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create async engine: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper cleanup"""
        if not self._session_factory:
            self.create_async_engine()
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get raw database connection from pool"""
        if not self._connection_pool:
            self._connection_pool = await self.create_connection_pool()
        
        async with self._connection_pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                raise
    
    async def execute_query(self, query: str, *args, fetch_all: bool = True):
        """Execute query with connection pooling"""
        async with self.get_connection() as conn:
            if fetch_all:
                return await conn.fetch(query, *args)
            else:
                return await conn.fetchval(query, *args)
    
    async def execute_transaction(self, queries: list):
        """Execute multiple queries in a transaction"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                results = []
                for query, args in queries:
                    result = await conn.fetch(query, *args) if args else await conn.fetch(query)
                    results.append(result)
                return results
    
    async def test_connection(self) -> bool:
        """Test database connectivity and performance"""
        try:
            async with self.get_connection() as conn:
                # Basic connectivity test
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    return False
                
                # Test RL experience tables exist
                tables_exist = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                """)
                
                if tables_exist != 3:
                    logger.warning(f"RL experience tables missing: {tables_exist}/3 found")
                    return False
                
                # Performance test - should complete quickly
                import time
                start_time = time.time()
                await conn.fetchval("SELECT COUNT(*) FROM rl_experiences LIMIT 1")
                query_time = time.time() - start_time
                
                if query_time > 1.0:  # More than 1 second indicates performance issues
                    logger.warning(f"Slow query performance: {query_time}s")
                
                logger.info("Production database connection test successful")
                return True
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection pool and database information"""
        try:
            async with self.get_connection() as conn:
                # Database version and configuration
                db_version = await conn.fetchval("SELECT version()")
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                
                # Connection pool status
                pool_info = {
                    "size": self._connection_pool.get_size() if self._connection_pool else 0,
                    "idle_connections": self._connection_pool.get_idle_size() if self._connection_pool else 0,
                    "min_size": self.config.database.pool_size,
                    "max_size": self.config.database.pool_size + self.config.database.max_overflow,
                }
                
                # RL experience table statistics
                experience_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_experiences,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        MAX(created_at) as latest_experience,
                        MIN(created_at) as earliest_experience
                    FROM rl_experiences
                """)
                
                return {
                    "database_version": db_version,
                    "database_size": db_size,
                    "connection_pool": pool_info,
                    "experience_statistics": dict(experience_stats),
                    "environment": self.config.app.environment,
                    "connection_status": "healthy"
                }
                
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {
                "connection_status": "error",
                "error": str(e)
            }
    
    async def close(self):
        """Close all connections and cleanup"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("Closed production connection pool")
        
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Disposed async engine")


# Global production database manager instance
_prod_db_manager = None

def get_production_db_manager() -> ProductionDatabaseManager:
    """Get global production database manager instance"""
    global _prod_db_manager
    if _prod_db_manager is None:
        _prod_db_manager = ProductionDatabaseManager()
    return _prod_db_manager


async def test_production_database():
    """Test production database connectivity and performance"""
    manager = get_production_db_manager()
    
    # Test basic connectivity
    if not await manager.test_connection():
        raise RuntimeError("Production database connection test failed")
    
    # Get detailed connection information
    info = await manager.get_connection_info()
    logger.info(f"Production database info: {info}")
    
    return info


# Context manager for production database operations
@asynccontextmanager
async def production_database():
    """Context manager for production database operations"""
    manager = get_production_db_manager()
    try:
        yield manager
    finally:
        # Cleanup is handled by the manager's connection pooling
        pass