"""
Database utilities for the RLTE system
Provides connection pooling and database operations for activity logging
"""

import asyncio
import logging
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
import asyncpg
from .config import get_config

logger = logging.getLogger(__name__)

# Global database pool
_pool: Optional[asyncpg.Pool] = None


class DatabaseError(Exception):
    """Custom database error"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection error"""
    pass


class DatabaseQueryError(DatabaseError):
    """Database query error"""
    pass


async def get_database_pool() -> asyncpg.Pool:
    """Get or create database connection pool with retry logic"""
    global _pool
    
    if _pool is None:
        config = get_config()
        db_config = config.database
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                _pool = await asyncpg.create_pool(
                    host=db_config.host,
                    port=db_config.port,
                    database=db_config.database,
                    user=db_config.username,
                    password=db_config.password,
                    min_size=1,
                    max_size=db_config.pool_size,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'rlte_activity_logger',
                    }
                )
                
                logger.info(f"Database pool created: {db_config.host}:{db_config.port}/{db_config.database}")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create database pool after {max_retries} attempts: {e}")
                    raise DatabaseConnectionError(f"Unable to connect to database: {e}")
                
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    return _pool


async def close_database_pool() -> None:
    """Close database connection pool"""
    global _pool
    
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("Database pool closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")
        finally:
            _pool = None


async def test_database_connection() -> bool:
    """Test database connectivity"""
    try:
        pool = await get_database_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


@asynccontextmanager
async def get_database_connection():
    """Get database connection context manager"""
    pool = await get_database_pool()
    async with pool.acquire() as conn:
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise DatabaseQueryError(f"Database operation failed: {e}")


async def execute_query(
    query: str,
    *args,
    fetch: str = "none"
) -> Any:
    """
    Execute database query with error handling
    
    Args:
        query: SQL query string
        *args: Query parameters
        fetch: Return type - "none", "one", "many", "val"
    
    Returns:
        Query result based on fetch type
    """
    async with get_database_connection() as conn:
        try:
            if fetch == "one":
                return await conn.fetchrow(query, *args)
            elif fetch == "many":
                return await conn.fetch(query, *args)
            elif fetch == "val":
                return await conn.fetchval(query, *args)
            else:
                return await conn.execute(query, *args)
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error: {e}")
            raise DatabaseQueryError(f"PostgreSQL error: {e}")
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise DatabaseQueryError(f"Unexpected database error: {e}")


async def execute_transaction(queries: List[tuple]) -> None:
    """
    Execute multiple queries in a transaction
    
    Args:
        queries: List of (query, *args) tuples
    """
    async with get_database_connection() as conn:
        async with conn.transaction():
            try:
                for query_data in queries:
                    if len(query_data) == 1:
                        await conn.execute(query_data[0])
                    else:
                        query, *args = query_data
                        await conn.execute(query, *args)
            except Exception as e:
                logger.error(f"Transaction failed: {e}")
                raise DatabaseQueryError(f"Transaction failed: {e}")


async def check_database_health() -> Dict[str, Any]:
    """Check database health and return status information"""
    try:
        async with get_database_connection() as conn:
            # Basic connectivity test
            basic_test = await conn.fetchval("SELECT 1")
            
            # Check database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            
            # Check active connections
            active_connections = await conn.fetchval("""
                SELECT count(*) FROM pg_stat_activity 
                WHERE state = 'active' AND datname = current_database()
            """)
            
            # Check table existence
            tables_exist = await conn.fetchval("""
                SELECT count(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('activity_logs', 'users', 'activity_summaries')
            """)
            
            return {
                "status": "healthy",
                "connectivity": basic_test == 1,
                "database_size": db_size,
                "active_connections": active_connections,
                "tables_exist": tables_exist == 3,
                "timestamp": "NOW()"
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "NOW()"
        }


async def get_database_stats() -> Dict[str, Any]:
    """Get database statistics for monitoring"""
    try:
        async with get_database_connection() as conn:
            # Activity logs count
            activity_count = await conn.fetchval(
                "SELECT count(*) FROM activity_logs"
            )
            
            # Recent activity count (last 24h)
            recent_count = await conn.fetchval(
                "SELECT count(*) FROM activity_logs WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            
            # Error count (last 24h)
            error_count = await conn.fetchval("""
                SELECT count(*) FROM activity_logs 
                WHERE severity IN ('error', 'critical', 'alert', 'emergency')
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            # User count
            user_count = await conn.fetchval(
                "SELECT count(*) FROM users"
            )
            
            # Active sessions
            active_sessions = await conn.fetchval(
                "SELECT count(*) FROM user_activity_sessions WHERE ended_at IS NULL"
            )
            
            return {
                "total_activities": activity_count,
                "recent_activities_24h": recent_count,
                "errors_24h": error_count,
                "total_users": user_count,
                "active_sessions": active_sessions
            }
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}