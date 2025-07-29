#!/usr/bin/env python3
"""
Production Database Health Check
Runs periodic health checks on the RL experience database
"""

import asyncio
import logging
import sys
from pathlib import Path
import asyncpg
from google.cloud import secretmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

async def check_database_health():
    """Perform comprehensive database health check"""
    try:
        # Get credentials
        client = secretmanager.SecretManagerServiceClient()
        password = client.access_secret_version(
            request={"name": "projects/shvyr-ai-bots/secrets/db-password-production/versions/latest"}
        ).payload.data.decode("UTF-8")
        
        # Connect to database
        conn = await asyncpg.connect(
            host="localhost", 
            port=5433,
            database="shyvr_rlte_prod",
            user="rlte_prod_user", 
            password=password
        )
        
        try:
            # Basic connectivity
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            
            # Check table existence
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
            """)
            assert len(tables) == 3
            
            # Check data integrity
            experience_count = await conn.fetchval("SELECT COUNT(*) FROM rl_experiences")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM rl_training_sessions")
            
            # Performance test
            import time
            start = time.time()
            await conn.fetchval("SELECT COUNT(*) FROM rl_experiences LIMIT 1")
            query_time = time.time() - start
            
            health_status = {
                "status": "healthy",
                "experience_count": experience_count,
                "session_count": session_count,
                "query_time_ms": query_time * 1000,
                "tables_exist": len(tables) == 3
            }
            
            logger.info(f"Database health check passed: {health_status}")
            return health_status
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(check_database_health())
    sys.exit(0 if result["status"] == "healthy" else 1)
