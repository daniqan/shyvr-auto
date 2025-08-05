#!/usr/bin/env python3
"""
Production Health Check
Runs comprehensive health checks on all system components including transformers
"""

import asyncio
import logging
import sys
from pathlib import Path
import asyncpg
import httpx
import json
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


async def check_api_health(base_url="http://localhost:8080"):
    """Check API health including transformer endpoints"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Check main health endpoint
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"API health check response: {json.dumps(health_data, indent=2)}")
            
            # Validate transformer health specifically
            if "transformers" not in health_data.get("components", {}):
                return {
                    "status": "unhealthy",
                    "error": "Transformers component missing from health endpoint"
                }
            
            transformer_health = health_data["components"]["transformers"]
            
            # Check transformer health requirements
            required_fields = ["status", "models", "total_memory_usage_mb", "models_loaded", "models_healthy"]
            for field in required_fields:
                if field not in transformer_health:
                    return {
                        "status": "unhealthy", 
                        "error": f"Missing required transformer field: {field}"
                    }
            
            # Check individual transformer models
            expected_models = ["itransformer", "patchtst", "timesmixer", "timesfm"]
            for model_name in expected_models:
                if model_name not in transformer_health["models"]:
                    logger.warning(f"Transformer model {model_name} not found in health response")
                else:
                    model_health = transformer_health["models"][model_name]
                    logger.info(f"Transformer {model_name} health: {model_health}")
            
            # Validate overall health
            overall_status = health_data.get("status", "unknown")
            if overall_status not in ["healthy", "degraded"]:
                return {
                    "status": "unhealthy",
                    "error": f"Invalid overall health status: {overall_status}"
                }
            
            return {
                "status": "healthy",
                "api_status": overall_status,
                "transformers_loaded": transformer_health["models_loaded"],
                "transformers_healthy": transformer_health["models_healthy"],
                "total_memory_mb": transformer_health["total_memory_usage_mb"]
            }
            
    except httpx.TimeoutException:
        logger.error("API health check timed out")
        return {"status": "unhealthy", "error": "API health check timeout"}
    except httpx.HTTPStatusError as e:
        logger.error(f"API health check HTTP error: {e}")
        return {"status": "unhealthy", "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


async def comprehensive_health_check():
    """Run all health checks"""
    logger.info("Starting comprehensive health check...")
    
    results = {
        "timestamp": str(asyncio.get_event_loop().time()),
        "database": await check_database_health(),
        "api": await check_api_health()
    }
    
    # Determine overall status
    all_healthy = all(
        result.get("status") == "healthy" 
        for result in results.values() 
        if isinstance(result, dict) and "status" in result
    )
    
    results["overall_status"] = "healthy" if all_healthy else "unhealthy"
    
    logger.info(f"Comprehensive health check completed: {results['overall_status']}")
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production health check")
    parser.add_argument("--database-only", action="store_true", help="Check database only")
    parser.add_argument("--api-only", action="store_true", help="Check API only")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API base URL")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.database_only:
        result = asyncio.run(check_database_health())
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "healthy" else 1)
    elif args.api_only:
        result = asyncio.run(check_api_health(args.api_url))
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "healthy" else 1)
    else:
        result = asyncio.run(comprehensive_health_check())
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["overall_status"] == "healthy" else 1)
