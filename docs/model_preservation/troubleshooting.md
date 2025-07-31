# Model Preservation System - Troubleshooting Guide

## Table of Contents
- [Common Issues](#common-issues)
- [Error Messages](#error-messages)
- [Performance Issues](#performance-issues)
- [Storage Problems](#storage-problems)
- [Database Issues](#database-issues)
- [Cache Problems](#cache-problems)
- [API Errors](#api-errors)
- [Debugging Techniques](#debugging-techniques)
- [FAQ](#faq)

## Common Issues

### Model Loading Failures

**Issue**: Models fail to load with various error messages

**Symptoms**:
- `FileNotFoundError` when loading models
- `KeyError` for missing model metadata
- Models appear in database but fail to load

**Diagnosis**:
```python
# Check if model exists in database
from src.model_preservation.manager import ModelPreservationManager
manager = ModelPreservationManager()

# List all models
models = await manager.list_models()
print(f"Found {len(models)} models")

# Check specific model
try:
    model = await manager.load_model("dqn_agent", "1.2.3")
    print("Model loaded successfully")
except Exception as e:
    print(f"Load failed: {e}")
```

**Solutions**:
1. **Check GCS connectivity**:
   ```bash
   # Test GCS access
   gsutil ls gs://your-bucket-name/models/
   ```

2. **Verify database connection**:
   ```python
   from src.utils.database import DatabaseManager
   db = DatabaseManager()
   await db.check_connection()
   ```

3. **Clear cache and retry**:
   ```python
   await manager.clear_cache("dqn_agent")
   ```

### Version Conflicts

**Issue**: Version numbering conflicts or inconsistencies

**Symptoms**:
- Duplicate version numbers
- Missing versions in sequence
- Version rollback fails

**Solutions**:
1. **Check version history**:
   ```python
   history = await manager.get_model_history("dqn_agent")
   for version in history:
       print(f"{version['version']}: {version['created_at']}")
   ```

2. **Force version cleanup**:
   ```python
   await manager.cleanup_orphaned_versions("dqn_agent")
   ```

3. **Reset version sequence**:
   ```python
   await manager.reset_version_counter("dqn_agent")
   ```

### Authentication Errors

**Issue**: API authentication failures

**Symptoms**:
- 401 Unauthorized responses
- Token validation errors
- Permission denied errors

**Solutions**:
1. **Check API key validity**:
   ```bash
   curl -H "Authorization: Bearer YOUR_KEY" \
        https://your-domain.com/api/preservation/health
   ```

2. **Verify user permissions**:
   ```python
   from src.dashboard.auth import get_user_permissions
   permissions = get_user_permissions(user_id)
   print(f"User permissions: {permissions}")
   ```

3. **Refresh authentication tokens**:
   ```python
   from src.dashboard.auth import refresh_token
   new_token = refresh_token(current_token)
   ```

## Error Messages

### "Preservation system not initialized"

**Error Code**: `HTTP 503 Service Unavailable`

**Cause**: The ModelPreservationManager is not properly initialized

**Solution**:
```python
# Check initialization in main.py
from src.model_preservation.manager import ModelPreservationManager

# Ensure proper initialization
manager = ModelPreservationManager()
await manager.initialize()

# Set in API
from src.model_preservation.api import set_preservation_manager
set_preservation_manager(manager)
```

### "Model {type} version {version} not found"

**Error Code**: `HTTP 404 Not Found`

**Cause**: Requested model version doesn't exist

**Diagnosis**:
```python
# Check available versions
versions = await manager.get_model_versions("dqn_agent")
print(f"Available versions: {versions}")

# Check in specific mode
versions = await manager.get_model_versions("dqn_agent", mode="simulation")
```

**Solution**:
- Use `list_models` API to find available versions
- Check if model was saved in different mode/branch
- Verify model wasn't accidentally deleted

### "Failed to connect to Google Cloud Storage"

**Error Code**: Various GCS-related errors

**Cause**: GCS connectivity or permission issues

**Diagnosis**:
```bash
# Check GCS configuration
echo $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth list
gcloud config list project

# Test bucket access
gsutil ls gs://your-bucket-name/
```

**Solutions**:
1. **Verify service account**:
   ```bash
   gcloud auth activate-service-account --key-file=path/to/key.json
   ```

2. **Check bucket permissions**:
   ```bash
   gsutil iam get gs://your-bucket-name/
   ```

3. **Test connectivity**:
   ```python
   from google.cloud import storage
   client = storage.Client()
   bucket = client.bucket("your-bucket-name")
   print(f"Bucket exists: {bucket.exists()}")
   ```

### "Database connection failed"

**Error Code**: Database connectivity errors

**Cause**: PostgreSQL connection issues

**Diagnosis**:
```bash
# Test database connection
pg_isready -h localhost -p 5432 -U your_user -d shyvr_rlte

# Check if database exists
psql -h localhost -U your_user -l | grep shyvr_rlte
```

**Solutions**:
1. **Verify connection parameters**:
   ```bash
   # Check environment variables
   echo $POSTGRES_HOST
   echo $POSTGRES_PORT
   echo $POSTGRES_DB
   echo $POSTGRES_USER
   ```

2. **Test direct connection**:
   ```bash
   psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB
   ```

3. **Check migration status**:
   ```bash
   cd database
   python migrate.py --status
   ```

## Performance Issues

### Slow Model Loading

**Symptoms**:
- High latency for model retrieval
- Timeouts during model loading
- High memory usage

**Diagnosis**:
```python
import time
start_time = time.time()
model = await manager.load_model("dqn_agent", "1.2.3")
load_time = time.time() - start_time
print(f"Load time: {load_time:.2f} seconds")

# Check model size
import sys
model_size = sys.getsizeof(model) / (1024 * 1024)  # MB
print(f"Model size in memory: {model_size:.2f} MB")
```

**Solutions**:
1. **Enable compression**:
   ```python
   # In configuration
   config = {
       "compression": True,
       "compression_level": 6
   }
   ```

2. **Increase cache size**:
   ```python
   cache_config = {
       "memory_cache_size": 500,  # MB
       "disk_cache_size": 2000,   # MB
   }
   ```

3. **Use async loading**:
   ```python
   # Load multiple models concurrently
   import asyncio
   models = await asyncio.gather(
       manager.load_model("dqn_agent", "1.2.3"),
       manager.load_model("lstm_predictor", "2.1.0")
   )
   ```

### High Memory Usage

**Symptoms**:
- Gradual memory increase
- Out of memory errors
- Cache bloat

**Solutions**:
1. **Monitor cache usage**:
   ```python
   cache_stats = await manager.get_cache_stats()
   print(f"Cache usage: {cache_stats['memory_usage_mb']} MB")
   ```

2. **Implement cache cleanup**:
   ```python
   # Clear old cache entries
   await manager.cleanup_cache(max_age_hours=24)
   
   # Clear specific model cache
   await manager.clear_cache("dqn_agent")
   ```

3. **Configure cache limits**:
   ```python
   # Set cache size limits
   await manager.configure_cache(
       max_memory_mb=1000,
       max_entries=100,
       ttl_seconds=3600
   )
   ```

## Storage Problems

### GCS Upload Failures

**Symptoms**:
- Model save operations fail
- Partial uploads
- Quota exceeded errors

**Diagnosis**:
```python
# Check GCS quota and usage
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("your-bucket-name")

# List recent uploads
blobs = list(bucket.list_blobs(prefix="models/", max_results=10))
for blob in blobs:
    print(f"{blob.name}: {blob.size} bytes, {blob.time_created}")
```

**Solutions**:
1. **Check storage quota**:
   ```bash
   gsutil du -s gs://your-bucket-name/
   ```

2. **Implement retry logic**:
   ```python
   from google.api_core import retry
   
   @retry.Retry()
   async def upload_with_retry(blob, data):
       return blob.upload_from_string(data)
   ```

3. **Use resumable uploads**:
   ```python
   # For large models
   blob.chunk_size = 256 * 1024  # 256KB chunks
   blob.upload_from_file(file, checksum="md5")
   ```

### Storage Corruption

**Symptoms**:
- Checksum mismatches
- Partial model data
- Load failures after successful save

**Diagnosis**:
```python
# Verify model integrity
checksum = await manager.verify_model_integrity("dqn_agent", "1.2.3")
if checksum['valid']:
    print("Model integrity OK")
else:
    print(f"Corruption detected: {checksum['error']}")
```

**Solutions**:
1. **Re-upload corrupted models**:
   ```python
   await manager.repair_model("dqn_agent", "1.2.3")
   ```

2. **Enable integrity checks**:
   ```python
   config = {
       "verify_checksums": True,
       "use_redundant_storage": True
   }
   ```

## Database Issues

### Migration Failures

**Symptoms**:
- Database schema mismatch errors
- Missing tables or columns
- Migration rollback failures

**Diagnosis**:
```bash
# Check migration status
cd database
python migrate.py --status

# Check schema version
psql -d shyvr_rlte -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
```

**Solutions**:
1. **Run pending migrations**:
   ```bash
   python migrate.py --up
   ```

2. **Rollback problematic migration**:
   ```bash
   python migrate.py --down --target=004
   ```

3. **Reset database** (destructive):
   ```bash
   python migrate.py --reset
   python migrate.py --up
   ```

### Connection Pool Exhaustion

**Symptoms**:
- "Too many connections" errors
- API timeouts
- Database deadlocks

**Solutions**:
1. **Increase connection pool size**:
   ```python
   # In database configuration
   pool_config = {
       "min_connections": 5,
       "max_connections": 20,
       "connection_timeout": 30
   }
   ```

2. **Monitor active connections**:
   ```sql
   -- Check active connections
   SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
   
   -- Kill long-running queries
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
   WHERE state = 'active' AND query_start < now() - interval '5 minutes';
   ```

## Cache Problems

### Cache Inconsistency

**Symptoms**:
- Stale model data
- Cache misses for recently saved models
- Inconsistent model metadata

**Solutions**:
1. **Force cache refresh**:
   ```python
   await manager.refresh_cache("dqn_agent", "1.2.3")
   ```

2. **Enable cache validation**:
   ```python
   config = {
       "validate_cache": True,
       "cache_ttl": 1800  # 30 minutes
   }
   ```

3. **Clear all caches**:
   ```python
   await manager.clear_all_caches()
   ```

## API Errors

### Rate Limiting

**Symptoms**:
- HTTP 429 Too Many Requests
- API slowdowns
- Throttling messages

**Solutions**:
1. **Implement exponential backoff**:
   ```python
   import time
   import random
   
   def api_call_with_backoff(func, max_retries=5):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               wait_time = 2 ** attempt + random.uniform(0, 1)
               time.sleep(wait_time)
       raise Exception("Max retries exceeded")
   ```

2. **Use batch operations**:
   ```python
   # Instead of multiple individual calls
   models = await manager.load_models_batch([
       ("dqn_agent", "1.2.3"),
       ("lstm_predictor", "2.1.0")
   ])
   ```

### Authentication Failures

**Solutions**:
1. **Check token expiration**:
   ```python
   import jwt
   
   def check_token_expiry(token):
       try:
           decoded = jwt.decode(token, verify=False)
           exp = decoded.get('exp', 0)
           return exp > time.time()
       except:
           return False
   ```

2. **Implement token refresh**:
   ```python
   if not check_token_expiry(token):
       token = refresh_token(token)
   ```

## Debugging Techniques

### Enable Debug Logging

```python
import logging
import structlog

# Configure structured logging
logging.basicConfig(level=logging.DEBUG)
logger = structlog.get_logger()

# Enable preservation system debug logging
from src.model_preservation import manager
manager.logger.setLevel(logging.DEBUG)
```

### Use Performance Profiling

```python
import cProfile
import pstats

# Profile model loading
def profile_model_loading():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your model loading code
    model = await manager.load_model("dqn_agent", "1.2.3")
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions

profile_model_loading()
```

### Monitor System Resources

```bash
# Monitor system resources during operations
htop

# Monitor disk I/O
iotop

# Monitor network
nethogs

# Check database performance
# Connect to PostgreSQL and run:
# SELECT * FROM pg_stat_activity;
# SELECT * FROM pg_stat_user_tables;
```

### Check System Health

```python
# Comprehensive health check
async def system_health_check():
    health = await manager.health_check()
    
    # Check each component
    checks = {
        'Database': health['database_connected'],
        'GCS': health['gcs_connected'],
        'Cache': health['cache_operational'],
        'Models': health['total_models'] > 0
    }
    
    for component, status in checks.items():
        print(f"{component}: {'✓' if status else '✗'}")
    
    return all(checks.values())

# Run health check
healthy = await system_health_check()
print(f"System healthy: {healthy}")
```

## FAQ

### Q: Why are my models taking so long to save?

**A**: Large models can take time to upload to GCS. Consider:
- Enabling compression
- Using batch operations
- Checking network bandwidth
- Implementing async saves

### Q: Can I recover a deleted model?

**A**: If you have backups enabled:
```python
# Check available backups
backups = await manager.list_backups("dqn_agent", "1.2.3")

# Restore from backup
if backups:
    await manager.restore_from_backup("dqn_agent", "1.2.3", backups[0]['backup_id'])
```

### Q: How do I migrate models between environments?

**A**: Use the export/import functionality:
```python
# Export model
export_data = await manager.export_model("dqn_agent", "1.2.3")

# Import in new environment
await manager.import_model(export_data)
```

### Q: What's the maximum model size supported?

**A**: The system supports models up to 5GB by default. For larger models:
- Enable streaming uploads
- Use model sharding
- Contact admin for quota increases

### Q: How do I set up monitoring and alerts?

**A**: Configure monitoring in your deployment:
```python
# Enable monitoring
from src.model_preservation.monitoring import PreservationMonitor

monitor = PreservationMonitor()
await monitor.enable_alerts(
    email="admin@yourcompany.com",
    thresholds={
        'storage_usage_gb': 100,
        'failed_operations_per_hour': 10,
        'avg_response_time_ms': 5000
    }
)
```

### Q: Can I use the system offline?

**A**: Limited offline functionality is available:
- Cache provides offline access to recently used models
- Database operations work without GCS
- New model saves require GCS connectivity

### Q: How do I backup the entire system?

**A**: Implement comprehensive backups:
```bash
# Database backup
pg_dump shyvr_rlte > backup.sql

# GCS backup
gsutil -m cp -r gs://your-bucket-name/models/ ./gcs-backup/

# Configuration backup
cp -r config/ ./config-backup/
```

---

For additional support, check the [User Guide](user_guide.md) and [API Reference](api_reference.md), or contact the development team.