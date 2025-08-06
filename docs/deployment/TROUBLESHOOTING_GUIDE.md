# Comprehensive Troubleshooting Guide

## Table of Contents
1. [Overview](#overview)
2. [System Health Assessment](#system-health-assessment)
3. [Transformer Model Issues](#transformer-model-issues)
4. [Memory and Resource Problems](#memory-and-resource-problems)
5. [Performance Issues](#performance-issues)
6. [Deployment Problems](#deployment-problems)
7. [Database Connectivity Issues](#database-connectivity-issues)
8. [API and External Service Issues](#api-and-external-service-issues)
9. [Emergency Procedures](#emergency-procedures)
10. [Diagnostic Tools and Commands](#diagnostic-tools-and-commands)
11. [Common Error Codes](#common-error-codes)

## Overview

This troubleshooting guide provides systematic approaches to diagnosing and resolving issues in the transformer-enabled RLTE system, covering production environment problems and solutions.

### Troubleshooting Philosophy
1. **Systematic Approach**: Follow structured diagnostic procedures
2. **Evidence-Based**: Collect data before making changes
3. **Minimal Impact**: Prefer non-disruptive solutions
4. **Documentation**: Record findings and solutions
5. **Prevention**: Implement fixes that prevent recurrence

## System Health Assessment

### Quick Health Check
```bash
# System health check
SERVICE_URL="https://your-service-url"
echo "=== RLTE Health Check ==="

# Service status
curl -f -s "${SERVICE_URL}/health" && echo "✅ Service OK" || echo "❌ Service Down"

# Database status  
curl -f -s "${SERVICE_URL}/health/database" && echo "✅ Database OK" || echo "❌ Database Issues"

# Transformer models
curl -f -s "${SERVICE_URL}/health/transformers" && echo "✅ Models OK" || echo "❌ Model Issues"
```

## Transformer Model Issues

### Model Loading Problems
**Symptoms**: Models fail to load, timeout errors, memory allocation failures

**Diagnostic Steps**:
```bash
# Check model status
curl -s "${SERVICE_URL}/debug/models/status" | jq '.'

# Check memory usage
free -h

# Verify model cache
ls -la /app/models/cache/
```

**Solutions**:
```bash
# Clear model cache
curl -X POST "${SERVICE_URL}/admin/cache/clear" -H "Authorization: Bearer $TOKEN"

# Increase memory temporarily
gcloud run services update shyvr-rlte --memory=12Gi --region=us-central1

# Restart service
gcloud run services replace service.yaml --region=us-central1
```

### Model Inference Issues
**Symptoms**: Slow inference (>1000ms), timeouts, inconsistent results

**Diagnostic**:
```bash
# Performance benchmark
curl -X POST "${SERVICE_URL}/debug/performance/benchmark" \
    -H "Content-Type: application/json" \
    -d '{"model_type": "itransformer"}'

# Monitor inference latency
curl -s "${SERVICE_URL}/metrics" | grep transformer_inference_latency
```

**Solutions**:
```python
# Optimize inference
torch.set_num_threads(6)
torch.set_grad_enabled(False)
model = torch.compile(model, mode="reduce-overhead")
```

## Memory and Resource Problems

### Memory Exhaustion
**Symptoms**: OOM kills, service restarts, slow performance

**Immediate Response**:
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Emergency memory increase
gcloud run services update shyvr-rlte --memory=16Gi --region=us-central1

# Memory analysis
curl -s "${SERVICE_URL}/debug/memory/breakdown" | jq '.'
```

**Solutions**:
```python
# Memory optimization
def optimize_memory():
    # Clear caches
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Use model sharding for large models
    model = load_model_with_sharding(max_memory_gb=6)
    
    # Implement cache eviction
    cache_manager.evict_lru_items()
```

## Performance Issues

### High Latency
**Symptoms**: API responses >100ms, model inference >1000ms, slow database queries

**Analysis**:
```bash
# End-to-end latency measurement
curl -w "@curl-format.txt" -s "${SERVICE_URL}/api/predict"

# Database performance
curl -s "${SERVICE_URL}/debug/database/slow-queries" | jq '.'

# Cache performance
curl -s "${SERVICE_URL}/debug/cache/stats" | jq '.'
```

**Optimization**:
```sql
-- Add database indexes
CREATE INDEX CONCURRENTLY idx_trades_symbol_timestamp 
ON trades(symbol, created_at DESC);

-- Update statistics
ANALYZE trades;
```

```python
# Implement batching
class RequestBatcher:
    def __init__(self, batch_size=10, max_wait_ms=50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        
    async def process_batch(self, requests):
        return await model.predict_batch(requests)
```

## Deployment Problems

### Build Failures
**Common Issues**:
```bash
# Docker build timeout - increase timeout
gcloud builds submit --timeout=3600s --machine-type=e2-highcpu-16

# Dependency conflicts - clear cache
gcloud builds submit --no-cache

# Model download failures - add retry logic
```

### Blue-Green Deployment Issues
```bash
# Check deployment status
gcloud run revisions list --service=shyvr-rlte --region=us-central1

# Emergency rollback
PREV_REVISION=$(gcloud run revisions list --service=shyvr-rlte --region=us-central1 \
    --format="value(metadata.name)" --limit=2 | tail -1)
gcloud run services update-traffic shyvr-rlte --to-revisions=$PREV_REVISION=100 --region=us-central1

# Gradual traffic migration
gcloud run services update-traffic shyvr-rlte --to-revisions=new=25,old=75 --region=us-central1
```

## Database Connectivity Issues

### Connection Problems
**Symptoms**: Connection refused, too many connections, timeouts

**Diagnostic**:
```bash
# Test connectivity
gcloud sql connect shyvr-rlte-db --user=postgres

# Check connection pool
curl -s "${SERVICE_URL}/debug/database/pool-status" | jq '.'

# Monitor connections
psql -c "SELECT count(*), max_conn FROM pg_stat_activity 
         CROSS JOIN (SELECT setting::int as max_conn FROM pg_settings 
         WHERE name = 'max_connections') t;"
```

**Solutions**:
```python
# Optimize connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Connection leak detection
@track_db_connections
async def database_operation():
    # Database operations
    pass
```

## API and External Service Issues

### External API Failures
**Issues**: Rate limits, auth failures, timeouts

**Resilience Patterns**:
```python
# Circuit breaker
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = "CLOSED"
    
    async def call(self, func, *args):
        if self.state == "OPEN":
            raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            return result
        except Exception:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise

# Retry with backoff
async def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)
```

## Emergency Procedures

### System-Wide Outage Response

#### Immediate Actions (0-5 minutes)
```bash
#!/bin/bash
# Emergency response script
echo "=== EMERGENCY RESPONSE ==="

# Check service status
gcloud run services list --filter="shyvr-rlte"

# Immediate rollback if recent deployment
PREV_REV=$(gcloud run revisions list --service=shyvr-rlte --region=us-central1 \
    --format="value(metadata.name)" --limit=2 | tail -1)
gcloud run services update-traffic shyvr-rlte --to-revisions=$PREV_REV=100 --region=us-central1

# Scale up resources
gcloud run services update shyvr-rlte --memory=16Gi --cpu=8 --max-instances=20 --region=us-central1
```

#### Recovery Phase (5-60 minutes)
```bash
# Clear caches
curl -X POST "${SERVICE_URL}/admin/cache/clear-all" -H "Authorization: Bearer $TOKEN"

# Restart models
curl -X POST "${SERVICE_URL}/admin/models/restart-all" -H "Authorization: Bearer $TOKEN"

# Health verification
for i in {1..10}; do
    curl -f -s "${SERVICE_URL}/health" && echo "Health check $i: OK" && break
    sleep 30
done
```

### Database Emergency Recovery
```sql
-- Kill long-running queries
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE state = 'active' AND now() - query_start > interval '5 minutes';

-- Check connections
SELECT datname, count(*) FROM pg_stat_activity GROUP BY datname;

-- Check locks
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
    JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
    JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

## Diagnostic Tools and Commands

### Comprehensive Diagnostics
```bash
#!/bin/bash
# comprehensive_diagnostics.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="diagnostics_$TIMESTAMP"
mkdir -p $REPORT_DIR

echo "🔍 Running comprehensive diagnostics..."

# System info
{
    echo "=== System Information ==="
    date; uname -a
    gcloud config get-value project
} > $REPORT_DIR/system_info.txt

# Service status
{
    echo "=== Service Status ==="
    gcloud run services describe shyvr-rlte --region=us-central1
} > $REPORT_DIR/service_status.txt

# Health checks
{
    echo "=== Health Checks ==="
    curl -s "${SERVICE_URL}/health" | jq '.'
    curl -s "${SERVICE_URL}/health/transformers" | jq '.'
    curl -s "${SERVICE_URL}/health/database" | jq '.'
} > $REPORT_DIR/health_checks.json

# Performance metrics
curl -s "${SERVICE_URL}/metrics" > $REPORT_DIR/metrics.txt
curl -s "${SERVICE_URL}/debug/memory/breakdown" | jq '.' > $REPORT_DIR/memory_usage.json

# Recent logs
gcloud logging read "resource.type=cloud_run_revision AND \
    resource.labels.service_name=shyvr-rlte AND \
    timestamp >= \"$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)\"" \
    --limit=500 > $REPORT_DIR/recent_logs.json

# Error logs
gcloud logging read "resource.type=cloud_run_revision AND \
    resource.labels.service_name=shyvr-rlte AND \
    severity>=ERROR" --limit=100 > $REPORT_DIR/error_logs.json

echo "✅ Diagnostics complete: $REPORT_DIR"
```

### Monitoring Commands
```bash
# Real-time health monitoring
watch -n 5 'curl -s "${SERVICE_URL}/health" | jq ".status, .checks"'

# Resource monitoring
watch -n 10 'curl -s "${SERVICE_URL}/metrics" | grep -E "(memory|cpu|latency)"'

# Error monitoring
watch -n 30 'gcloud logging read "resource.type=cloud_run_revision AND \
    severity>=ERROR" --limit=10 --format="value(textPayload)"'

# Model performance monitoring
watch -n 15 'for model in itransformer patchtst timesmixer timesfm; do
    echo "=== $model ==="
    curl -s "${SERVICE_URL}/health/models/$model" | jq ".status, .avg_inference_latency_ms"
done'
```

## Common Error Codes

### HTTP Status Codes
```yaml
client_errors:
  400: "Bad Request - Invalid input parameters"
  401: "Unauthorized - Missing or invalid authentication"
  403: "Forbidden - Insufficient permissions"
  404: "Not Found - Resource does not exist"
  429: "Too Many Requests - Rate limit exceeded"

server_errors:
  500: "Internal Server Error - Unexpected error"
  502: "Bad Gateway - Upstream service error"
  503: "Service Unavailable - Service temporarily down"
  504: "Gateway Timeout - Request timeout"
```

### Application Error Codes
```yaml
model_errors:
  MODEL_001: "Model loading failed"
  MODEL_002: "Model inference timeout"
  MODEL_003: "Model memory allocation failed"
  MODEL_004: "Model version incompatible"

database_errors:
  DB_001: "Database connection failed"
  DB_002: "Database query timeout"
  DB_003: "Database transaction failed"
  DB_004: "Connection pool exhausted"

api_errors:
  API_001: "External API unavailable"
  API_002: "API rate limit exceeded"
  API_003: "API authentication failed"
  API_004: "API response timeout"
```

### Resolution Strategies by Error Type

#### Model Errors
- **MODEL_001**: Clear cache, restart service, check memory
- **MODEL_002**: Optimize inference, increase timeout, check resources
- **MODEL_003**: Increase memory allocation, enable model sharding
- **MODEL_004**: Rollback to compatible version, update model

#### Database Errors  
- **DB_001**: Check network, verify credentials, test connectivity
- **DB_002**: Optimize queries, add indexes, increase timeout
- **DB_003**: Check locks, retry transaction, investigate conflicts
- **DB_004**: Increase pool size, kill idle connections, restart service

#### API Errors
- **API_001**: Enable circuit breaker, use fallback data, check network
- **API_002**: Implement rate limiting, add delays, upgrade plan
- **API_003**: Verify credentials, refresh tokens, check permissions
- **API_004**: Increase timeout, implement retry logic, check network

## Best Practices for Troubleshooting

### General Guidelines
1. **Start with health checks** - Always verify overall system health first
2. **Collect evidence** - Gather logs, metrics, and diagnostic data before changes
3. **Make minimal changes** - Prefer small, targeted fixes over large changes
4. **Test changes** - Verify fixes work before considering issue resolved
5. **Document solutions** - Record what worked for future reference

### Escalation Procedures
1. **Level 1** (0-15 minutes): Self-service diagnostics and common fixes
2. **Level 2** (15-60 minutes): Advanced troubleshooting and service restarts  
3. **Level 3** (1+ hours): Engineering escalation and architectural changes

### Post-Incident Actions
- Conduct root cause analysis
- Update monitoring and alerting
- Improve documentation
- Implement preventive measures
- Schedule follow-up review

## Conclusion

This troubleshooting guide provides systematic approaches to resolving issues in the transformer-enabled RLTE system. Key takeaways:

- **Use systematic diagnosis** to identify root causes efficiently
- **Implement comprehensive monitoring** to detect issues early
- **Maintain emergency procedures** for rapid incident response
- **Document all solutions** to improve future troubleshooting
- **Focus on prevention** through proper monitoring and alerting

For critical issues requiring immediate escalation, contact the technical team using the emergency procedures outlined above.

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-06  
**Maintained By**: ML Development Team