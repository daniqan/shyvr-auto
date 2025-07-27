# Activity Logging Database Schema Documentation

## Overview

This document describes the comprehensive activity logging database schema designed for the Shyvr RLTE dashboard. The schema is optimized for high-volume logging while maintaining excellent query performance for dashboard analytics and real-time monitoring.

## Design Principles

### 1. **High Performance**
- Optimized indexes for time-based queries (most common dashboard pattern)
- Partial indexes to reduce storage overhead
- GIN indexes for JSONB and array searching
- Pre-aggregated summaries for dashboard performance

### 2. **Comprehensive Coverage**
- Covers all system activities: trading, ML/RL, user interactions, security, etc.
- Rich context and metadata capture
- Hierarchical activity relationships
- Session and request correlation

### 3. **Scalability**
- Designed for table partitioning by time
- Automated data retention policies
- Summary tables for historical analysis
- Efficient storage with appropriate data types

### 4. **Dashboard Integration**
- Optimized views for common dashboard queries
- Real-time activity feeds
- Performance metrics aggregation
- Error tracking and alerting

## Schema Components

### Core Tables

#### 1. `activity_logs` - Main Activity Logging Table

The primary table for all system activities with comprehensive fields for context, performance metrics, and metadata.

**Key Features:**
- High-precision timestamps for accurate ordering
- Rich categorization with enums
- Flexible JSONB metadata storage
- Performance and error tracking
- Security and audit context

**Primary Fields:**

| Field | Type | Description | Usage Pattern |
|-------|------|-------------|---------------|
| `id` | BIGSERIAL | Primary key for internal use | Database operations |
| `activity_id` | UUID | Unique identifier for external references | API responses, correlation |
| `created_at` | TIMESTAMP(6) WITH TIME ZONE | High-precision timestamp | Time-based queries, ordering |
| `category` | activity_category | Activity classification | Filtering, dashboard sections |
| `action` | activity_action | Specific action type | Granular filtering |
| `severity` | activity_severity | Log level/importance | Error monitoring, alerting |
| `source` | VARCHAR(100) | Component generating the log | Component monitoring |
| `event_type` | VARCHAR(100) | Specific event identifier | Event analysis |
| `title` | VARCHAR(255) | Human-readable summary | Dashboard display |
| `description` | TEXT | Detailed information | Error investigation |

**Context Fields:**

| Field | Type | Description | Usage Pattern |
|-------|------|-------------|---------------|
| `user_id` | BIGINT | User association | User activity tracking |
| `session_id` | UUID | Session tracking | Session analysis |
| `request_id` | UUID | Request correlation | Request tracing |
| `trading_mode` | trading_mode | Trading context | Mode-specific analysis |
| `token_address` | VARCHAR(64) | Related token | Token activity tracking |
| `chain` | chain_type | Blockchain context | Chain-specific analysis |

**Performance Fields:**

| Field | Type | Description | Usage Pattern |
|-------|------|-------------|---------------|
| `execution_time_ms` | INTEGER | Operation duration | Performance monitoring |
| `response_time_ms` | INTEGER | Response time | API performance |
| `memory_usage_mb` | INTEGER | Memory usage | Resource monitoring |
| `cpu_usage_pct` | DECIMAL(5,2) | CPU usage | Performance analysis |

**Metadata and Flexibility:**

| Field | Type | Description | Usage Pattern |
|-------|------|-------------|---------------|
| `metadata` | JSONB | Flexible additional data | Custom analytics |
| `tags` | TEXT[] | Searchable tags | Flexible categorization |
| `correlation_id` | VARCHAR(100) | External correlation | Integration tracking |

#### 2. `activity_summaries` - Pre-aggregated Analytics

Pre-computed summaries for dashboard performance, reducing query load on the main table.

**Features:**
- Hourly, daily, weekly summaries
- Performance metrics aggregation
- Error rate calculations
- Top sources and events tracking

#### 3. `user_activity_sessions` - Session Tracking

Tracks user sessions for comprehensive user behavior analysis.

**Features:**
- Session lifecycle management
- Activity counting and metrics
- Cross-component usage tracking
- Error correlation per session

### Enums and Types

#### `activity_category`
Organizes activities into logical groups for dashboard sections:

| Value | Description | Dashboard Section |
|-------|-------------|-------------------|
| `system` | System operations, startup, shutdown | System Health |
| `trading` | Trading activities, orders, executions | Trading Dashboard |
| `user` | User interactions, authentication | User Management |
| `ml_rl` | Machine learning and RL activities | ML/RL Monitoring |
| `security` | Security events, violations | Security Dashboard |
| `api` | External API interactions | API Monitoring |
| `performance` | Performance alerts and metrics | Performance Dashboard |
| `data` | Data processing, validation | Data Quality |
| `configuration` | Configuration changes | System Settings |
| `integration` | Component integration events | Integration Health |
| `dashboard` | Dashboard-specific events | Dashboard Analytics |

#### `activity_severity`
Standard logging levels for filtering and alerting:

| Level | Purpose | Dashboard Treatment |
|-------|---------|-------------------|
| `trace` | Fine debugging | Hidden by default |
| `debug` | Development debugging | Debug mode only |
| `info` | General information | Standard display |
| `notice` | Notable conditions | Highlighted |
| `warning` | Warning conditions | Yellow indicators |
| `error` | Error conditions | Red indicators |
| `critical` | Critical system issues | Alert notifications |
| `alert` | Immediate action required | High-priority alerts |
| `emergency` | System unusable | Emergency notifications |

#### `activity_action`
Granular action classification for detailed analysis:

| Action | Description | Use Cases |
|--------|-------------|-----------|
| `create` | Resource creation | New trades, users, configurations |
| `read` | Data access | API calls, data queries |
| `update` | Modifications | Settings changes, position updates |
| `delete` | Resource removal | Order cancellations, cleanup |
| `execute` | Action execution | Trade execution, model training |
| `start`/`stop` | Service lifecycle | Component management |
| `error`/`success` | Operation outcomes | Success/failure tracking |
| `login`/`logout` | Authentication | Security monitoring |

## Performance Optimization

### Index Strategy

#### 1. **Time-based Indexes (Primary)**
```sql
-- Most important - dashboard queries are primarily time-based
CREATE INDEX idx_activity_logs_created_at_desc ON activity_logs (created_at DESC);
CREATE INDEX idx_activity_logs_created_at_category ON activity_logs (created_at DESC, category);
```

#### 2. **Category and Severity Filtering**
```sql
-- Dashboard sections and error monitoring
CREATE INDEX idx_activity_logs_category_action ON activity_logs (category, action);
CREATE INDEX idx_activity_logs_severity_created ON activity_logs (severity, created_at DESC) 
WHERE severity IN ('error', 'critical', 'alert', 'emergency');
```

#### 3. **Partial Indexes**
```sql
-- Only index relevant data to reduce storage
CREATE INDEX idx_activity_logs_user_id_created ON activity_logs (user_id, created_at DESC) 
WHERE user_id IS NOT NULL;
```

#### 4. **JSONB and Array Indexes**
```sql
-- Fast searching of metadata and tags
CREATE INDEX idx_activity_logs_metadata_gin ON activity_logs USING GIN (metadata);
CREATE INDEX idx_activity_logs_tags_gin ON activity_logs USING GIN (tags);
```

### Query Optimization

#### Common Dashboard Queries

**1. Recent Activity Feed:**
```sql
-- Optimized with idx_activity_logs_created_at_desc
SELECT activity_id, created_at, category, title, severity 
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC 
LIMIT 100;
```

**2. Error Monitoring:**
```sql
-- Optimized with idx_activity_logs_severity_created
SELECT source, error_code, COUNT(*) 
FROM activity_logs 
WHERE severity IN ('error', 'critical') 
AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY source, error_code
ORDER BY COUNT(*) DESC;
```

**3. User Activity Tracking:**
```sql
-- Optimized with idx_activity_logs_user_id_created
SELECT category, COUNT(*) 
FROM activity_logs 
WHERE user_id = $1 
AND created_at > NOW() - INTERVAL '7 days'
GROUP BY category;
```

**4. Performance Metrics:**
```sql
-- Fast aggregation for dashboard
SELECT source, AVG(execution_time_ms), COUNT(*)
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '1 hour'
AND execution_time_ms IS NOT NULL
GROUP BY source;
```

## Scalability Features

### 1. Table Partitioning

For high-volume installations, implement monthly partitioning:

```sql
-- Example monthly partition
CREATE TABLE activity_logs_2024_01 PARTITION OF activity_logs 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### 2. Data Retention

Automated cleanup with configurable retention periods:

```sql
-- Retention function (default 90 days)
SELECT cleanup_old_activity_logs();
```

**Retention Policy:**
- **Regular logs**: 90 days (configurable)
- **Critical logs**: 1 year (archived, not deleted)
- **Summaries**: 2 years
- **User sessions**: 6 months

### 3. Summary Generation

Pre-aggregated summaries for historical analysis:

```sql
-- Generate hourly summaries
SELECT generate_activity_summaries(
    '2024-01-01 00:00:00'::timestamp, 
    '2024-01-01 01:00:00'::timestamp, 
    'hour'
);
```

## Dashboard Integration

### Real-time Updates

The schema supports real-time dashboard updates through:

1. **WebSocket Integration**: Activity logs trigger real-time notifications
2. **Cached Views**: Pre-computed views for fast dashboard loading
3. **Session Tracking**: Real-time user session monitoring
4. **Performance Metrics**: Live system health monitoring

### Dashboard Views

#### 1. `recent_activity`
Recent 24-hour activity feed for dashboard display:
```sql
SELECT * FROM recent_activity LIMIT 50;
```

#### 2. `error_summary`
Weekly error summary for monitoring dashboard:
```sql
SELECT * FROM error_summary WHERE error_count > 10;
```

#### 3. `performance_metrics`
Hourly performance metrics for system health:
```sql
SELECT * FROM performance_metrics WHERE error_rate_pct > 5;
```

#### 4. `user_activity_overview`
30-day user activity overview:
```sql
SELECT * FROM user_activity_overview WHERE total_activities > 0;
```

#### 5. `trading_activity_summary`
Daily trading activity breakdown:
```sql
SELECT * FROM trading_activity_summary 
WHERE activity_date >= CURRENT_DATE - INTERVAL '7 days';
```

## Usage Patterns

### 1. Application Logging

**Basic Activity Logging:**
```python
# Python example
activity_logger.log_activity(
    category=ActivityCategory.TRADING,
    action=ActivityAction.EXECUTE,
    severity=ActivitySeverity.INFO,
    source="trading_engine",
    event_type="trade_execution",
    title="Trade executed successfully",
    user_id=user_id,
    session_id=session_id,
    trading_mode=TradingMode.SIMULATION,
    token_address="0x123...",
    amount_usd=Decimal("100.50"),
    execution_time_ms=250,
    metadata={
        "order_id": "order_123",
        "strategy": "momentum",
        "confidence": 0.85
    },
    tags=["automated", "high_confidence"]
)
```

**Error Logging:**
```python
# Error logging with context
activity_logger.log_error(
    category=ActivityCategory.SYSTEM,
    source="ml_model",
    event_type="prediction_error",
    title="Model prediction failed",
    error_code="ML_PRED_001",
    error_message="Insufficient data for prediction",
    stack_trace=traceback.format_exc(),
    execution_time_ms=150,
    metadata={
        "model_name": "lstm_v2",
        "data_points": 45,
        "required_points": 60
    }
)
```

### 2. Dashboard Queries

**Activity Feed:**
```sql
-- Real-time activity feed
SELECT 
    activity_id,
    created_at,
    category,
    title,
    severity,
    source,
    execution_time_ms
FROM recent_activity 
WHERE category = 'trading' OR severity IN ('error', 'warning')
ORDER BY created_at DESC 
LIMIT 50;
```

**Error Dashboard:**
```sql
-- Error tracking dashboard
SELECT 
    source,
    error_code,
    COUNT(*) as occurrences,
    MAX(created_at) as last_seen,
    array_agg(DISTINCT severity) as severities
FROM activity_logs 
WHERE severity IN ('error', 'critical', 'alert', 'emergency')
AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY source, error_code
ORDER BY occurrences DESC;
```

**Performance Monitoring:**
```sql
-- System performance overview
SELECT 
    source,
    COUNT(*) as total_operations,
    AVG(execution_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_response_time,
    COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) as errors,
    ROUND(COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) * 100.0 / COUNT(*), 2) as error_rate
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '1 hour'
AND execution_time_ms IS NOT NULL
GROUP BY source
ORDER BY total_operations DESC;
```

### 3. Analytics and Reporting

**User Behavior Analysis:**
```sql
-- User engagement patterns
SELECT 
    DATE(created_at) as date,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(*) as total_activities,
    COUNT(DISTINCT session_id) as sessions,
    AVG(execution_time_ms) as avg_response_time
FROM activity_logs 
WHERE created_at > NOW() - INTERVAL '30 days'
AND user_id IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

**Trading Activity Analysis:**
```sql
-- Trading mode effectiveness
SELECT 
    trading_mode,
    COUNT(*) as total_trades,
    SUM(amount_usd) as total_volume,
    COUNT(*) FILTER (WHERE severity = 'error') as failed_trades,
    AVG(execution_time_ms) as avg_execution_time
FROM activity_logs 
WHERE category = 'trading' 
AND action = 'execute'
AND created_at > NOW() - INTERVAL '7 days'
GROUP BY trading_mode;
```

## Maintenance and Operations

### 1. Regular Maintenance

**Daily Tasks:**
- Monitor disk usage and growth rates
- Check index usage and performance
- Review error patterns and alerts
- Validate data retention policies

**Weekly Tasks:**
- Generate and review summary reports
- Analyze query performance
- Update statistics for query optimization
- Review and adjust retention policies

**Monthly Tasks:**
- Archive old data according to policies
- Review and optimize index strategies
- Analyze long-term trends
- Plan for storage scaling

### 2. Monitoring

**Key Metrics to Monitor:**
- Insert rate and volume
- Query response times
- Index usage efficiency
- Storage growth patterns
- Error rates and patterns

**Alert Thresholds:**
- Insert rate > 1000 records/minute
- Query response time > 100ms (95th percentile)
- Error rate > 5% in any 5-minute window
- Storage usage > 80% of allocated space

### 3. Backup and Recovery

**Backup Strategy:**
- Daily incremental backups
- Weekly full backups
- Real-time replication for critical systems
- Point-in-time recovery capabilities

**Recovery Testing:**
- Monthly recovery drills
- Automated backup validation
- Cross-region backup verification
- RTO/RPO compliance testing

## Migration and Updates

### Schema Evolution

The schema includes version tracking for migrations:

```sql
-- Version tracking in activity_logs table
version INTEGER DEFAULT 1
```

### Migration Strategy

1. **Backward Compatibility**: New fields are added as nullable
2. **Gradual Migration**: Dual-write during transition periods
3. **Index Management**: Online index creation for minimal downtime
4. **Data Migration**: Batch processing for large updates

### Future Enhancements

**Planned Improvements:**
1. **Time-series Optimization**: Specialized time-series storage for metrics
2. **ML Integration**: Automated anomaly detection on activity patterns
3. **Real-time Analytics**: Stream processing for instant insights
4. **Advanced Partitioning**: Automated partition management
5. **Compression**: Data compression for older partitions

## Security Considerations

### 1. Data Protection

- **PII Handling**: User data properly referenced, not duplicated
- **Sensitive Data**: Financial amounts stored with appropriate precision
- **Access Control**: Row-level security for multi-tenant scenarios
- **Audit Trail**: Complete audit trail for all modifications

### 2. Access Patterns

- **Read-only Views**: Restricted views for different user roles
- **Query Limits**: Rate limiting for expensive queries
- **Data Masking**: Automatic masking of sensitive information
- **Encryption**: At-rest and in-transit encryption

### 3. Compliance

- **Data Retention**: Configurable retention for compliance requirements
- **Right to be Forgotten**: User data deletion capabilities
- **Audit Requirements**: Complete audit trail maintenance
- **Regulatory Reporting**: Structured data for compliance reporting

---

This schema provides a robust foundation for comprehensive activity logging in the Shyvr RLTE system, balancing performance, scalability, and analytical capabilities while maintaining the flexibility needed for a complex trading and ML system.