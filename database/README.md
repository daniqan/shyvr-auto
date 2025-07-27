# Database Integration for RLTE Activity Logging

This directory contains the database schema and setup files for the Shyvr RLTE activity logging system.

## Overview

The database integration provides:

- **Comprehensive activity logging** to PostgreSQL database
- **Real-time health monitoring** through the `/health` API endpoint  
- **High-performance async connection pooling** with retry logic
- **Structured data storage** with JSONB metadata and array fields
- **Performance-optimized indexes** for common query patterns
- **Automatic data summarization** and cleanup functions
- **User session tracking** with activity metrics

## Schema Files

### `schema/001_create_activity_logging_schema.sql`

Complete database schema including:

- **Core Tables:**
  - `activity_logs` - Main activity logging table with 40+ fields
  - `activity_summaries` - Pre-aggregated metrics for performance
  - `user_activity_sessions` - User session tracking
  - `users` - Basic user information

- **Enums:**
  - `activity_category` - System, trading, user, ML/RL, etc.
  - `activity_action` - Create, execute, error, success, etc.  
  - `activity_severity` - Trace through emergency levels
  - `trading_mode` - Analysis, simulation, live
  - `chain_type` - Ethereum, Solana, Base

- **Performance Features:**
  - 15+ optimized indexes including GIN indexes for JSONB/arrays
  - Composite indexes for common query patterns
  - Materialized views for frequently accessed data

- **Automation:**
  - Triggers for automatic session tracking
  - Functions for data cleanup and summarization
  - Proper foreign key constraints and data validation

## Setup Instructions

### 1. Install PostgreSQL

```bash
# macOS with Homebrew
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE shyvr_rlte;
CREATE USER rlte_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE shyvr_rlte TO rlte_user;
\q
```

### 2. Apply Database Schema

```bash
# From the project root directory
psql -d shyvr_rlte -f database/schema/001_create_activity_logging_schema.sql
```

### 3. Update Configuration

Ensure your configuration includes database credentials:

```yaml
database:
  host: localhost
  port: 5432  
  database: shyvr_rlte
  username: rlte_user
  password: your_secure_password
  pool_size: 10
```

### 4. Verify Integration

Test the database integration:

```bash
# Check health endpoint
curl http://localhost:8000/health

# Should return database status in components.database
```

## Key Features Implemented

### ✅ Database Connection Logic
- Async connection pooling with `asyncpg`
- Automatic retry logic with exponential backoff
- Connection health monitoring and error handling
- Proper resource cleanup and connection management

### ✅ Activity Logger Integration  
- Buffered writes for performance (100 entries or 5s timeout)
- Comprehensive structured logging with 40+ fields
- Support for trading, ML/RL, user, system, and API activities
- Automatic checksum generation for data integrity
- High-severity events trigger immediate database writes

### ✅ Health Check Integration
- Real-time database connectivity testing
- Database size and connection monitoring  
- Table existence verification
- Integrated into `/health` API endpoint
- Proper error handling and status reporting

### ✅ Performance Optimizations
- Materialized views for common queries
- Optimized indexes for time-series and categorical data
- GIN indexes for JSONB metadata searches
- Automatic data summarization functions
- Connection pooling with configurable sizing

## API Integration

### Health Endpoint (`/health`)

The health endpoint now includes comprehensive database status:

```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "environment": "production",
  "timestamp": "2024-01-01T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "connectivity": true,
      "database_size": "256 MB",
      "active_connections": 5,
      "tables_exist": true,
      "details": {...}
    }
  }
}
```

### Activity Logging API

Activity logging is automatically integrated into:

- Dashboard API calls (`/dashboard/activity/logs`)  
- User actions tracking
- Trading activity monitoring
- System events and errors
- Performance metrics collection

## Database Statistics

The system provides detailed database statistics through:

```python
from src.utils.database import get_database_stats

stats = await get_database_stats()
# Returns activity counts, error rates, user metrics, etc.
```

## Materialized Views

Pre-computed views for performance:

- `recent_activity` - Last 24 hours of activity
- `error_summary` - Error counts by category/source  
- `performance_metrics` - Execution time statistics
- `user_activity_overview` - User engagement metrics
- `trading_activity_summary` - Trading volume and performance

## Data Retention

Automatic cleanup functions:

```sql
-- Clean up logs older than 90 days (keeps critical/alert/emergency)
SELECT cleanup_old_activity_logs(90);

-- Generate hourly summaries 
SELECT generate_activity_summaries('hour');
```

## Testing

The database integration includes comprehensive tests:

- Connection pooling and retry logic
- Health check functionality  
- Activity logger database writes
- Data structure validation
- Error handling and recovery

Tests can be run with mocked PostgreSQL connections for CI/CD environments.

## Security

- Dedicated database user (`rlte_user`) with minimal required permissions
- Password-based authentication 
- Connection encryption support
- Data integrity checksums
- SQL injection prevention through parameterized queries

## Troubleshooting

### Connection Issues

1. **Check PostgreSQL is running:**
   ```bash
   sudo systemctl status postgresql  # Linux
   brew services list | grep postgres  # macOS
   ```

2. **Verify database exists:**
   ```bash
   psql -l | grep shyvr_rlte
   ```

3. **Test connectivity:**
   ```bash
   psql -h localhost -U rlte_user -d shyvr_rlte -c "SELECT 1;"
   ```

### Performance Issues

1. **Check database size:**
   ```sql
   SELECT pg_size_pretty(pg_database_size('shyvr_rlte'));
   ```

2. **Monitor active connections:**
   ```sql
   SELECT count(*) FROM pg_stat_activity WHERE datname = 'shyvr_rlte';
   ```

3. **Refresh materialized views:**
   ```sql
   REFRESH MATERIALIZED VIEW CONCURRENTLY recent_activity;
   ```

The database integration is now fully functional and ready for production use with proper PostgreSQL setup.