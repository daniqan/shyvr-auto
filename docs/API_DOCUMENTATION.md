# Shyvr AI RLTE API Documentation

## Overview

The Shyvr AI RLTE provides a comprehensive REST API for trading bot management, RL experience storage, XAI analysis, and real-time monitoring. The API is built with FastAPI and supports both REST endpoints and WebSocket connections for real-time updates.

## Base URL

```
Production: https://shyvr-rlte-service-url.run.app
Development: http://localhost:8000
```

## Authentication

The API uses JWT-based authentication with role-based access control:

- **Read**: View-only access to data and status
- **Write**: Modify settings and configurations
- **Admin**: Full system administration
- **Trading**: Execute trades and manage positions

### Authentication Headers

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

## Core API Endpoints

### System Health & Status

#### GET /health
Returns comprehensive system health information including database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "timestamp": "2025-07-30T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "connectivity": true,
      "database_size": "256 MB",
      "active_connections": 5,
      "tables_exist": true
    },
    "rl_experience_storage": {
      "status": "healthy",
      "total_experiences": 150000,
      "storage_rate_per_second": 850,
      "avg_query_latency_ms": 35
    }
  }
}
```

#### GET /config
Returns current system configuration.

**Permissions:** Read  
**Response:** Configuration object with current settings

### Trading Control

#### POST /dashboard/trading/mode
Switch trading mode between analysis, simulation, and live trading.

**Permissions:** Trading  
**Request Body:**
```json
{
  "mode": "simulation"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Trading mode switched to simulation",
  "current_mode": "simulation"
}
```

#### GET /dashboard/trading/status
Get current trading status and active positions.

**Permissions:** Read  
**Response:**
```json
{
  "mode": "simulation",
  "active_positions": 3,
  "total_pnl": 150.25,
  "daily_pnl": 25.75,
  "last_trade": "2025-07-30T11:45:00Z"
}
```

#### POST /dashboard/trading/risk-limits
Update risk management limits.

**Permissions:** Trading  
**Request Body:**
```json
{
  "max_position_size_usd": 1000,
  "max_daily_loss_usd": 500,
  "max_drawdown_pct": 15,
  "stop_loss_pct": 5,
  "take_profit_pct": 10
}
```

## RL Experience Storage API

### Experience Collection

#### GET /dashboard/rl/experiences
Retrieve RL experiences with filtering and pagination.

**Permissions:** Read  
**Query Parameters:**
- `session_id` (optional): Filter by training session
- `trading_mode` (optional): Filter by trading mode
- `limit` (default: 100): Number of experiences to return
- `offset` (default: 0): Pagination offset
- `start_date` (optional): Filter by creation date
- `end_date` (optional): Filter by creation date

**Response:**
```json
{
  "experiences": [
    {
      "id": 12345,
      "experience_id": "uuid-string",
      "session_id": "session-uuid",
      "state_data": {
        "price": 0.00123,
        "volume_24h": 150000,
        "rsi": 65.0,
        "position_size": 0.1
      },
      "action": 1,
      "reward": 0.15,
      "next_state_data": {
        "price": 0.00125,
        "volume_24h": 160000,
        "rsi": 70.0,
        "position_size": 0.2
      },
      "done": false,
      "priority": 0.8,
      "trading_mode": "simulation",
      "token_address": "0x1234...",
      "chain": "ethereum",
      "created_at": "2025-07-30T12:00:00Z"
    }
  ],
  "total_count": 150000,
  "pagination": {
    "limit": 100,
    "offset": 0,
    "has_more": true
  }
}
```

#### POST /dashboard/rl/experiences
Store new RL experience (internal use, typically called by trading modes).

**Permissions:** Trading  
**Request Body:**
```json
{
  "session_id": "session-uuid",
  "state_data": {
    "price": 0.00123,
    "rsi": 65.0,
    "volume_24h": 150000
  },
  "action": 1,
  "reward": 0.15,
  "next_state_data": {
    "price": 0.00125,
    "rsi": 70.0,
    "volume_24h": 160000
  },
  "done": false,
  "trading_mode": "simulation",
  "token_address": "0x1234...",
  "chain": "ethereum"
}
```

### Training Sessions

#### GET /dashboard/rl/sessions
Get list of training sessions with statistics.

**Permissions:** Read  
**Query Parameters:**
- `status` (optional): Filter by session status
- `trading_mode` (optional): Filter by trading mode
- `limit` (default: 50): Number of sessions to return

**Response:**
```json
{
  "sessions": [
    {
      "id": 123,
      "session_id": "session-uuid",
      "session_name": "DQN Training Session 1",
      "trading_mode": "simulation",
      "total_experiences": 50000,
      "successful_experiences": 32000,
      "failed_experiences": 18000,
      "total_reward": 1250.75,
      "average_reward": 0.025,
      "session_status": "completed",
      "started_at": "2025-07-29T10:00:00Z",
      "ended_at": "2025-07-30T08:00:00Z",
      "duration_seconds": 79200
    }
  ]
}
```

#### POST /dashboard/rl/sessions
Create new training session.

**Permissions:** Trading  
**Request Body:**
```json
{
  "session_name": "New DQN Training",
  "trading_mode": "simulation",
  "agent_config": {
    "algorithm": "DQN",
    "learning_rate": 0.001,
    "epsilon": 0.1,
    "batch_size": 32
  },
  "environment_config": {
    "initial_balance": 1000.0,
    "max_position_size": 0.5,
    "transaction_cost": 0.001
  }
}
```

#### GET /dashboard/rl/sessions/{session_id}
Get detailed information about a specific training session.

**Permissions:** Read  
**Response:**
```json
{
  "session": {
    "id": 123,
    "session_id": "session-uuid",
    "session_name": "DQN Training Session 1",
    "trading_mode": "simulation",
    "agent_config": {
      "algorithm": "DQN",
      "learning_rate": 0.001,
      "epsilon": 0.1
    },
    "total_experiences": 50000,
    "performance_metrics": {
      "sharpe_ratio": 1.85,
      "max_drawdown": 0.15,
      "win_rate": 0.64
    },
    "created_at": "2025-07-29T10:00:00Z"
  }
}
```

### Performance Metrics

#### GET /dashboard/rl/metrics
Get aggregated performance metrics.

**Permissions:** Read  
**Query Parameters:**
- `session_id` (optional): Filter by session
- `metric_type` (optional): Filter by metric type
- `aggregation_level` (optional): instant, episode, session, hourly, daily
- `start_date` (optional): Date range filter
- `end_date` (optional): Date range filter

**Response:**
```json
{
  "metrics": [
    {
      "metric_id": "metric-uuid",
      "session_id": "session-uuid",
      "metric_type": "reward",
      "metric_name": "average_episode_reward",
      "metric_value": 0.125,
      "metric_unit": "reward_units",
      "aggregation_level": "session",
      "created_at": "2025-07-30T12:00:00Z"
    }
  ]
}
```

#### GET /dashboard/rl/performance-summary
Get high-level performance summary across all sessions.

**Permissions:** Read  
**Response:**
```json
{
  "summary": {
    "total_experiences": 500000,
    "total_sessions": 25,
    "active_sessions": 2,
    "avg_reward_per_experience": 0.034,
    "best_session_reward": 2.45,
    "storage_rate_per_second": 850,
    "avg_query_latency_ms": 35
  }
}
```

## XAI (Explainable AI) API

### Explanations

#### GET /xai/explanations
Get recent XAI explanations with pagination.

**Permissions:** Read  
**Query Parameters:**
- `limit` (default: 50): Number of explanations
- `offset` (default: 0): Pagination offset
- `decision_type` (optional): Filter by decision type

**Response:**
```json
{
  "explanations": [
    {
      "explanation_id": "uuid",
      "decision_id": "decision-uuid",
      "explainer_type": "LIME",
      "feature_importance": {
        "rsi": 0.35,
        "price_change": 0.28,
        "volume": 0.22,
        "position_size": 0.15
      },
      "confidence": 0.87,
      "explanation_data": {
        "local_explanation": "High RSI indicates overbought condition",
        "model_prediction": "SELL"
      },
      "created_at": "2025-07-30T12:00:00Z"
    }
  ]
}
```

#### GET /xai/explanations/{decision_id}
Get detailed explanation for a specific trading decision.

**Permissions:** Read  
**Response:**
```json
{
  "explanation": {
    "decision_id": "decision-uuid",
    "explainer_type": "LIME",
    "confidence": 0.87,
    "feature_importance": {
      "rsi": 0.35,
      "price_change": 0.28,
      "volume": 0.22
    },
    "explanation_text": "Decision based primarily on RSI overbought signal...",
    "model_data": {
      "prediction": "SELL",
      "prediction_probability": 0.85
    }
  }
}
```

#### GET /xai/feature-importance
Get feature importance summary across models.

**Permissions:** Read  
**Response:**
```json
{
  "feature_importance": {
    "global_importance": {
      "rsi": 0.32,
      "price_change": 0.28,
      "volume": 0.25,
      "moving_average": 0.15
    },
    "by_explainer": {
      "LIME": {
        "rsi": 0.35,
        "price_change": 0.30
      },
      "Permutation": {
        "rsi": 0.29,
        "volume": 0.31
      }
    }
  }
}
```

## Portfolio Management

#### GET /dashboard/portfolio/overview
Get current portfolio status and allocation.

**Permissions:** Read  
**Response:**
```json
{
  "portfolio": {
    "total_value_usd": 10500.25,
    "cash_balance": 2500.00,
    "invested_value": 8000.25,
    "total_pnl": 500.25,
    "daily_pnl": 125.75,
    "positions": [
      {
        "token_symbol": "ETH",
        "token_address": "0x123...",
        "quantity": 5.25,
        "avg_cost": 1800.00,
        "current_price": 1850.00,
        "pnl": 262.50,
        "pnl_percentage": 2.78
      }
    ]
  }
}
```

#### GET /dashboard/portfolio/performance
Get historical portfolio performance data.

**Permissions:** Read  
**Query Parameters:**
- `timeframe`: 1h, 4h, 1d, 7d, 30d
- `metrics`: pnl, sharpe, drawdown, win_rate

**Response:**
```json
{
  "performance": {
    "timeframe": "7d",
    "data_points": [
      {
        "timestamp": "2025-07-30T12:00:00Z",
        "total_value": 10500.25,
        "pnl": 500.25,
        "drawdown": 0.05
      }
    ],
    "summary": {
      "total_return": 0.05,
      "sharpe_ratio": 1.25,
      "max_drawdown": 0.12,
      "win_rate": 0.68
    }
  }
}
```

## Activity Logging

#### GET /dashboard/activity/logs
Retrieve system activity logs with filtering.

**Permissions:** Read  
**Query Parameters:**
- `category`: system, trading, user, ml_rl
- `severity`: trace, debug, info, warning, error, critical, alert, emergency
- `start_date`, `end_date`: Date range filters
- `limit`, `offset`: Pagination

**Response:**
```json
{
  "logs": [
    {
      "id": 12345,
      "timestamp": "2025-07-30T12:00:00Z",
      "category": "trading",
      "action": "execute",
      "severity": "info",
      "source": "simulation_mode",
      "message": "Executed BUY order for ETH",
      "metadata": {
        "token": "ETH",
        "quantity": 1.0,
        "price": 1850.00
      }
    }
  ]
}
```

## WebSocket API

### Real-Time Updates

Connect to WebSocket for real-time updates:

```
ws://localhost:8000/ws
wss://production-url/ws
```

#### Connection Authentication
Send JWT token in connection query parameter:
```
wss://production-url/ws?token=<jwt_token>
```

#### Message Types

**Trading Updates:**
```json
{
  "type": "trading_update",
  "data": {
    "mode": "simulation",
    "action": "BUY",
    "token": "ETH",
    "quantity": 1.0,
    "price": 1850.00,
    "timestamp": "2025-07-30T12:00:00Z"
  }
}
```

**RL Experience Updates:**
```json
{
  "type": "rl_experience",
  "data": {
    "session_id": "session-uuid",
    "experiences_count": 50000,
    "avg_reward": 0.025,
    "storage_rate": 850
  }
}
```

**System Health Updates:**
```json
{
  "type": "system_health",
  "data": {
    "status": "healthy",
    "database_connectivity": true,
    "active_connections": 5,
    "experience_storage_rate": 850
  }
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Trading mode must be one of: analysis, simulation, live",
    "details": {
      "field": "mode",
      "provided_value": "invalid_mode"
    },
    "timestamp": "2025-07-30T12:00:00Z"
  }
}
```

## Rate Limiting

- **General API**: 100 requests per minute per IP
- **WebSocket**: 10 connections per IP
- **Database Operations**: 1000 requests per minute per authenticated user

## Data Export

#### GET /dashboard/export/experiences
Export RL experiences as CSV.

**Permissions:** Read  
**Query Parameters:**
- `session_id` (optional): Filter by session
- `format`: csv, json
- `start_date`, `end_date`: Date range

**Response:** CSV download or JSON array

#### GET /dashboard/export/portfolio
Export portfolio performance data.

**Permissions:** Read  
**Response:** CSV/JSON with historical performance data

## Monitoring & Observability

The API includes comprehensive monitoring and observability features:

- **Prometheus Metrics**: All endpoints expose metrics for monitoring
- **Structured Logging**: Comprehensive request/response logging
- **Health Checks**: Deep health validation including database connectivity
- **Performance Tracking**: Real-time performance metrics collection

## Development & Testing

### Local Development

```bash
# Start local server
uvicorn src.main:app --reload --port 8000

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
# OpenAPI spec at http://localhost:8000/openapi.json
```

### Testing Endpoints

Use the interactive documentation at `/docs` for testing, or use curl:

```bash
# Get system health
curl -X GET "http://localhost:8000/health"

# Get RL experiences (with auth)
curl -X GET "http://localhost:8000/dashboard/rl/experiences" \
  -H "Authorization: Bearer <token>"
```

## SDK & Client Libraries

Official client libraries are planned for:
- Python SDK
- JavaScript/TypeScript SDK
- CLI tool

---

*API Documentation Version: 1.0*  
*Last Updated: 2025-07-30*  
*Corresponds to RLTE System with RL Experience Storage*