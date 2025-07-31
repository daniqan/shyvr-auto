# Model Preservation API Reference

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Common Response Formats](#common-response-formats)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [List Models](#list-models)
  - [Get Model](#get-model)
  - [Rollback Model](#rollback-model)
  - [Delete Model Version](#delete-model-version)
  - [Health Check](#health-check)
- [Examples](#examples)

## Overview

The Model Preservation API provides REST endpoints for managing machine learning model versions, performing rollbacks, and monitoring system health. All endpoints require authentication and follow RESTful conventions.

### Features

- **Model Listing**: Browse preserved models with filtering and pagination
- **Model Retrieval**: Get specific model versions and metadata
- **Model Operations**: Rollback to previous versions, delete versions
- **Health Monitoring**: Check system status and statistics
- **Activity Logging**: All operations are logged for audit trails

## Authentication

The API uses the same authentication system as the main dashboard. All requests require valid authentication tokens.

### Authentication Methods

1. **Session Authentication**: For web dashboard integration
2. **API Key Authentication**: For programmatic access
3. **Bearer Token**: For service-to-service communication

### Permission Levels

- **Read Access**: Required for listing and retrieving models
- **Write Access**: Required for rollback operations
- **Admin Access**: Required for delete operations

```bash
# Example with API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     "https://your-domain.com/api/preservation/models"
```

## Base URL

```
https://your-domain.com/api/preservation
```

All endpoints are prefixed with `/api/preservation`.

## Common Response Formats

### Success Response Structure

```json
{
  "success": true,
  "data": {...},
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {...}
}
```

### Error Response Structure

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model dqn_agent version 1.2.3 not found",
    "details": {...}
  },
  "timestamp": "2024-01-15T10:30:00Z"
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
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### HTTP Methods Used

The API uses the following HTTP methods:

- `GET`: Retrieve resources (list models, get model details, health checks)
- `POST`: Create resources or perform actions (rollback operations)
- `DELETE`: Remove resources (delete model versions)
- `PUT`: Currently not implemented - would be used for updating model metadata in future versions

### Error Response Examples

```json
{
  "detail": "Model dqn_agent version 1.2.3 not found",
  "status_code": 404,
  "type": "ModelNotFoundError"
}
```

## Endpoints

### List Models

Get a paginated list of preserved models with optional filtering.

**Endpoint:** `GET /api/preservation/models`

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 50 | Number of models to return (1-1000) |
| `offset` | integer | No | 0 | Pagination offset |
| `model_type` | string | No | - | Filter by model type |
| `mode` | string | No | - | Filter by mode (analysis, simulation, live) |
| `state` | string | No | - | Filter by state |
| `branch` | string | No | - | Filter by branch |

**Response:**

```json
{
  "models": [
    {
      "model_id": "dqn_agent_1.2.3_main",
      "model_type": "dqn_agent",
      "version": "1.2.3",
      "mode": "simulation",
      "state": "active",
      "created_at": "2024-01-15T10:30:00Z",
      "size_mb": 45.2,
      "checksum": "sha256:abc123..."
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 127,
    "has_more": true
  },
  "filters": {
    "model_type": "dqn_agent"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Request:**

```bash
curl -X GET \
  "https://your-domain.com/api/preservation/models?limit=10&model_type=dqn_agent&mode=simulation" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

### Get Model

Retrieve a specific model by type and version.

**Endpoint:** `GET /api/preservation/models/{model_type}/{version}`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_type` | string | Yes | Type of model to retrieve |
| `version` | string | Yes | Specific version to retrieve |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | Mode for model retrieval |

**Response:**

```json
{
  "model": {
    "model_type": "dqn_agent",
    "version": "1.2.3",
    "mode": "simulation",
    "data_available": true,
    "size_bytes": 47423488
  },
  "metadata": {
    "description": "Enhanced reward function with 95% accuracy",
    "training_epochs": 100,
    "accuracy": 0.95,
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32
    },
    "tags": ["production", "stable"],
    "created_by": "admin",
    "branch": "main"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Request:**

```bash
curl -X GET \
  "https://your-domain.com/api/preservation/models/dqn_agent/1.2.3?mode=simulation" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

### Rollback Model

Rollback to a previous model version.

**Endpoint:** `POST /api/preservation/rollback`

**Request Body:**

```json
{
  "model_type": "dqn_agent",
  "target_version": "1.1.0",
  "mode": "simulation",
  "reason": "Performance regression in current version"
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_type` | string | Yes | Type of model to rollback |
| `target_version` | string | Yes | Target version to rollback to |
| `mode` | string | No | Mode to rollback in |
| `reason` | string | No | Reason for rollback |

**Response:**

```json
{
  "success": true,
  "message": "Successfully rolled back dqn_agent to version 1.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "model_type": "dqn_agent",
    "target_version": "1.1.0",
    "mode": "simulation",
    "reason": "Performance regression in current version",
    "user": "admin"
  }
}
```

**Example Request:**

```bash
curl -X POST \
  "https://your-domain.com/api/preservation/rollback" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dqn_agent",
    "target_version": "1.1.0",
    "mode": "simulation",
    "reason": "Performance regression in current version"
  }'
```

### Delete Model Version

Delete a specific model version (admin only).

**Endpoint:** `DELETE /api/preservation/models/{model_type}/{version}`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_type` | string | Yes | Type of model to delete |
| `version` | string | Yes | Specific version to delete |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | Mode for model deletion |

**Response:**

```json
{
  "success": true,
  "message": "Successfully deleted dqn_agent version 1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "model_type": "dqn_agent",
    "version": "1.0.0",
    "mode": "simulation",
    "user": "admin"
  }
}
```

**Example Request:**

```bash
curl -X DELETE \
  "https://your-domain.com/api/preservation/models/dqn_agent/1.0.0?mode=simulation" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

### Health Check

Get model preservation system health and statistics.

**Endpoint:** `GET /api/preservation/health`

**Response:**

```json
{
  "status": "healthy",
  "statistics": {
    "total_models": 127,
    "active_models": 95,
    "total_size_mb": 2048.7,
    "models_by_type": {
      "dqn_agent": 45,
      "lstm_predictor": 32,
      "transformer_model": 25,
      "cnn_classifier": 25
    },
    "models_by_mode": {
      "simulation": 67,
      "analysis": 35,
      "live": 25
    },
    "last_backup": "2024-01-15T09:00:00Z",
    "backup_success_rate": 98.5
  },
  "storage_health": {
    "status": "healthy",
    "gcs_connectivity": true,
    "database_connectivity": true,
    "cache_status": "operational"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Values:**

- `healthy`: All systems operational
- `degraded`: Some issues but functional
- `warning`: Potential issues detected
- `unhealthy`: Critical issues require attention
- `unavailable`: System not accessible

**Example Request:**

```bash
curl -X GET \
  "https://your-domain.com/api/preservation/health" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

## Examples

### Complete Workflow Example

Here's a complete example showing typical API usage:

```bash
#!/bin/bash

# Configuration
API_BASE="https://your-domain.com/api/preservation"
API_KEY="your-api-key"
HEADERS="Authorization: Bearer $API_KEY"

# 1. Check system health
echo "Checking system health..."
curl -s -X GET "$API_BASE/health" -H "$HEADERS" | jq '.'

# 2. List all DQN agent models
echo -e "\nListing DQN agent models..."
curl -s -X GET "$API_BASE/models?model_type=dqn_agent&limit=5" -H "$HEADERS" | jq '.models[]'

# 3. Get specific model details
echo -e "\nGetting model details..."
curl -s -X GET "$API_BASE/models/dqn_agent/1.2.3" -H "$HEADERS" | jq '.'

# 4. Rollback model (if needed)
echo -e "\nRolling back model..."
curl -s -X POST "$API_BASE/rollback" \
  -H "$HEADERS" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dqn_agent",
    "target_version": "1.1.0",
    "mode": "simulation",
    "reason": "Testing rollback functionality"
  }' | jq '.'

# 5. Verify rollback
echo -e "\nVerifying rollback..."
curl -s -X GET "$API_BASE/models/dqn_agent/1.1.0" -H "$HEADERS" | jq '.model'
```

### Additional curl Examples

#### List Models with Pagination
```bash
curl -X GET \
  "https://your-domain.com/api/preservation/models?limit=20&offset=40&mode=live" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

#### Get Model with Specific Mode
```bash
curl -X GET \
  "https://your-domain.com/api/preservation/models/lstm_predictor/2.1.0?mode=analysis" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

#### Rollback with Detailed Reason
```bash
curl -X POST \
  "https://your-domain.com/api/preservation/rollback" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "transformer_model",
    "target_version": "3.0.1",
    "mode": "live",
    "reason": "Critical bug found in version 3.1.0 affecting accuracy"
  }'
```

#### Delete Model Version (Admin Only)
```bash
curl -X DELETE \
  "https://your-domain.com/api/preservation/models/cnn_classifier/1.0.0-beta?mode=simulation" \
  -H "Authorization: Bearer YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json"
```

### Python Client Example

```python
import requests
import json
from typing import Dict, List, Optional

class ModelPreservationClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def list_models(self, limit: int = 50, offset: int = 0, 
                   model_type: Optional[str] = None, 
                   mode: Optional[str] = None) -> Dict:
        """List preserved models with optional filtering"""
        params = {'limit': limit, 'offset': offset}
        if model_type:
            params['model_type'] = model_type
        if mode:
            params['mode'] = mode
            
        response = requests.get(
            f"{self.base_url}/api/preservation/models",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_model(self, model_type: str, version: str, 
                  mode: Optional[str] = None) -> Dict:
        """Get specific model details"""
        url = f"{self.base_url}/api/preservation/models/{model_type}/{version}"
        params = {}
        if mode:
            params['mode'] = mode
            
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def rollback_model(self, model_type: str, target_version: str,
                      mode: Optional[str] = None, 
                      reason: Optional[str] = None) -> Dict:
        """Rollback to a previous model version"""
        data = {
            'model_type': model_type,
            'target_version': target_version
        }
        if mode:
            data['mode'] = mode
        if reason:
            data['reason'] = reason
            
        response = requests.post(
            f"{self.base_url}/api/preservation/rollback",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def delete_model_version(self, model_type: str, version: str,
                           mode: Optional[str] = None) -> Dict:
        """Delete a specific model version (admin only)"""
        url = f"{self.base_url}/api/preservation/models/{model_type}/{version}"
        params = {}
        if mode:
            params['mode'] = mode
            
        response = requests.delete(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Get system health and statistics"""
        response = requests.get(
            f"{self.base_url}/api/preservation/health",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == "__main__":
    client = ModelPreservationClient(
        base_url="https://your-domain.com",
        api_key="your-api-key"
    )
    
    # Check system health
    health = client.health_check()
    print(f"System status: {health['status']}")
    
    # List DQN agent models
    models = client.list_models(model_type="dqn_agent", limit=10)
    print(f"Found {len(models['models'])} DQN agent models")
    
    # Get specific model
    if models['models']:
        first_model = models['models'][0]
        model_details = client.get_model(
            first_model['model_type'], 
            first_model['version']
        )
        print(f"Model size: {model_details['model']['size_bytes']} bytes")
```

### JavaScript Client Example

```javascript
class ModelPreservationClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async listModels(options = {}) {
        const params = new URLSearchParams();
        if (options.limit) params.append('limit', options.limit);
        if (options.offset) params.append('offset', options.offset);
        if (options.model_type) params.append('model_type', options.model_type);
        if (options.mode) params.append('mode', options.mode);
        
        const response = await fetch(
            `${this.baseUrl}/api/preservation/models?${params}`,
            { headers: this.headers }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async getModel(modelType, version, mode = null) {
        const params = new URLSearchParams();
        if (mode) params.append('mode', mode);
        
        const response = await fetch(
            `${this.baseUrl}/api/preservation/models/${modelType}/${version}?${params}`,
            { headers: this.headers }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async rollbackModel(modelType, targetVersion, options = {}) {
        const data = {
            model_type: modelType,
            target_version: targetVersion,
            ...options
        };
        
        const response = await fetch(
            `${this.baseUrl}/api/preservation/rollback`,
            {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(data)
            }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async healthCheck() {
        const response = await fetch(
            `${this.baseUrl}/api/preservation/health`,
            { headers: this.headers }
        );
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
}

// Usage example
const client = new ModelPreservationClient(
    'https://your-domain.com',
    'your-api-key'
);

// Check system health
client.healthCheck()
    .then(health => console.log('System status:', health.status))
    .catch(error => console.error('Health check failed:', error));

// List models
client.listModels({ model_type: 'dqn_agent', limit: 10 })
    .then(response => console.log('Models:', response.models))
    .catch(error => console.error('Failed to list models:', error));
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Read operations**: 100 requests per minute per user
- **Write operations**: 20 requests per minute per user
- **Admin operations**: 10 requests per minute per user

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642252800
```

## Webhooks

The system can be configured to send webhooks for important events:

- Model rollbacks
- Model deletions
- System health changes
- Performance threshold breaches

Configure webhooks in your system configuration or through the admin panel.

---

For more information about the Model Preservation System, see the [User Guide](user_guide.md) and [Troubleshooting Guide](troubleshooting.md).