# Model Preservation System - Architecture Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Database Schema](#database-schema)
- [Cache Hierarchy](#cache-hierarchy)
- [Integration Points](#integration-points)
- [Security Architecture](#security-architecture)
- [Scalability Considerations](#scalability-considerations)

## System Overview

The Model Preservation System is a comprehensive solution for managing machine learning model lifecycles in the Shyvr RLTE system. It provides versioning, storage, caching, and monitoring capabilities across multiple environments.

```mermaid
graph TB
    subgraph "User Interfaces"
        UI[Web Dashboard]
        API[REST API]
        CLI[CLI Tools]
    end
    
    subgraph "Model Preservation Core"
        MGR[Preservation Manager]
        VER[Version Controller]
        CACHE[Cache Manager]
        MON[Monitoring System]
    end
    
    subgraph "Storage Layer"
        GCS[Google Cloud Storage]
        DB[(PostgreSQL Database)]
        REDIS[(Redis Cache)]
    end
    
    subgraph "ML/RL Systems"
        ML[ML Analysis System]
        RL[RL Agent System]
        MODES[Mode Manager]
    end
    
    UI --> API
    API --> MGR
    CLI --> MGR
    
    MGR --> VER
    MGR --> CACHE
    MGR --> MON
    
    VER --> DB
    CACHE --> REDIS
    MGR --> GCS
    
    ML --> MGR
    RL --> MGR
    MODES --> MGR
    
    MON --> DB
    MON --> GCS
```

### Key Components

- **Preservation Manager**: Central orchestrator for all model operations
- **Version Controller**: Handles semantic versioning and branching
- **Cache Manager**: Multi-level caching for performance optimization
- **Storage Handlers**: Abstracted storage operations for GCS and database
- **Monitoring System**: Performance tracking and health monitoring
- **API Layer**: RESTful interface for external integrations

## Component Architecture

### Core Components Interaction

```mermaid
graph LR
    subgraph "Preservation Manager"
        PM[Preservation Manager]
        
        subgraph "Handlers"
            GCS_H[GCS Handler]
            DB_H[Database Handler]
            CACHE_H[Cache Handler]
        end
        
        subgraph "Controllers"
            VER_C[Version Controller]
            BR_C[Branch Controller]
            TAG_C[Tag Controller]
        end
    end
    
    subgraph "External Systems"
        GCS[(Google Cloud Storage)]
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis)]
    end
    
    PM --> GCS_H
    PM --> DB_H
    PM --> CACHE_H
    PM --> VER_C
    PM --> BR_C
    PM --> TAG_C
    
    GCS_H --> GCS
    DB_H --> POSTGRES
    CACHE_H --> REDIS
    
    VER_C --> DB_H
    BR_C --> DB_H
    TAG_C --> DB_H
```

### Class Hierarchy

```mermaid
classDiagram
    class BasePreservation {
        +logger: Logger
        +config: Dict
        +initialize() async
        +health_check() async
    }
    
    class ModelPreservationManager {
        +gcs_handler: GCSHandler
        +db_handler: DatabaseHandler
        +cache_handler: CacheHandler
        +version_controller: VersionController
        +save_model(data, type, metadata) async
        +load_model(type, version) async
        +list_models() async
        +rollback_model(type, target_version) async
    }
    
    class GCSHandler {
        +client: storage.Client
        +bucket: Bucket
        +upload_model(model_id, data) async
        +download_model(model_id) async
        +delete_model(model_id) async
    }
    
    class DatabaseHandler {
        +pool: asyncpg.Pool
        +save_metadata(model_info) async
        +load_metadata(model_id) async
        +list_models(filters) async
    }
    
    class CacheHandler {
        +redis_client: Redis
        +memory_cache: Dict
        +get_cached_model(key) async
        +set_cached_model(key, data) async
        +clear_cache(pattern) async
    }
    
    class VersionController {
        +get_next_version(model_type) async
        +validate_version(version) bool
        +create_branch(source, target) async
        +merge_branch(source, target) async
    }
    
    BasePreservation <|-- ModelPreservationManager
    BasePreservation <|-- GCSHandler
    BasePreservation <|-- DatabaseHandler
    BasePreservation <|-- CacheHandler
    BasePreservation <|-- VersionController
    
    ModelPreservationManager --> GCSHandler
    ModelPreservationManager --> DatabaseHandler
    ModelPreservationManager --> CacheHandler
    ModelPreservationManager --> VersionController
```

## Data Flow

### Model Save Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Manager
    participant VersionCtrl
    participant Cache
    participant Database
    participant GCS
    
    Client->>API: POST /save_model
    API->>Manager: save_model(data, type, metadata)
    
    Manager->>VersionCtrl: get_next_version(type)
    VersionCtrl->>Database: SELECT max_version
    Database-->>VersionCtrl: current_version
    VersionCtrl-->>Manager: next_version
    
    Manager->>GCS: upload_model(model_id, data)
    GCS-->>Manager: upload_success, checksum
    
    Manager->>Database: save_metadata(model_info)
    Database-->>Manager: metadata_saved
    
    Manager->>Cache: cache_model(model_id, data)
    Cache-->>Manager: cached
    
    Manager-->>API: version_info
    API-->>Client: 201 Created
```

### Model Load Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Manager
    participant Cache
    participant Database
    participant GCS
    
    Client->>API: GET /models/{type}/{version}
    API->>Manager: load_model(type, version)
    
    Manager->>Cache: get_cached_model(model_id)
    alt Cache Hit
        Cache-->>Manager: cached_data
        Manager-->>API: model_data
    else Cache Miss
        Cache-->>Manager: None
        
        Manager->>Database: load_metadata(model_id)
        Database-->>Manager: metadata
        
        Manager->>GCS: download_model(model_id)
        GCS-->>Manager: model_data
        
        Manager->>Cache: cache_model(model_id, data)
        Manager-->>API: model_data
    end
    
    API-->>Client: 200 OK + model_data
```

### Rollback Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Manager
    participant Database
    participant GCS
    participant Cache
    participant MLSystem
    
    Client->>API: POST /rollback
    API->>Manager: rollback_model(type, target_version)
    
    Manager->>Database: verify_target_version(type, version)
    Database-->>Manager: version_exists
    
    alt Version Exists
        Manager->>GCS: download_model(target_model_id)
        GCS-->>Manager: target_model_data
        
        Manager->>Database: update_active_version(type, target_version)
        Database-->>Manager: updated
        
        Manager->>Cache: clear_cache(type)
        Manager->>Cache: cache_model(target_model_id, data)
        
        Manager->>MLSystem: notify_model_change(type, target_version)
        MLSystem-->>Manager: acknowledged
        
        Manager-->>API: rollback_success
        API-->>Client: 200 OK
    else Version Not Found
        Manager-->>API: version_not_found_error
        API-->>Client: 404 Not Found
    end
```

## Database Schema

### Core Tables Structure

```mermaid
erDiagram
    MODEL_VERSIONS {
        string model_id PK
        string model_type
        string version
        string mode
        string branch
        string state
        timestamp created_at
        timestamp updated_at
        string checksum
        integer size_bytes
        json metadata
        string created_by
        text description
    }
    
    MODEL_TAGS {
        string tag_id PK
        string model_id FK
        string tag_name
        timestamp created_at
        string created_by
    }
    
    MODEL_BRANCHES {
        string branch_id PK
        string model_type
        string branch_name
        string source_version
        string source_branch
        timestamp created_at
        string created_by
        text description
        boolean active
    }
    
    MODEL_PERFORMANCE {
        string performance_id PK
        string model_id FK
        timestamp recorded_at
        json metrics
        string environment
        string test_dataset
    }
    
    MODEL_CACHE_STATS {
        string cache_id PK
        string model_id FK
        integer hit_count
        integer miss_count
        timestamp last_accessed
        integer access_frequency
    }
    
    MODEL_VERSIONS ||--o{ MODEL_TAGS : has
    MODEL_VERSIONS ||--o{ MODEL_PERFORMANCE : tracks
    MODEL_VERSIONS ||--o{ MODEL_CACHE_STATS : cached
    MODEL_BRANCHES ||--o{ MODEL_VERSIONS : contains
```

### Schema Details

```sql
-- Model versions table
CREATE TABLE model_versions (
    model_id VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    mode VARCHAR(50) DEFAULT 'default',
    branch VARCHAR(100) DEFAULT 'main',
    state VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum VARCHAR(64),
    size_bytes BIGINT,
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(100),
    description TEXT,
    
    UNIQUE(model_type, version, mode, branch)
);

-- Indexes for efficient queries
CREATE INDEX idx_model_versions_type_version ON model_versions(model_type, version);
CREATE INDEX idx_model_versions_created_at ON model_versions(created_at DESC);
CREATE INDEX idx_model_versions_branch ON model_versions(branch);
CREATE INDEX idx_model_versions_state ON model_versions(state);
CREATE INDEX idx_model_versions_metadata ON model_versions USING GIN(metadata);
```

## Cache Hierarchy

The system implements a multi-level caching strategy for optimal performance:

```mermaid
graph TD
    subgraph "Cache Levels"
        L1[L1: In-Memory Cache<br/>100MB, 1min TTL]
        L2[L2: Redis Cache<br/>1GB, 1hour TTL]
        L3[L3: Disk Cache<br/>10GB, 24hour TTL]
        L4[L4: GCS Storage<br/>Unlimited, Persistent]
    end
    
    subgraph "Cache Policies"
        LRU[LRU Eviction]
        TTL[TTL Expiration]
        SIZE[Size-based Eviction]
    end
    
    REQUEST[Model Request] --> L1
    L1 -->|Miss| L2
    L2 -->|Miss| L3
    L3 -->|Miss| L4
    
    L1 --> LRU
    L2 --> TTL
    L3 --> SIZE
    
    L4 -->|Populate| L3
    L3 -->|Populate| L2
    L2 -->|Populate| L1
```

### Cache Implementation Details

```mermaid
flowchart LR
    subgraph "Cache Manager"
        CM[Cache Manager]
        
        subgraph "Cache Backends"
            MC[Memory Cache]
            RC[Redis Cache]
            DC[Disk Cache]
        end
        
        subgraph "Cache Strategies"
            WT[Write-Through]
            WB[Write-Behind]
            LA[Lazy Loading]
        end
    end
    
    CM --> MC
    CM --> RC
    CM --> DC
    
    CM --> WT
    CM --> WB
    CM --> LA
    
    MC -->|Fast Access| APP[Application]
    RC -->|Distributed| APP
    DC -->|Persistent| APP
```

## Integration Points

### ML/RL System Integration

```mermaid
graph TB
    subgraph "Model Preservation"
        MP[Model Preservation Manager]
        API[Preservation API]
    end
    
    subgraph "ML Analysis System"
        ML_MGR[Model Manager]
        ML_TRAIN[Training Pipeline]
        ML_PRED[Prediction Engine]
    end
    
    subgraph "RL Agent System"
        RL_AGENT[DQN Agent]
        RL_TRAIN[Training Pipeline]
        RL_ENV[Trading Environment]
    end
    
    subgraph "Mode Management"
        MODE_MGR[Mode Manager]
        SIM_MODE[Simulation Mode]
        LIVE_MODE[Live Mode]
        ANALYSIS_MODE[Analysis Mode]
    end
    
    ML_MGR <--> MP
    ML_TRAIN --> MP
    ML_PRED --> MP
    
    RL_AGENT <--> MP
    RL_TRAIN --> MP
    RL_ENV --> MP
    
    MODE_MGR --> MP
    SIM_MODE --> MP
    LIVE_MODE --> MP
    ANALYSIS_MODE --> MP
    
    MP --> API
```

### External System Interfaces

```mermaid
graph LR
    subgraph "Shyvr RLTE Core"
        PRESERVE[Model Preservation]
        DASHBOARD[Dashboard]
        MONITOR[Monitoring]
    end
    
    subgraph "External Storage"
        GCS[Google Cloud Storage]
        POSTGRES[PostgreSQL]
        REDIS[Redis]
    end
    
    subgraph "External Services"
        GRAFANA[Grafana]
        PROMETHEUS[Prometheus]
        ALERTS[Alert Manager]
    end
    
    subgraph "Client Access"
        WEB[Web UI]
        API_CLIENT[API Clients]
        CLI_TOOL[CLI Tools]
    end
    
    PRESERVE --> GCS
    PRESERVE --> POSTGRES
    PRESERVE --> REDIS
    
    PRESERVE --> MONITOR
    MONITOR --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    PROMETHEUS --> ALERTS
    
    DASHBOARD --> PRESERVE
    WEB --> DASHBOARD
    API_CLIENT --> PRESERVE
    CLI_TOOL --> PRESERVE
```

## Security Architecture

### Authentication and Authorization Flow

```mermaid
sequenceDiagram
    participant User
    participant API_Gateway
    participant Auth_Service
    participant Preservation_API
    participant Database
    
    User->>API_Gateway: Request with Token
    API_Gateway->>Auth_Service: Validate Token
    Auth_Service->>Database: Check User Permissions
    Database-->>Auth_Service: User Role & Permissions
    Auth_Service-->>API_Gateway: Token Valid + Permissions
    
    alt Authorized
        API_Gateway->>Preservation_API: Forward Request
        Preservation_API->>Database: Execute Operation
        Database-->>Preservation_API: Result
        Preservation_API-->>API_Gateway: Response
        API_Gateway-->>User: 200 OK + Data
    else Unauthorized
        API_Gateway-->>User: 403 Forbidden
    end
```

### Security Layers

```mermaid
graph TB
    subgraph "Security Layers"
        subgraph "L1: Network Security"
            HTTPS[HTTPS/TLS 1.3]
            VPC[VPC Network]
            FIREWALL[Firewall Rules]
        end
        
        subgraph "L2: Authentication"
            JWT[JWT Tokens]
            OAUTH[OAuth 2.0]
            MFA[Multi-Factor Auth]
        end
        
        subgraph "L3: Authorization"
            RBAC[Role-Based Access]
            ABAC[Attribute-Based Access]
            API_KEYS[API Key Management]
        end
        
        subgraph "L4: Data Security"
            ENCRYPT[Encryption at Rest]
            TRANSIT[Encryption in Transit]
            CHECKSUM[Data Integrity Checks]
        end
        
        subgraph "L5: Audit & Monitoring"
            LOGGING[Audit Logging]
            MONITORING[Security Monitoring]
            ALERTS[Security Alerts]
        end
    end
    
    USER[Users] --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
```

## Scalability Considerations

### Horizontal Scaling Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Application Load Balancer]
    end
    
    subgraph "API Tier (Auto-scaling)"
        API1[API Instance 1]
        API2[API Instance 2]
        API3[API Instance N...]
    end
    
    subgraph "Service Tier (Auto-scaling)"
        PRESERVE1[Preservation Service 1]
        PRESERVE2[Preservation Service 2]
        PRESERVE3[Preservation Service N...]
    end
    
    subgraph "Storage Tier (Managed)"
        GCS[Google Cloud Storage<br/>Auto-scaling]
        DB_CLUSTER[PostgreSQL Cluster<br/>Read Replicas]
        REDIS_CLUSTER[Redis Cluster<br/>Sharded]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> PRESERVE1
    API2 --> PRESERVE2
    API3 --> PRESERVE3
    
    PRESERVE1 --> GCS
    PRESERVE2 --> GCS
    PRESERVE3 --> GCS
    
    PRESERVE1 --> DB_CLUSTER
    PRESERVE2 --> DB_CLUSTER
    PRESERVE3 --> DB_CLUSTER
    
    PRESERVE1 --> REDIS_CLUSTER
    PRESERVE2 --> REDIS_CLUSTER
    PRESERVE3 --> REDIS_CLUSTER
```

### Performance Optimization Patterns

```mermaid
graph LR
    subgraph "Optimization Strategies"
        subgraph "Caching Patterns"
            CACHE_ASIDE[Cache-Aside]
            WRITE_THROUGH[Write-Through]
            WRITE_BEHIND[Write-Behind]
        end
        
        subgraph "Async Patterns"
            ASYNC_UPLOAD[Async Upload]
            BACKGROUND_JOBS[Background Jobs]
            STREAMING[Streaming Transfer]
        end
        
        subgraph "Data Patterns"
            COMPRESSION[Data Compression]
            PARTITIONING[Data Partitioning]
            INDEXING[Smart Indexing]
        end
        
        subgraph "Connection Patterns"
            POOLING[Connection Pooling]
            MULTIPLEXING[Request Multiplexing]
            KEEPALIVE[Keep-Alive]
        end
    end
    
    PERFORMANCE[Performance Goals] --> CACHE_ASIDE
    PERFORMANCE --> ASYNC_UPLOAD
    PERFORMANCE --> COMPRESSION
    PERFORMANCE --> POOLING
```

### Monitoring and Observability

```mermaid
graph TB
    subgraph "Observability Stack"
        subgraph "Metrics Collection"
            PROM[Prometheus]
            GRAFANA[Grafana]
            CUSTOM[Custom Metrics]
        end
        
        subgraph "Logging"
            STRUCT_LOG[Structured Logging]
            LOG_AGG[Log Aggregation]
            LOG_SEARCH[Log Search]
        end
        
        subgraph "Tracing"
            JAEGER[Jaeger Tracing]
            SPAN[Span Collection]
            TRACE_VIZ[Trace Visualization]
        end
        
        subgraph "Alerting"
            ALERT_MGR[Alert Manager]
            NOTIFICATION[Notifications]
            ESCALATION[Escalation Rules]
        end
    end
    
    subgraph "Model Preservation System"
        PRESERVE_SVC[Preservation Services]
        DB_METRICS[(Database Metrics)]
        CACHE_METRICS[(Cache Metrics)]
        STORAGE_METRICS[(Storage Metrics)]
    end
    
    PRESERVE_SVC --> PROM
    PRESERVE_SVC --> STRUCT_LOG
    PRESERVE_SVC --> JAEGER
    
    DB_METRICS --> PROM
    CACHE_METRICS --> PROM
    STORAGE_METRICS --> PROM
    
    PROM --> GRAFANA
    PROM --> ALERT_MGR
    
    STRUCT_LOG --> LOG_AGG
    LOG_AGG --> LOG_SEARCH
    
    JAEGER --> SPAN
    SPAN --> TRACE_VIZ
    
    ALERT_MGR --> NOTIFICATION
    NOTIFICATION --> ESCALATION
```

This architecture documentation provides a comprehensive view of the Model Preservation System's design, from high-level system architecture to detailed component interactions, data flows, and scalability considerations. The mermaid diagrams offer visual representations that make the complex system easier to understand and maintain.