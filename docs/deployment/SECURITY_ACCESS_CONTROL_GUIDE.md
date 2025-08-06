# Security and Access Control Configuration Guide

## Table of Contents
1. [Overview](#overview)
2. [Security Architecture](#security-architecture)
3. [Authentication and Authorization](#authentication-and-authorization)
4. [GCP IAM Configuration](#gcp-iam-configuration)
5. [Secret Management](#secret-management)
6. [Network Security](#network-security)
7. [API Security](#api-security)
8. [Data Protection](#data-protection)
9. [Container Security](#container-security)
10. [Audit Logging](#audit-logging)
11. [Compliance and Governance](#compliance-and-governance)
12. [Security Monitoring](#security-monitoring)
13. [Incident Response](#incident-response)
14. [Security Best Practices](#security-best-practices)

## Overview

This guide provides comprehensive security and access control configuration for the transformer-enabled RLTE system. The security framework follows defense-in-depth principles and implements enterprise-grade security controls for production cryptocurrency trading systems.

### Security Principles
1. **Zero Trust Architecture**: Never trust, always verify
2. **Principle of Least Privilege**: Minimal required access
3. **Defense in Depth**: Multiple security layers
4. **Security by Design**: Security integrated from the start
5. **Compliance First**: Regulatory compliance built-in

### Security Scope
- **Infrastructure Security**: GCP resources and network protection
- **Application Security**: Authentication, authorization, and data protection
- **API Security**: Rate limiting, input validation, and secure communication
- **Data Security**: Encryption at rest and in transit
- **Operational Security**: Monitoring, logging, and incident response

## Security Architecture

### Multi-Layer Security Model
```
┌─────────────────────────────────────────────────────────────┐
│                    External Layer                           │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Network Security                           │
│  │  • Cloud Load Balancer • WAF • DDoS Protection         │
│  │  • VPC • Private Subnets • Network Policies            │
│  │  ┌─────────────────────────────────────────────────────┤
│  │  │           Application Security                      │
│  │  │  • Authentication • Authorization • Rate Limiting   │
│  │  │  • Input Validation • Output Encoding              │
│  │  │  ┌─────────────────────────────────────────────────┤
│  │  │  │          Data Security                          │
│  │  │  │  • Encryption at Rest • Encryption in Transit  │
│  │  │  │  • Key Management • Data Classification         │
│  │  │  │  ┌─────────────────────────────────────────────┤
│  │  │  │  │      Infrastructure Security              │
│  │  │  │  │  • IAM • Service Accounts • RBAC           │
│  │  │  │  │  • Container Security • Runtime Protection  │
│  │  │  │  └─────────────────────────────────────────────┘
│  │  │  └─────────────────────────────────────────────────┘
│  │  └─────────────────────────────────────────────────────┘
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### Security Components
```yaml
security_components:
  # Identity and Access Management
  iam_system:
    provider: "GCP IAM"
    rbac_enabled: true
    mfa_required: true
    service_accounts: true
  
  # Network Security
  network_security:
    vpc_isolation: true
    private_subnets: true
    network_policies: true
    load_balancer_ssl: true
  
  # Application Security
  application_security:
    authentication: "JWT + API Keys"
    authorization: "RBAC"
    rate_limiting: true
    input_validation: true
    csrf_protection: true
  
  # Data Security
  data_security:
    encryption_at_rest: "AES-256"
    encryption_in_transit: "TLS 1.3"
    key_management: "GCP KMS"
    data_classification: true
  
  # Monitoring and Compliance
  security_monitoring:
    audit_logging: true
    security_scanning: true
    vulnerability_assessment: true
    compliance_monitoring: true
```

## Authentication and Authorization

### JWT-Based Authentication

#### 1. JWT Configuration
```yaml
# JWT authentication configuration
jwt_config:
  # Algorithm and security
  algorithm: "HS256"
  secret_key: "${JWT_SECRET}"  # From Secret Manager
  
  # Token lifetimes (production settings)
  access_token_expire_minutes: 720    # 12 hours
  refresh_token_expire_days: 30       # 30 days
  
  # Security settings
  require_https: true
  secure_cookies: true
  same_site: "strict"
  
  # Claims configuration
  issuer: "shyvr-rlte-system"
  audience: "shyvr-rlte-api"
  
  # Additional security
  not_before_grace_seconds: 30
  clock_skew_seconds: 60
```

#### 2. Authentication Implementation
```python
# JWT authentication service
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class JWTAuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_access_token(self, data: dict, 
                          expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token with security claims."""
        to_encode = data.copy()
        
        # Set expiration
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=720)
        
        # Add standard claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "nbf": datetime.now(timezone.utc) - timedelta(seconds=30),
            "iss": "shyvr-rlte-system",
            "aud": "shyvr-rlte-api",
            "sub": data.get("user_id"),
            "jti": generate_token_id()  # Unique token ID
        })
        
        encoded_jwt = jwt.encode(
            to_encode, self.secret_key, algorithm=self.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm],
                audience="shyvr-rlte-api",
                issuer="shyvr-rlte-system",
                options={
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_signature": True
                }
            )
            
            # Additional validation
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
            # Check token blacklist (if implemented)
            if await self.is_token_blacklisted(payload.get("jti")):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

### Role-Based Access Control (RBAC)

#### 1. Permission System
```python
# RBAC permission system
from enum import Enum
from typing import List, Set

class Permission(Enum):
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # Trading permissions
    TRADING_VIEW = "trading:view"
    TRADING_EXECUTE = "trading:execute"
    TRADING_ADMIN = "trading:admin"
    
    # Model permissions
    MODEL_VIEW = "model:view"
    MODEL_DEPLOY = "model:deploy"
    MODEL_ADMIN = "model:admin"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"

class Role(Enum):
    # Basic roles
    VIEWER = "viewer"
    TRADER = "trader"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

# Role-permission mapping
ROLE_PERMISSIONS = {
    Role.VIEWER: {
        Permission.SYSTEM_READ,
        Permission.TRADING_VIEW,
        Permission.MODEL_VIEW,
        Permission.DATA_READ,
        Permission.ANALYTICS_VIEW
    },
    
    Role.TRADER: {
        Permission.SYSTEM_READ,
        Permission.TRADING_VIEW,
        Permission.TRADING_EXECUTE,
        Permission.MODEL_VIEW,
        Permission.DATA_READ,
        Permission.ANALYTICS_VIEW
    },
    
    Role.ANALYST: {
        Permission.SYSTEM_READ,
        Permission.TRADING_VIEW,
        Permission.MODEL_VIEW,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT
    },
    
    Role.DEVELOPER: {
        Permission.SYSTEM_READ,
        Permission.SYSTEM_WRITE,
        Permission.MODEL_VIEW,
        Permission.MODEL_DEPLOY,
        Permission.DATA_READ,
        Permission.DATA_WRITE
    },
    
    Role.ADMIN: {
        Permission.SYSTEM_READ,
        Permission.SYSTEM_WRITE,
        Permission.TRADING_VIEW,
        Permission.TRADING_EXECUTE,
        Permission.TRADING_ADMIN,
        Permission.MODEL_VIEW,
        Permission.MODEL_DEPLOY,
        Permission.MODEL_ADMIN,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_EXPORT,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT
    },
    
    Role.SUPER_ADMIN: set(Permission)  # All permissions
}
```

#### 2. Authorization Decorators
```python
# Authorization decorators and middleware
from functools import wraps
from fastapi import Request, HTTPException

def require_permissions(*required_permissions: Permission):
    """Decorator to enforce permission requirements."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from JWT token
            current_user = get_current_user()  # From JWT verification
            user_permissions = get_user_permissions(current_user.user_id)
            
            # Check permissions
            for permission in required_permissions:
                if permission not in user_permissions:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required permission: {permission.value}"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(*required_roles: Role):
    """Decorator to enforce role requirements."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = get_current_user()
            user_roles = get_user_roles(current_user.user_id)
            
            if not any(role in user_roles for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient role privileges"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@app.post("/api/trading/execute")
@require_permissions(Permission.TRADING_EXECUTE)
async def execute_trade(trade_data: TradeRequest):
    """Execute trading operation."""
    pass

@app.post("/api/models/deploy")
@require_role(Role.DEVELOPER, Role.ADMIN, Role.SUPER_ADMIN)
async def deploy_model(model_data: ModelDeployment):
    """Deploy new model version."""
    pass
```

### API Key Management

#### 1. API Key Authentication
```python
# API key authentication system
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyService:
    def __init__(self, db_session):
        self.db = db_session
    
    def create_api_key(self, user_id: str, name: str, 
                      expires_days: int = 90, 
                      permissions: List[Permission] = None) -> str:
        """Create new API key with specified permissions."""
        # Generate secure API key
        raw_key = secrets.token_urlsafe(32)
        key_prefix = "rlte_"
        full_key = f"{key_prefix}{raw_key}"
        
        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # Store in database
        api_key_record = {
            "key_hash": key_hash,
            "key_prefix": key_prefix + raw_key[:8],  # For identification
            "user_id": user_id,
            "name": name,
            "permissions": [p.value for p in (permissions or [])],
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=expires_days),
            "is_active": True,
            "last_used": None,
            "usage_count": 0
        }
        
        self.db.api_keys.insert_one(api_key_record)
        
        # Return full key (only shown once)
        return full_key
    
    async def verify_api_key(self, api_key: str) -> Optional[dict]:
        """Verify API key and return associated permissions."""
        if not api_key or not api_key.startswith("rlte_"):
            return None
        
        # Hash provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Find in database
        key_record = await self.db.api_keys.find_one({
            "key_hash": key_hash,
            "is_active": True,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if not key_record:
            return None
        
        # Update usage statistics
        await self.db.api_keys.update_one(
            {"_id": key_record["_id"]},
            {
                "$set": {"last_used": datetime.utcnow()},
                "$inc": {"usage_count": 1}
            }
        )
        
        return {
            "user_id": key_record["user_id"],
            "permissions": key_record["permissions"],
            "key_name": key_record["name"]
        }
```

## GCP IAM Configuration

### Service Account Setup

#### 1. Core Service Accounts
```bash
# Create service accounts for different components
PROJECT_ID="your-project-id"

# Main application service account
gcloud iam service-accounts create rlte-app-service \
  --description="Main RLTE application service account" \
  --display-name="RLTE Application Service"

# Monitoring service account
gcloud iam service-accounts create rlte-monitoring-service \
  --description="RLTE monitoring and logging service account" \
  --display-name="RLTE Monitoring Service"

# CI/CD service account
gcloud iam service-accounts create rlte-cicd-service \
  --description="RLTE CI/CD pipeline service account" \
  --display-name="RLTE CI/CD Service"

# Backup service account
gcloud iam service-accounts create rlte-backup-service \
  --description="RLTE backup and disaster recovery service account" \
  --display-name="RLTE Backup Service"
```

#### 2. Role Assignments
```bash
# Application service account permissions
APP_SA="rlte-app-service@${PROJECT_ID}.iam.gserviceaccount.com"

# Database access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="roles/cloudsql.client"

# Secret access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="roles/secretmanager.secretAccessor"

# Storage access (for model persistence)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="roles/storage.objectViewer"

# Monitoring access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="roles/logging.logWriter"

# Monitoring service account permissions
MONITORING_SA="rlte-monitoring-service@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${MONITORING_SA}" \
  --role="roles/monitoring.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${MONITORING_SA}" \
  --role="roles/logging.viewer"
```

#### 3. Custom IAM Roles
```bash
# Create custom role for RLTE operations
cat << EOF > rlte-operator-role.yaml
title: "RLTE Operator"
description: "Custom role for RLTE system operations"
stage: "GA"
includedPermissions:
- cloudsql.instances.connect
- secretmanager.versions.access
- storage.objects.get
- storage.objects.list
- monitoring.metricDescriptors.create
- monitoring.metricDescriptors.list
- monitoring.timeSeries.create
- logging.entries.create
- compute.instances.get
- run.services.get
- run.services.list
EOF

gcloud iam roles create rlteOperator \
  --project=$PROJECT_ID \
  --file=rlte-operator-role.yaml

# Assign custom role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${APP_SA}" \
  --role="projects/${PROJECT_ID}/roles/rlteOperator"
```

### IAM Policy Management

#### 1. Environment-Specific IAM
```yaml
# IAM configuration by environment
iam_policies:
  production:
    # Restrictive production policies
    enable_audit_logs: true
    require_mfa: true
    session_duration: "12h"
    allowed_ip_ranges:
      - "10.0.0.0/8"      # VPC networks
      - "34.96.0.0/20"    # Cloud Run IP range
    
    service_account_permissions:
      rlte-app-service:
        - "roles/cloudsql.client"
        - "roles/secretmanager.secretAccessor" 
        - "roles/storage.objectViewer"
        - "projects/PROJECT_ID/roles/rlteOperator"
  
  staging:
    # More permissive for testing
    enable_audit_logs: true
    require_mfa: false
    session_duration: "24h"
    allowed_ip_ranges:
      - "0.0.0.0/0"       # Allow all for staging
    
    service_account_permissions:
      rlte-staging-service:
        - "roles/cloudsql.client"
        - "roles/secretmanager.secretAccessor"
        - "roles/storage.objectAdmin"
        - "roles/monitoring.editor"
```

## Secret Management

### GCP Secret Manager Integration

#### 1. Secret Organization
```bash
# Organize secrets by environment and type
PREFIX="rlte"
ENV="production"

# Application secrets
gcloud secrets create "${PREFIX}-${ENV}-jwt-secret" \
  --labels=environment=${ENV},component=auth
  
gcloud secrets create "${PREFIX}-${ENV}-encryption-key" \
  --labels=environment=${ENV},component=encryption

# Database secrets  
gcloud secrets create "${PREFIX}-${ENV}-db-password" \
  --labels=environment=${ENV},component=database

gcloud secrets create "${PREFIX}-${ENV}-db-url" \
  --labels=environment=${ENV},component=database

# API secrets
gcloud secrets create "${PREFIX}-${ENV}-helius-api-key" \
  --labels=environment=${ENV},component=api

gcloud secrets create "${PREFIX}-${ENV}-birdeye-api-key" \
  --labels=environment=${ENV},component=api

# Notification secrets
gcloud secrets create "${PREFIX}-${ENV}-telegram-token" \
  --labels=environment=${ENV},component=notifications

gcloud secrets create "${PREFIX}-${ENV}-slack-webhook" \
  --labels=environment=${ENV},component=notifications
```

#### 2. Secret Access Policies
```bash
# Configure secret access policies
SECRET_NAME="${PREFIX}-${ENV}-jwt-secret"

# Allow application service account access
gcloud secrets add-iam-policy-binding $SECRET_NAME \
  --member="serviceAccount:rlte-app-service@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Allow CI/CD service account for deployments
gcloud secrets add-iam-policy-binding $SECRET_NAME \
  --member="serviceAccount:rlte-cicd-service@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deny access to default compute service account
gcloud secrets remove-iam-policy-binding $SECRET_NAME \
  --member="serviceAccount:${PROJECT_ID}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" || true
```

### Secret Rotation Strategy

#### 1. Automated Secret Rotation
```python
# Automated secret rotation system
import asyncio
from datetime import datetime, timedelta
from google.cloud import secretmanager
from typing import Dict, List

class SecretRotationManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        
        # Rotation schedules
        self.rotation_schedules = {
            "jwt-secret": 7,        # Weekly
            "api-keys": 30,         # Monthly  
            "db-password": 90,      # Quarterly
            "encryption-key": 365,  # Yearly
        }
    
    async def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotate a secret to a new value."""
        try:
            # Create new version
            parent = f"projects/{self.project_id}/secrets/{secret_name}"
            payload = {"data": new_value.encode("utf-8")}
            
            response = self.client.add_secret_version(
                request={"parent": parent, "payload": payload}
            )
            
            # Verify new version is accessible
            await asyncio.sleep(30)  # Wait for propagation
            
            # Test new secret
            if await self.test_secret_validity(secret_name, new_value):
                # Disable old versions (keep last 2 versions)
                await self.cleanup_old_versions(secret_name, keep_count=2)
                return True
            else:
                # Rollback by disabling the new version
                await self.disable_secret_version(response.name)
                return False
                
        except Exception as e:
            logger.error(f"Secret rotation failed for {secret_name}: {str(e)}")
            return False
    
    async def check_rotation_due(self) -> List[str]:
        """Check which secrets need rotation."""
        secrets_to_rotate = []
        
        for secret_type, days_interval in self.rotation_schedules.items():
            secrets = await self.list_secrets_by_type(secret_type)
            
            for secret_name in secrets:
                last_rotation = await self.get_last_rotation_date(secret_name)
                
                if (datetime.utcnow() - last_rotation).days >= days_interval:
                    secrets_to_rotate.append(secret_name)
        
        return secrets_to_rotate
    
    async def automated_rotation_cycle(self):
        """Run automated rotation cycle."""
        secrets_to_rotate = await self.check_rotation_due()
        
        for secret_name in secrets_to_rotate:
            new_value = await self.generate_new_secret_value(secret_name)
            
            success = await self.rotate_secret(secret_name, new_value)
            
            if success:
                logger.info(f"Successfully rotated secret: {secret_name}")
                await self.notify_rotation_success(secret_name)
            else:
                logger.error(f"Failed to rotate secret: {secret_name}")
                await self.notify_rotation_failure(secret_name)
```

## Network Security

### VPC and Network Configuration

#### 1. VPC Setup
```bash
# Create dedicated VPC for RLTE system
gcloud compute networks create rlte-vpc \
  --subnet-mode=custom \
  --description="Dedicated VPC for RLTE system"

# Create private subnet for application
gcloud compute networks subnets create rlte-app-subnet \
  --network=rlte-vpc \
  --range=10.1.0.0/24 \
  --region=us-central1 \
  --description="Private subnet for RLTE application"

# Create private subnet for database
gcloud compute networks subnets create rlte-db-subnet \
  --network=rlte-vpc \
  --range=10.1.1.0/24 \
  --region=us-central1 \
  --description="Private subnet for database"

# Create subnet for VPC connector
gcloud compute networks subnets create rlte-connector-subnet \
  --network=rlte-vpc \
  --range=10.1.2.0/28 \
  --region=us-central1 \
  --description="Subnet for VPC connector"
```

#### 2. Firewall Rules
```bash
# Deny all traffic by default
gcloud compute firewall-rules create rlte-deny-all \
  --network=rlte-vpc \
  --action=deny \
  --rules=all \
  --source-ranges=0.0.0.0/0 \
  --priority=65534 \
  --description="Deny all traffic by default"

# Allow HTTPS traffic to load balancer
gcloud compute firewall-rules create rlte-allow-https \
  --network=rlte-vpc \
  --action=allow \
  --rules=tcp:443 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=rlte-https \
  --priority=1000 \
  --description="Allow HTTPS traffic"

# Allow internal communication
gcloud compute firewall-rules create rlte-allow-internal \
  --network=rlte-vpc \
  --action=allow \
  --rules=all \
  --source-ranges=10.1.0.0/16 \
  --priority=1000 \
  --description="Allow internal VPC communication"

# Allow health checks
gcloud compute firewall-rules create rlte-allow-health-checks \
  --network=rlte-vpc \
  --action=allow \
  --rules=tcp:8080,tcp:8081 \
  --source-ranges=35.191.0.0/16,130.211.0.0/22 \
  --target-tags=rlte-health-check \
  --priority=1000 \
  --description="Allow Google health checks"
```

#### 3. VPC Connector Setup
```bash
# Create VPC connector for Cloud Run
gcloud compute networks vpc-access connectors create rlte-connector \
  --network=rlte-vpc \
  --region=us-central1 \
  --subnet=rlte-connector-subnet \
  --subnet-project=$PROJECT_ID \
  --min-instances=2 \
  --max-instances=10 \
  --machine-type=e2-micro \
  --description="VPC connector for RLTE Cloud Run service"
```

### SSL/TLS Configuration

#### 1. SSL Certificate Management
```bash
# Create managed SSL certificate
gcloud compute ssl-certificates create rlte-ssl-cert \
  --domains=api.shyvr.ai,rlte.shyvr.ai \
  --global \
  --description="SSL certificate for RLTE API"

# Create load balancer with SSL
gcloud compute url-maps create rlte-url-map \
  --default-service=rlte-backend-service \
  --description="URL map for RLTE system"

# Create HTTPS proxy
gcloud compute target-https-proxies create rlte-https-proxy \
  --url-map=rlte-url-map \
  --ssl-certificates=rlte-ssl-cert \
  --ssl-policy=rlte-ssl-policy

# Create SSL policy with strong security
gcloud compute ssl-policies create rlte-ssl-policy \
  --profile=RESTRICTED \
  --min-tls-version=1.2 \
  --description="Restrictive SSL policy for RLTE"
```

#### 2. Certificate Rotation
```yaml
# Automated certificate management
ssl_certificate_config:
  # Certificate lifecycle
  auto_renewal: true
  renewal_days_before_expiry: 30
  notification_days_before_expiry: 7
  
  # Security settings
  min_tls_version: "1.2"
  cipher_suites:
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
    - "TLS_AES_128_GCM_SHA256"
  
  # HSTS settings
  hsts_enabled: true
  hsts_max_age: 31536000  # 1 year
  hsts_include_subdomains: true
  hsts_preload: true
```

## API Security

### Rate Limiting and Throttling

#### 1. Rate Limiting Configuration
```python
# Advanced rate limiting system
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Rate limiting by user type
RATE_LIMITS = {
    "anonymous": "60/hour",           # Anonymous users
    "authenticated": "600/hour",      # Authenticated users
    "premium": "1200/hour",           # Premium users
    "admin": "unlimited",             # Admin users
    "api_key": "300/hour",           # API key users
}

# Endpoint-specific rate limits
ENDPOINT_LIMITS = {
    "/api/auth/login": "5/minute",          # Login attempts
    "/api/auth/register": "3/hour",         # Registration
    "/api/trading/execute": "100/hour",     # Trading execution
    "/api/models/predict": "1000/hour",     # Model predictions
    "/api/data/export": "10/hour",          # Data exports
}

class AdvancedRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.limiter = Limiter(key_func=self.get_rate_limit_key)
    
    def get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limiting key based on user type."""
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Check for authenticated user
        try:
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token:
                payload = jwt.decode(token, verify=False)  # Just for rate limiting
                user_id = payload.get("sub")
                user_type = payload.get("user_type", "authenticated")
                return f"user:{user_id}:{user_type}"
        except:
            pass
        
        # Fall back to IP-based limiting
        return f"ip:{get_remote_address(request)}"
    
    def get_rate_limit(self, request: Request) -> str:
        """Get rate limit based on user type and endpoint."""
        # Check endpoint-specific limits first
        endpoint_limit = ENDPOINT_LIMITS.get(request.url.path)
        if endpoint_limit:
            return endpoint_limit
        
        # Check user type limits
        key = self.get_rate_limit_key(request)
        if key.startswith("api_key:"):
            return RATE_LIMITS["api_key"]
        elif key.startswith("user:"):
            user_type = key.split(":")[-1]
            return RATE_LIMITS.get(user_type, RATE_LIMITS["authenticated"])
        else:
            return RATE_LIMITS["anonymous"]

# Apply rate limiting to FastAPI app
limiter = AdvancedRateLimiter(redis_client)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

#### 2. DDoS Protection
```python
# DDoS detection and mitigation
class DDoSProtection:
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Thresholds
        self.request_threshold = 1000    # Requests per minute
        self.ip_threshold = 100          # Requests per minute per IP
        self.error_threshold = 50        # Error responses per minute
    
    async def check_ddos_patterns(self, request: Request) -> bool:
        """Check for DDoS attack patterns."""
        client_ip = get_remote_address(request)
        current_minute = int(time.time() // 60)
        
        # Track requests per IP
        ip_key = f"ddos:ip:{client_ip}:{current_minute}"
        ip_requests = await self.redis.incr(ip_key)
        await self.redis.expire(ip_key, 60)
        
        # Track total requests
        total_key = f"ddos:total:{current_minute}"
        total_requests = await self.redis.incr(total_key)
        await self.redis.expire(total_key, 60)
        
        # Track error responses
        if hasattr(request.state, "error_response"):
            error_key = f"ddos:errors:{current_minute}"
            error_count = await self.redis.incr(error_key)
            await self.redis.expire(error_key, 60)
            
            if error_count > self.error_threshold:
                await self.trigger_ddos_alert("high_error_rate", {
                    "error_count": error_count,
                    "threshold": self.error_threshold
                })
        
        # Check thresholds
        if ip_requests > self.ip_threshold:
            await self.block_ip(client_ip, duration=300)  # 5 minutes
            return False
        
        if total_requests > self.request_threshold:
            await self.trigger_ddos_alert("high_request_volume", {
                "total_requests": total_requests,
                "threshold": self.request_threshold
            })
        
        return True
```

### Input Validation and Sanitization

#### 1. Request Validation
```python
# Comprehensive input validation
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import re

class SecureBaseModel(BaseModel):
    """Base model with security validations."""
    
    class Config:
        # Security settings
        validate_assignment = True
        extra = "forbid"  # Reject extra fields
        max_anystr_length = 1000  # Maximum string length
        
    @validator('*', pre=True)
    def prevent_xss(cls, v):
        """Prevent XSS attacks in string fields."""
        if isinstance(v, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', 'javascript:', 'data:']
            for char in dangerous_chars:
                if char in v.lower():
                    raise ValueError(f"Potentially dangerous input detected")
        return v

class TradingRequest(SecureBaseModel):
    """Secure trading request validation."""
    symbol: str = Field(..., regex=r'^[A-Z]{3,10}$')
    amount: float = Field(..., gt=0, le=1000000)
    order_type: str = Field(..., regex=r'^(market|limit|stop)$')
    price: Optional[float] = Field(None, gt=0, le=1000000)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate trading symbol."""
        if not re.match(r'^[A-Z]{3,10}$', v):
            raise ValueError("Invalid symbol format")
        return v

class ModelPredictionRequest(SecureBaseModel):
    """Secure model prediction request."""
    model_type: str = Field(..., regex=r'^(itransformer|patchtst|timesmixer|timesfm)$')
    data: List[float] = Field(..., min_items=1, max_items=1000)
    
    @validator('data')
    def validate_data_range(cls, v):
        """Validate data is within reasonable ranges."""
        for value in v:
            if not -1000000 <= value <= 1000000:
                raise ValueError("Data values out of acceptable range")
        return v
```

#### 2. SQL Injection Prevention
```python
# SQL injection prevention
from sqlalchemy import text
from sqlalchemy.orm import Session

class SecureDatabase:
    def __init__(self, session: Session):
        self.session = session
    
    def safe_query(self, query_template: str, parameters: dict):
        """Execute parameterized query safely."""
        # Validate parameters
        self.validate_parameters(parameters)
        
        # Use parameterized query
        stmt = text(query_template)
        result = self.session.execute(stmt, parameters)
        return result
    
    def validate_parameters(self, parameters: dict):
        """Validate query parameters."""
        for key, value in parameters.items():
            # Check parameter names
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid parameter name: {key}")
            
            # Check for SQL injection patterns
            if isinstance(value, str):
                dangerous_patterns = [
                    r'union\s+select',
                    r'drop\s+table',
                    r'delete\s+from',
                    r'insert\s+into',
                    r'update\s+set',
                    r'exec\s*\(',
                    r'--',
                    r'/\*.*\*/',
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, value.lower()):
                        raise ValueError("Potentially dangerous SQL detected")

# Usage example
@app.get("/api/trading/history")
@require_permissions(Permission.TRADING_VIEW)
async def get_trading_history(
    symbol: str = Field(..., regex=r'^[A-Z]{3,10}$'),
    limit: int = Field(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get trading history with secure database query."""
    secure_db = SecureDatabase(db)
    
    result = secure_db.safe_query(
        "SELECT * FROM trades WHERE symbol = :symbol ORDER BY created_at DESC LIMIT :limit",
        {"symbol": symbol, "limit": limit}
    )
    
    return result.fetchall()
```

## Data Protection

### Encryption at Rest

#### 1. Database Encryption
```yaml
# Cloud SQL encryption configuration
database_encryption:
  # Encryption at rest
  disk_encryption: "Google-managed"  # or "Customer-managed"
  
  # Customer-managed encryption key (optional)
  cmek_config:
    key_ring: "rlte-keyring"
    key_name: "rlte-db-key"
    location: "us-central1"
  
  # Backup encryption
  backup_encryption: true
  backup_retention_days: 30
  
  # Connection encryption
  require_ssl: true
  ssl_mode: "VERIFY_CA"
```

#### 2. Application-Level Encryption
```python
# Application-level encryption for sensitive data
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

class DataEncryption:
    def __init__(self, password: str):
        """Initialize encryption with password-derived key."""
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'salt_',  # Use proper random salt in production
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher_suite = Fernet(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception:
            raise ValueError("Failed to decrypt data")

# Usage for encrypting API keys and sensitive configuration
class SecureConfig:
    def __init__(self, encryption_key: str):
        self.encryptor = DataEncryption(encryption_key)
    
    def store_api_key(self, service: str, api_key: str):
        """Store encrypted API key."""
        encrypted_key = self.encryptor.encrypt_sensitive_data(api_key)
        # Store in database with encryption flag
        return {"service": service, "encrypted_key": encrypted_key, "encrypted": True}
    
    def retrieve_api_key(self, service: str) -> str:
        """Retrieve and decrypt API key."""
        # Get from database
        record = get_api_key_record(service)
        if record.get("encrypted"):
            return self.encryptor.decrypt_sensitive_data(record["encrypted_key"])
        else:
            return record["api_key"]
```

### Encryption in Transit

#### 1. TLS Configuration
```python
# TLS configuration for external API calls
import httpx
import ssl

class SecureHTTPClient:
    def __init__(self):
        # Create secure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Configure cipher suites
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        self.client = httpx.AsyncClient(
            verify=ssl_context,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def secure_request(self, method: str, url: str, **kwargs):
        """Make secure HTTP request with proper headers."""
        headers = kwargs.get('headers', {})
        
        # Add security headers
        headers.update({
            'User-Agent': 'RLTE-System/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
        })
        
        kwargs['headers'] = headers
        
        response = await self.client.request(method, url, **kwargs)
        
        # Validate response
        if response.status_code >= 400:
            logger.warning(f"HTTP error {response.status_code} for {url}")
        
        return response
```

## Container Security

### Docker Security Configuration

#### 1. Secure Dockerfile Practices
```dockerfile
# Security-focused Dockerfile configuration
FROM python:3.12-slim as base

# Create non-root user
RUN groupadd -r rlte && useradd -r -g rlte -d /app -s /sbin/nologin rlte

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up application directory with proper permissions
WORKDIR /app
COPY --chown=rlte:rlte requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with proper ownership
COPY --chown=rlte:rlte src/ ./src/
COPY --chown=rlte:rlte config/ ./config/

# Remove unnecessary packages
RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean

# Security configurations
USER rlte
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with security options
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### 2. Cloud Run Security Settings
```yaml
# Cloud Run security configuration
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rlte-secure
  annotations:
    run.googleapis.com/execution-environment: gen2
    run.googleapis.com/cpu-boost: "true"
spec:
  template:
    metadata:
      annotations:
        # Security annotations
        run.googleapis.com/vpc-access-connector: "rlte-connector"
        run.googleapis.com/vpc-access-egress: "private-ranges-only"
        
        # Container security
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/sandbox: gvisor
    spec:
      serviceAccountName: rlte-app-service
      containerConcurrency: 15
      timeoutSeconds: 4200
      
      containers:
      - image: us-central1-docker.pkg.dev/PROJECT_ID/shyvr-ai-prod/shyvr-rlte:latest
        
        # Resource limits
        resources:
          limits:
            memory: 8Gi
            cpu: 6
          requests:
            memory: 4Gi
            cpu: 3
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          runAsGroup: 1000
          readOnlyRootFilesystem: false  # Required for model caching
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
        
        # Environment variables (secrets from Secret Manager)
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: rlte-production-jwt-secret
              key: latest
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            httpHeaders:
            - name: X-Health-Check
              value: liveness
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
```

### Image Security Scanning

#### 1. Automated Security Scanning
```bash
# Container image security scanning
#!/bin/bash
# scripts/security_scan.sh

IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/shyvr-ai-prod/shyvr-rlte:latest"

echo "Running comprehensive security scan..."

# Vulnerability scanning with Trivy
echo "🔍 Scanning for vulnerabilities..."
trivy image \
  --security-checks vuln,config,secret \
  --severity HIGH,CRITICAL \
  --format json \
  --output vulnerability-report.json \
  $IMAGE_NAME

# Check for secrets in image
echo "🔐 Scanning for secrets..."
trivy image \
  --security-checks secret \
  --format json \
  --output secrets-report.json \
  $IMAGE_NAME

# Configuration audit
echo "⚙️  Auditing configuration..."
trivy image \
  --security-checks config \
  --format json \
  --output config-audit.json \
  $IMAGE_NAME

# Custom security checks
echo "🛡️  Running custom security checks..."
python scripts/custom_security_audit.py $IMAGE_NAME

echo "✅ Security scan complete"
```

## Audit Logging

### Comprehensive Audit Trail

#### 1. Security Event Logging
```python
# Comprehensive security audit logging
import structlog
from datetime import datetime
from enum import Enum

class SecurityEventType(Enum):
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    API_KEY_USAGE = "api_key_usage"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class SecurityAuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("security_audit")
    
    def log_security_event(self, event_type: SecurityEventType, 
                          user_id: str, details: dict):
        """Log security-related events."""
        self.logger.info(
            "security_event",
            event_type=event_type.value,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            source_ip=details.get("source_ip"),
            user_agent=details.get("user_agent"),
            endpoint=details.get("endpoint"),
            success=details.get("success", True),
            error_message=details.get("error_message"),
            additional_data=details.get("additional_data", {}),
            severity="INFO" if details.get("success", True) else "WARNING"
        )
    
    def log_authentication_attempt(self, user_id: str, success: bool, 
                                 source_ip: str, method: str):
        """Log authentication attempts."""
        event_type = (SecurityEventType.AUTHENTICATION_SUCCESS 
                     if success else SecurityEventType.AUTHENTICATION_FAILURE)
        
        self.log_security_event(event_type, user_id, {
            "source_ip": source_ip,
            "method": method,
            "success": success
        })
    
    def log_data_access(self, user_id: str, resource: str, 
                       action: str, success: bool):
        """Log data access events."""
        self.log_security_event(SecurityEventType.DATA_ACCESS, user_id, {
            "resource": resource,
            "action": action,
            "success": success,
            "additional_data": {"resource_type": "trading_data"}
        })

# Integration with FastAPI middleware
class SecurityAuditMiddleware:
    def __init__(self, app):
        self.app = app
        self.audit_logger = SecurityAuditLogger()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract request information
            request_info = {
                "path": scope["path"],
                "method": scope["method"],
                "client_ip": scope.get("client", ["unknown"])[0],
                "headers": dict(scope.get("headers", []))
            }
            
            # Log request
            user_id = await self.extract_user_id(scope)
            
            self.audit_logger.log_security_event(
                SecurityEventType.SYSTEM_ACCESS,
                user_id or "anonymous",
                request_info
            )
        
        await self.app(scope, receive, send)
```

#### 2. Audit Log Analysis
```python
# Automated audit log analysis
class SecurityAuditAnalyzer:
    def __init__(self):
        self.suspicious_patterns = [
            # Multiple failed logins
            {"type": "auth_failure", "count": 5, "timeframe": 300},
            
            # Unusual data access patterns
            {"type": "data_access", "count": 100, "timeframe": 3600},
            
            # Privilege escalation attempts
            {"type": "authz_failure", "count": 3, "timeframe": 600},
        ]
    
    async def analyze_security_logs(self, time_window: int = 3600):
        """Analyze security logs for suspicious patterns."""
        suspicious_events = []
        
        for pattern in self.suspicious_patterns:
            events = await self.query_security_events(
                event_type=pattern["type"],
                timeframe=pattern["timeframe"]
            )
            
            if len(events) >= pattern["count"]:
                suspicious_events.append({
                    "pattern": pattern,
                    "events": events,
                    "severity": "HIGH",
                    "recommendation": "Investigate immediately"
                })
        
        if suspicious_events:
            await self.trigger_security_alert(suspicious_events)
        
        return suspicious_events
    
    async def trigger_security_alert(self, events: list):
        """Trigger security alerts for suspicious activity."""
        alert_payload = {
            "alert_type": "suspicious_activity",
            "severity": "HIGH",
            "events_count": len(events),
            "details": events,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to security team
        await self.send_security_notification(alert_payload)
```

## Compliance and Governance

### Regulatory Compliance

#### 1. Data Governance Framework
```yaml
# Data governance and compliance configuration
data_governance:
  # Data classification
  classification_levels:
    - public        # Non-sensitive data
    - internal      # Internal business data  
    - confidential  # Trading data and strategies
    - restricted    # PII and financial data
  
  # Data retention policies
  retention_policies:
    trading_data: "7_years"      # Financial regulations
    user_data: "5_years"         # GDPR requirements
    audit_logs: "10_years"       # Compliance requirements
    system_logs: "1_year"        # Operational data
  
  # Data access controls
  access_controls:
    encrypted_at_rest: true
    encrypted_in_transit: true
    access_logging: true
    data_masking: true
  
  # Privacy controls
  privacy_controls:
    gdpr_compliance: true
    ccpa_compliance: true
    right_to_erasure: true
    data_portability: true
```

#### 2. Compliance Monitoring
```python
# Compliance monitoring system
class ComplianceMonitor:
    def __init__(self):
        self.compliance_rules = {
            "gdpr": {
                "data_retention": 5 * 365,  # 5 years in days
                "consent_tracking": True,
                "right_to_erasure": True,
                "data_portability": True
            },
            "pci_dss": {
                "data_encryption": True,
                "access_logging": True,
                "network_segmentation": True,
                "regular_testing": True
            },
            "sox": {
                "audit_trail": True,
                "segregation_of_duties": True,
                "change_management": True,
                "financial_reporting_controls": True
            }
        }
    
    async def check_compliance(self, regulation: str) -> dict:
        """Check compliance status for a specific regulation."""
        rules = self.compliance_rules.get(regulation, {})
        compliance_status = {}
        
        for rule, requirement in rules.items():
            try:
                status = await self.evaluate_compliance_rule(rule, requirement)
                compliance_status[rule] = {
                    "compliant": status,
                    "requirement": requirement,
                    "last_checked": datetime.utcnow().isoformat()
                }
            except Exception as e:
                compliance_status[rule] = {
                    "compliant": False,
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }
        
        return {
            "regulation": regulation,
            "overall_compliance": all(r["compliant"] for r in compliance_status.values()),
            "rules": compliance_status
        }
    
    async def generate_compliance_report(self) -> dict:
        """Generate comprehensive compliance report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "regulations": {}
        }
        
        for regulation in self.compliance_rules.keys():
            report["regulations"][regulation] = await self.check_compliance(regulation)
        
        # Calculate overall compliance score
        total_rules = sum(len(rules) for rules in self.compliance_rules.values())
        compliant_rules = sum(
            sum(1 for rule in reg_data["rules"].values() if rule["compliant"])
            for reg_data in report["regulations"].values()
        )
        
        report["compliance_score"] = (compliant_rules / total_rules) * 100
        
        return report
```

## Security Monitoring

### Real-Time Security Monitoring

#### 1. Security Metrics Collection
```python
# Security metrics for monitoring
SECURITY_METRICS = {
    # Authentication metrics
    'authentication_failures_per_minute': {
        'description': 'Failed authentication attempts per minute',
        'threshold_warning': 10,
        'threshold_critical': 50,
        'labels': ['source_ip', 'user_agent']
    },
    
    # Authorization failures
    'authorization_failures_per_minute': {
        'description': 'Authorization failures per minute',
        'threshold_warning': 5,
        'threshold_critical': 20,
        'labels': ['user_id', 'resource', 'permission']
    },
    
    # Suspicious activity
    'suspicious_requests_per_minute': {
        'description': 'Suspicious requests per minute',
        'threshold_warning': 1,
        'threshold_critical': 5,
        'labels': ['pattern_type', 'source_ip']
    },
    
    # API abuse
    'rate_limit_violations_per_minute': {
        'description': 'Rate limit violations per minute',
        'threshold_warning': 10,
        'threshold_critical': 50,
        'labels': ['endpoint', 'client_type']
    }
}

class SecurityMetricsCollector:
    def __init__(self):
        self.metrics_client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{PROJECT_ID}"
    
    async def record_security_event(self, metric_name: str, value: int, 
                                  labels: dict = None):
        """Record security metric event."""
        series = monitoring_v3.TimeSeries(
            metric=monitoring_v3.Metric(
                type=f"custom.googleapis.com/security/{metric_name}",
                labels=labels or {}
            ),
            resource=monitoring_v3.MonitoredResource(
                type="cloud_run_revision",
                labels={"service_name": "shyvr-rlte"}
            ),
            points=[monitoring_v3.Point(
                interval=monitoring_v3.TimeInterval(
                    end_time={"seconds": int(time.time())}
                ),
                value=monitoring_v3.TypedValue(int64_value=value)
            )]
        )
        
        self.metrics_client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )
```

#### 2. Intrusion Detection
```python
# Intrusion detection system
class IntrusionDetectionSystem:
    def __init__(self):
        self.attack_patterns = {
            'sql_injection': [
                r'union\s+select',
                r'or\s+1\s*=\s*1',
                r'drop\s+table',
                r'exec\s*\(',
            ],
            'xss': [
                r'<script',
                r'javascript:',
                r'onerror\s*=',
                r'onload\s*=',
            ],
            'path_traversal': [
                r'\.\./\.\.',
                r'%2e%2e%2f',
                r'..\\',
            ],
            'command_injection': [
                r';\s*cat\s',
                r';\s*ls\s',
                r'`.*`',
                r'\$\(.*\)',
            ]
        }
    
    async def analyze_request(self, request_data: dict) -> dict:
        """Analyze request for malicious patterns."""
        threats_detected = []
        risk_score = 0
        
        # Analyze all string values in request
        for key, value in request_data.items():
            if isinstance(value, str):
                for attack_type, patterns in self.attack_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, value.lower()):
                            threats_detected.append({
                                'type': attack_type,
                                'pattern': pattern,
                                'field': key,
                                'value': value[:100]  # Truncate for logging
                            })
                            risk_score += 10
        
        # Additional checks
        if len(request_data.get('user_input', '')) > 10000:
            threats_detected.append({
                'type': 'oversized_input',
                'field': 'user_input',
                'size': len(request_data.get('user_input', ''))
            })
            risk_score += 5
        
        return {
            'threats_detected': threats_detected,
            'risk_score': risk_score,
            'action_required': risk_score > 20
        }
```

## Best Practices Summary

### Security Checklist

#### Production Security Requirements
- [ ] All secrets stored in GCP Secret Manager
- [ ] JWT tokens with proper expiration and validation
- [ ] Rate limiting implemented for all endpoints
- [ ] Input validation and sanitization enabled
- [ ] SQL injection prevention measures active
- [ ] TLS 1.2+ enforced for all communications
- [ ] Container running as non-root user
- [ ] VPC isolation and firewall rules configured
- [ ] Audit logging enabled for all security events
- [ ] Regular security scanning automated
- [ ] Incident response procedures documented
- [ ] Compliance monitoring active

#### Security Monitoring Requirements
- [ ] Real-time security metrics collection
- [ ] Automated threat detection
- [ ] Security alert policies configured
- [ ] Intrusion detection system active
- [ ] Log analysis and correlation enabled
- [ ] Compliance monitoring dashboard
- [ ] Security incident tracking
- [ ] Regular security audits scheduled

#### Access Control Requirements
- [ ] Role-based access control implemented
- [ ] Principle of least privilege enforced
- [ ] Multi-factor authentication required
- [ ] API key management system active
- [ ] Service account permissions minimal
- [ ] Regular access reviews conducted
- [ ] Privileged access monitoring
- [ ] Identity lifecycle management

## Conclusion

This comprehensive security and access control guide provides enterprise-grade security for the transformer-enabled RLTE system. The security framework includes:

- **Multi-Layer Defense**: Network, application, and data security layers
- **Zero Trust Architecture**: Never trust, always verify approach
- **Comprehensive Monitoring**: Real-time security monitoring and alerting
- **Regulatory Compliance**: GDPR, PCI-DSS, and SOX compliance features
- **Incident Response**: Automated detection and response capabilities

### Key Security Benefits
1. **Enterprise Security**: Bank-grade security controls and monitoring
2. **Regulatory Compliance**: Built-in compliance with major regulations
3. **Real-Time Protection**: Automated threat detection and response
4. **Comprehensive Auditing**: Complete audit trail for all security events
5. **Defense in Depth**: Multiple security layers for maximum protection

### Next Steps
1. Review the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
2. Implement security monitoring dashboards
3. Configure alert policies and notification channels
4. Conduct security testing and penetration testing
5. Establish security incident response procedures
6. Schedule regular security audits and compliance reviews

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-06  
**Maintained By**: ML Development Team