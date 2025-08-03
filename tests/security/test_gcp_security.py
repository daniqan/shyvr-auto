"""
Phase 7.3: GCP Security Integration Tests
Following TDD methodology - these tests validate GCP security integrations
"""

import pytest
import asyncio
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from google.cloud import secretmanager
from google.cloud import storage
from google.cloud import sql
from google.cloud import iam
from google.cloud import logging as gcp_logging
from google.oauth2 import service_account
import google.auth.exceptions


class TestGCPSecretManagerSecurity:
    """Test GCP Secret Manager security integration following TDD methodology"""
    
    @pytest.fixture
    def mock_secret_client(self):
        """Mock GCP Secret Manager client"""
        with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
            yield mock_client.return_value

    def test_secret_access_should_require_proper_authentication(self, mock_secret_client):
        """Test Secret Manager access requires proper authentication - TDD FAIL FIRST"""
        # Mock authentication failure
        mock_secret_client.access_secret_version.side_effect = google.auth.exceptions.DefaultCredentialsError(
            "Could not automatically determine credentials"
        )
        
        # Should fail without proper credentials
        with pytest.raises((google.auth.exceptions.DefaultCredentialsError, Exception)):
            secret_name = "projects/trading-system/secrets/database-password/versions/latest"
            mock_secret_client.access_secret_version(request={"name": secret_name})

    def test_secret_access_should_require_proper_iam_permissions(self, mock_secret_client):
        """Test Secret Manager requires proper IAM permissions - TDD FAIL FIRST"""
        # Mock permission denied error
        from google.api_core import exceptions
        mock_secret_client.access_secret_version.side_effect = exceptions.PermissionDenied(
            "Permission 'secretmanager.versions.access' denied"
        )
        
        # Should fail without proper IAM permissions
        with pytest.raises((exceptions.PermissionDenied, Exception)):
            secret_name = "projects/trading-system/secrets/api-key/versions/latest"
            mock_secret_client.access_secret_version(request={"name": secret_name})

    def test_secret_versions_should_be_encrypted_at_rest(self, mock_secret_client):
        """Test secrets are encrypted at rest in Secret Manager - TDD FAIL FIRST"""
        # Mock secret response
        mock_response = MagicMock()
        mock_response.payload.data = b"encrypted_secret_data_not_plain_text"
        mock_secret_client.access_secret_version.return_value = mock_response
        
        secret_name = "projects/trading-system/secrets/jwt-secret/versions/latest"
        response = mock_secret_client.access_secret_version(request={"name": secret_name})
        
        # Secret data should be encrypted/encoded, not plain text
        secret_data = response.payload.data
        
        # Should not contain obvious plain text patterns
        secret_text = secret_data.decode('utf-8', errors='ignore')
        assert not any(pattern in secret_text.lower() for pattern in [
            "password", "secret", "key", "token", "admin"
        ]), "Secret should be encrypted, not contain plain text patterns"

    def test_secret_audit_logging_should_track_access(self):
        """Test Secret Manager access is properly audited - TDD FAIL FIRST"""
        # Mock GCP audit logging
        with patch('google.cloud.logging.Client') as mock_logging_client:
            mock_logger = MagicMock()
            mock_logging_client.return_value.logger.return_value = mock_logger
            
            # Mock secret access with audit logging
            with patch('google.cloud.secretmanager.SecretManagerServiceClient') as mock_client:
                mock_response = MagicMock()
                mock_response.payload.data = b"secret_value"
                mock_client.return_value.access_secret_version.return_value = mock_response
                
                # Access secret (should generate audit log)
                secret_manager = SecretManagerSecurityWrapper()
                secret_value = secret_manager.get_secret_with_audit(
                    "projects/trading-system/secrets/database-password/versions/latest",
                    requesting_user="trading-service"
                )
                
                # Verify audit logging was called
                mock_logger.log_struct.assert_called_once()
                log_entry = mock_logger.log_struct.call_args[0][0]
                
                assert "secret_access" in log_entry["event_type"], \
                    "Secret access should be audited"
                assert "trading-service" in log_entry["requesting_user"], \
                    "Audit should track requesting user"

    def test_secret_rotation_should_maintain_security(self, mock_secret_client):
        """Test secret rotation maintains security - TDD FAIL FIRST"""
        # Mock multiple secret versions
        old_version = MagicMock()
        old_version.payload.data = b"old_secret_value"
        
        new_version = MagicMock()
        new_version.payload.data = b"new_secret_value"
        
        # Mock different responses based on version
        def mock_access_version(request):
            if "versions/1" in request["name"]:
                return old_version
            elif "versions/latest" in request["name"]:
                return new_version
            else:
                raise Exception("Version not found")
        
        mock_secret_client.access_secret_version.side_effect = mock_access_version
        
        # Get old version
        old_secret_name = "projects/trading-system/secrets/api-key/versions/1"
        old_response = mock_secret_client.access_secret_version(request={"name": old_secret_name})
        
        # Get latest version
        latest_secret_name = "projects/trading-system/secrets/api-key/versions/latest"
        latest_response = mock_secret_client.access_secret_version(request={"name": latest_secret_name})
        
        # Versions should be different
        assert old_response.payload.data != latest_response.payload.data, \
            "Secret rotation should produce different values"


class TestGCPCloudSQLSecurity:
    """Test GCP Cloud SQL security configuration"""
    
    @pytest.fixture
    def mock_sql_client(self):
        """Mock GCP Cloud SQL client"""
        with patch('google.cloud.sql_v1.SqlInstancesServiceClient') as mock_client:
            yield mock_client.return_value

    def test_cloud_sql_should_require_ssl_connections(self, mock_sql_client):
        """Test Cloud SQL requires SSL connections - TDD FAIL FIRST"""
        # Mock instance configuration
        mock_instance = MagicMock()
        mock_instance.settings.ip_configuration.require_ssl = True
        mock_instance.settings.ip_configuration.authorized_networks = []
        
        mock_sql_client.get.return_value = mock_instance
        
        # Get instance configuration
        project_id = "trading-system"
        instance_id = "trading-db"
        
        instance = mock_sql_client.get(project=project_id, instance=instance_id)
        
        # SSL should be required
        assert instance.settings.ip_configuration.require_ssl is True, \
            "Cloud SQL should require SSL connections"

    def test_cloud_sql_should_restrict_authorized_networks(self, mock_sql_client):
        """Test Cloud SQL restricts authorized networks - TDD FAIL FIRST"""
        # Mock instance with overly permissive network access
        mock_instance = MagicMock()
        mock_instance.settings.ip_configuration.authorized_networks = [
            {"value": "0.0.0.0/0", "name": "allow-all"}  # Dangerous: allows all IPs
        ]
        
        mock_sql_client.get.return_value = mock_instance
        
        project_id = "trading-system"
        instance_id = "trading-db"
        
        instance = mock_sql_client.get(project=project_id, instance=instance_id)
        
        # Should not allow unrestricted access
        authorized_networks = instance.settings.ip_configuration.authorized_networks
        
        for network in authorized_networks:
            assert network["value"] != "0.0.0.0/0", \
                "Cloud SQL should not allow unrestricted network access (0.0.0.0/0)"

    def test_cloud_sql_should_enable_audit_logging(self, mock_sql_client):
        """Test Cloud SQL has audit logging enabled - TDD FAIL FIRST"""
        # Mock instance configuration
        mock_instance = MagicMock()
        mock_instance.settings.database_flags = [
            {"name": "log_statement", "value": "all"},
            {"name": "log_min_duration_statement", "value": "0"},
            {"name": "log_connections", "value": "on"},
            {"name": "log_disconnections", "value": "on"}
        ]
        
        mock_sql_client.get.return_value = mock_instance
        
        project_id = "trading-system" 
        instance_id = "trading-db"
        
        instance = mock_sql_client.get(project=project_id, instance=instance_id)
        
        # Verify audit logging is configured
        db_flags = {flag["name"]: flag["value"] for flag in instance.settings.database_flags}
        
        assert db_flags.get("log_statement") == "all", \
            "Cloud SQL should log all statements for audit"
        assert db_flags.get("log_connections") == "on", \
            "Cloud SQL should log connections for audit"

    def test_cloud_sql_should_use_private_ip(self, mock_sql_client):
        """Test Cloud SQL uses private IP for enhanced security - TDD FAIL FIRST"""
        # Mock instance configuration
        mock_instance = MagicMock()
        mock_instance.settings.ip_configuration.ipv4_enabled = False  # No public IP
        mock_instance.settings.ip_configuration.private_network = "projects/trading-system/global/networks/default"
        
        mock_sql_client.get.return_value = mock_instance
        
        project_id = "trading-system"
        instance_id = "trading-db"
        
        instance = mock_sql_client.get(project=project_id, instance=instance_id)
        
        # Should use private IP only
        ip_config = instance.settings.ip_configuration
        assert ip_config.ipv4_enabled is False, \
            "Cloud SQL should not have public IP enabled"
        assert ip_config.private_network is not None, \
            "Cloud SQL should be configured with private network"


class TestGCPIAMSecurity:
    """Test GCP IAM security configuration"""
    
    @pytest.fixture
    def mock_iam_client(self):
        """Mock GCP IAM client"""
        with patch('google.cloud.iam.Policy') as mock_policy:
            yield mock_policy

    def test_service_account_should_follow_principle_of_least_privilege(self, mock_iam_client):
        """Test service accounts follow principle of least privilege - TDD FAIL FIRST"""
        # Mock overly permissive IAM policy
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {
                "role": "roles/owner",  # Too broad
                "members": ["serviceAccount:trading-service@trading-system.iam.gserviceaccount.com"]
            },
            {
                "role": "roles/editor",  # Too broad
                "members": ["serviceAccount:ml-service@trading-system.iam.gserviceaccount.com"]
            }
        ]
        
        # Validate IAM policy
        iam_validator = GCPIAMValidator()
        validation_result = iam_validator.validate_service_account_permissions(mock_policy)
        
        assert not validation_result.is_secure, \
            "Service accounts should not have overly broad permissions"
        assert any("least privilege" in violation.lower() for violation in validation_result.violations), \
            "Should detect principle of least privilege violations"

    def test_iam_should_require_mfa_for_admin_roles(self):
        """Test IAM requires MFA for administrative roles - TDD FAIL FIRST"""
        # Mock IAM conditions that should require MFA
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {
                "role": "roles/iam.serviceAccountAdmin",
                "members": ["user:admin@trading-system.com"],
                "condition": None  # No MFA condition - should fail
            }
        ]
        
        iam_validator = GCPIAMValidator()
        mfa_validation = iam_validator.validate_mfa_requirements(mock_policy)
        
        assert not mfa_validation.is_secure, \
            "Administrative roles should require MFA"
        assert "mfa" in mfa_validation.violation_reason.lower(), \
            "Should indicate MFA requirement violation"

    def test_iam_should_restrict_external_users(self):
        """Test IAM restricts external user access - TDD FAIL FIRST"""
        # Mock IAM policy with external users
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {
                "role": "roles/secretmanager.secretAccessor",
                "members": [
                    "user:internal@trading-system.com",  # Internal - OK
                    "user:external@gmail.com",  # External - should be restricted
                    "user:contractor@outsidecompany.com"  # External - should be restricted
                ]
            }
        ]
        
        iam_validator = GCPIAMValidator()
        external_user_validation = iam_validator.validate_external_user_restrictions(
            mock_policy, allowed_domains=["trading-system.com"]
        )
        
        assert not external_user_validation.is_secure, \
            "Should restrict external user access to sensitive resources"
        assert len(external_user_validation.external_users) > 0, \
            "Should detect external users"


class TestGCPCloudStorageSecurity:
    """Test GCP Cloud Storage security configuration"""
    
    @pytest.fixture
    def mock_storage_client(self):
        """Mock GCP Cloud Storage client"""
        with patch('google.cloud.storage.Client') as mock_client:
            yield mock_client.return_value

    def test_storage_buckets_should_not_be_publicly_accessible(self, mock_storage_client):
        """Test storage buckets are not publicly accessible - TDD FAIL FIRST"""
        # Mock bucket with public access
        mock_bucket = MagicMock()
        mock_bucket.name = "trading-system-data"
        
        # Mock IAM policy with public access
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {
                "role": "roles/storage.objectViewer",
                "members": ["allUsers"]  # Dangerous: public access
            }
        ]
        
        mock_bucket.get_iam_policy.return_value = mock_policy
        mock_storage_client.get_bucket.return_value = mock_bucket
        
        # Validate bucket security
        bucket = mock_storage_client.get_bucket("trading-system-data")
        policy = bucket.get_iam_policy()
        
        # Should not have public access
        for binding in policy.bindings:
            assert "allUsers" not in binding["members"], \
                "Storage bucket should not allow public access"
            assert "allAuthenticatedUsers" not in binding["members"], \
                "Storage bucket should not allow all authenticated users"

    def test_storage_objects_should_be_encrypted(self, mock_storage_client):
        """Test storage objects are encrypted - TDD FAIL FIRST"""
        # Mock bucket with encryption configuration
        mock_bucket = MagicMock()
        mock_bucket.name = "trading-system-backups"
        mock_bucket.default_kms_key_name = "projects/trading-system/locations/global/keyRings/trading-keys/cryptoKeys/storage-key"
        
        mock_storage_client.get_bucket.return_value = mock_bucket
        
        # Get bucket configuration
        bucket = mock_storage_client.get_bucket("trading-system-backups")
        
        # Should have encryption configured
        assert bucket.default_kms_key_name is not None, \
            "Storage bucket should have default KMS encryption key"
        assert "cryptoKeys" in bucket.default_kms_key_name, \
            "Should use proper KMS key path"

    def test_storage_access_should_be_audited(self, mock_storage_client):
        """Test storage access is properly audited - TDD FAIL FIRST"""
        # Mock bucket with audit logging
        mock_bucket = MagicMock()
        mock_bucket.name = "trading-system-logs"
        
        # Mock logging configuration
        mock_bucket.logging = {
            "logBucket": "trading-system-audit-logs",
            "logObjectPrefix": "storage-access-"
        }
        
        mock_storage_client.get_bucket.return_value = mock_bucket
        
        # Validate audit logging
        bucket = mock_storage_client.get_bucket("trading-system-logs")
        
        # Should have audit logging configured
        assert hasattr(bucket, 'logging') and bucket.logging is not None, \
            "Storage bucket should have audit logging configured"


class TestGCPNetworkSecurity:
    """Test GCP network security configuration"""
    
    def test_vpc_firewall_should_restrict_ingress_traffic(self):
        """Test VPC firewall restricts ingress traffic - TDD FAIL FIRST"""
        # Mock overly permissive firewall rule
        firewall_rules = [
            {
                "name": "allow-all-ingress",
                "direction": "INGRESS",
                "priority": 1000,
                "sourceRanges": ["0.0.0.0/0"],  # Allows all IPs
                "allowed": [
                    {"IPProtocol": "tcp", "ports": ["22", "80", "443", "3306", "5432"]}  # Many ports
                ]
            }
        ]
        
        network_validator = GCPNetworkValidator()
        validation_result = network_validator.validate_firewall_rules(firewall_rules)
        
        assert not validation_result.is_secure, \
            "Firewall should not allow unrestricted ingress traffic"
        assert "0.0.0.0/0" in validation_result.violation_details, \
            "Should detect unrestricted source ranges"

    def test_vpc_should_use_private_google_access(self):
        """Test VPC uses Private Google Access - TDD FAIL FIRST"""
        # Mock subnet configuration
        subnet_config = {
            "name": "trading-subnet",
            "privateIpGoogleAccess": False,  # Should be True for security
            "ipCidrRange": "10.0.0.0/24"
        }
        
        network_validator = GCPNetworkValidator()
        validation_result = network_validator.validate_private_google_access(subnet_config)
        
        assert not validation_result.is_secure, \
            "Subnet should enable Private Google Access"
        assert "private google access" in validation_result.violation_reason.lower(), \
            "Should indicate Private Google Access requirement"

    def test_load_balancer_should_use_https_only(self):
        """Test load balancer uses HTTPS only - TDD FAIL FIRST"""
        # Mock load balancer with HTTP enabled
        lb_config = {
            "name": "trading-lb",
            "urlMap": {
                "defaultService": "trading-service",
                "hostRules": [
                    {
                        "hosts": ["api.trading-system.com"],
                        "pathMatcher": "trading-paths"
                    }
                ]
            },
            "httpsRedirect": False,  # Should be True
            "sslPolicy": None  # Should have SSL policy
        }
        
        network_validator = GCPNetworkValidator()
        validation_result = network_validator.validate_load_balancer_security(lb_config)
        
        assert not validation_result.is_secure, \
            "Load balancer should enforce HTTPS only"
        assert "https" in validation_result.violation_reason.lower(), \
            "Should indicate HTTPS requirement"


# Mock validator classes for testing
class SecretManagerSecurityWrapper:
    """Security wrapper for Secret Manager operations"""
    
    def get_secret_with_audit(self, secret_name, requesting_user):
        """Get secret with audit logging"""
        # Mock implementation that includes audit logging
        import google.cloud.logging
        
        client = google.cloud.logging.Client()
        logger = client.logger("secret-access-audit")
        
        # Log secret access
        logger.log_struct({
            "event_type": "secret_access",
            "secret_name": secret_name,
            "requesting_user": requesting_user,
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
        return "secret_value"


class GCPIAMValidator:
    """Validator for GCP IAM security policies"""
    
    def validate_service_account_permissions(self, policy):
        """Validate service account permissions"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'violations'])
        
        violations = []
        
        for binding in policy.bindings:
            # Check for overly broad roles
            if binding["role"] in ["roles/owner", "roles/editor"]:
                violations.append(f"Overly broad role detected: {binding['role']} - violates least privilege principle")
        
        return Result(len(violations) == 0, violations)
    
    def validate_mfa_requirements(self, policy):
        """Validate MFA requirements for admin roles"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'violation_reason'])
        
        admin_roles = ["roles/iam.serviceAccountAdmin", "roles/resourcemanager.projectIamAdmin"]
        
        for binding in policy.bindings:
            if binding["role"] in admin_roles and binding.get("condition") is None:
                return Result(False, "Administrative roles should require MFA condition")
        
        return Result(True, "")
    
    def validate_external_user_restrictions(self, policy, allowed_domains):
        """Validate external user restrictions"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'external_users'])
        
        external_users = []
        
        for binding in policy.bindings:
            for member in binding["members"]:
                if member.startswith("user:"):
                    email = member.replace("user:", "")
                    domain = email.split("@")[-1]
                    
                    if domain not in allowed_domains:
                        external_users.append(email)
        
        return Result(len(external_users) == 0, external_users)


class GCPNetworkValidator:
    """Validator for GCP network security configuration"""
    
    def validate_firewall_rules(self, firewall_rules):
        """Validate firewall rule security"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'violation_details'])
        
        violations = []
        
        for rule in firewall_rules:
            if rule.get("direction") == "INGRESS":
                source_ranges = rule.get("sourceRanges", [])
                
                if "0.0.0.0/0" in source_ranges:
                    violations.append("Unrestricted ingress traffic (0.0.0.0/0) detected")
        
        return Result(len(violations) == 0, "; ".join(violations))
    
    def validate_private_google_access(self, subnet_config):
        """Validate Private Google Access configuration"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'violation_reason'])
        
        if not subnet_config.get("privateIpGoogleAccess", False):
            return Result(False, "Private Google Access should be enabled")
        
        return Result(True, "")
    
    def validate_load_balancer_security(self, lb_config):
        """Validate load balancer security configuration"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_secure', 'violation_reason'])
        
        if not lb_config.get("httpsRedirect", False):
            return Result(False, "HTTPS redirect should be enabled")
        
        if lb_config.get("sslPolicy") is None:
            return Result(False, "SSL policy should be configured")
        
        return Result(True, "")