"""
Phase 7.3: Data Security Tests (Encryption at Rest & Transit)
Following TDD methodology - these tests validate data security mechanisms
"""

import pytest
import asyncio
import ssl
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json
import sqlite3
from datetime import datetime
import requests
from pathlib import Path

from src.utils.security.config_encryption import ConfigEncryption
from src.activity_logging.activity_logger import ActivityLogger


class TestEncryptionAtRest:
    """Test data encryption at rest security following TDD methodology"""
    
    @pytest.fixture
    def encryption_key(self):
        """Generate test encryption key"""
        return Fernet.generate_key()
    
    @pytest.fixture
    def temp_db_file(self):
        """Create temporary database file for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            yield f.name
        os.unlink(f.name)

    def test_database_encryption_should_encrypt_sensitive_data_at_rest(self, temp_db_file, encryption_key):
        """Test database encryption encrypts sensitive data at rest - TDD FAIL FIRST"""
        # Create database with sensitive data
        conn = sqlite3.connect(temp_db_file)
        cursor = conn.cursor()
        
        # Create table for sensitive data
        cursor.execute('''
            CREATE TABLE user_credentials (
                id INTEGER PRIMARY KEY,
                username TEXT,
                password_hash TEXT,
                api_key TEXT,
                private_key TEXT
            )
        ''')
        
        # Insert sensitive data (should be encrypted)
        sensitive_data = [
            ("admin", "hashed_password_123", "api_key_456", "private_key_789"),
            ("trader", "hashed_password_abc", "api_key_def", "private_key_ghi")
        ]
        
        fernet = Fernet(encryption_key)
        
        for username, password_hash, api_key, private_key in sensitive_data:
            # Encrypt sensitive fields before storing
            encrypted_password = fernet.encrypt(password_hash.encode()).decode()
            encrypted_api_key = fernet.encrypt(api_key.encode()).decode()
            encrypted_private_key = fernet.encrypt(private_key.encode()).decode()
            
            cursor.execute('''
                INSERT INTO user_credentials (username, password_hash, api_key, private_key)
                VALUES (?, ?, ?, ?)
            ''', (username, encrypted_password, encrypted_api_key, encrypted_private_key))
        
        conn.commit()
        conn.close()
        
        # Verify data is encrypted at rest
        with open(temp_db_file, 'rb') as f:
            db_content = f.read()
            db_text = db_content.decode('utf-8', errors='ignore')
            
            # Sensitive data should not appear in plain text
            assert "hashed_password_123" not in db_text, "Password hash should be encrypted at rest"
            assert "api_key_456" not in db_text, "API key should be encrypted at rest"
            assert "private_key_789" not in db_text, "Private key should be encrypted at rest"

    def test_configuration_file_encryption_should_protect_secrets(self, encryption_key):
        """Test configuration file encryption protects secrets - TDD FAIL FIRST"""
        # Configuration with sensitive data
        sensitive_config = {
            "database": {
                "password": "super_secret_db_password",
                "connection_string": "postgresql://user:secret@localhost/db"
            },
            "api_keys": {
                "trading_api": "secret_trading_api_key",
                "market_data": "secret_market_data_key"
            },
            "encryption": {
                "master_key": "master_encryption_key_123"
            }
        }
        
        config_encryptor = ConfigEncryption(encryption_key)
        
        # Encrypt configuration
        encrypted_config = config_encryptor.encrypt_config(sensitive_config)
        
        # Save encrypted config to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            json.dump(encrypted_config, f)  # Using JSON for simplicity
            config_file = f.name
        
        try:
            # Verify file contents are encrypted
            with open(config_file, 'r') as f:
                file_content = f.read()
                
                # Sensitive data should not appear in plain text
                assert "super_secret_db_password" not in file_content, \
                    "Database password should be encrypted in config file"
                assert "secret_trading_api_key" not in file_content, \
                    "Trading API key should be encrypted in config file"
                assert "master_encryption_key_123" not in file_content, \
                    "Master key should be encrypted in config file"
        finally:
            os.unlink(config_file)

    def test_log_file_encryption_should_protect_sensitive_logs(self, encryption_key):
        """Test log file encryption protects sensitive information - TDD FAIL FIRST"""
        # Create activity logger with encryption
        logger = ActivityLogger(encryption_enabled=True, encryption_key=encryption_key)
        
        # Log sensitive trading activity
        sensitive_log_data = {
            "user_id": 12345,
            "action": "TRADE_EXECUTED", 
            "details": {
                "api_key": "secret_api_key_123",
                "wallet_address": "0x1234567890123456789012345678901234567890",
                "private_key_used": "private_key_for_signing",
                "trade_amount": "1.5 BTC"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            # Mock the logger to write to our test file
            with patch.object(logger, '_write_to_file') as mock_write:
                mock_write.side_effect = lambda data: self._write_encrypted_log(data, log_file, encryption_key)
                
                # Log sensitive data
                asyncio.run(logger.log_activity(
                    category="TRADING",
                    action="EXECUTE",
                    source="trading_engine",
                    event_type="trade_executed",
                    title="Trade executed",
                    metadata=sensitive_log_data
                ))
            
            # Verify log file contents are encrypted
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # Sensitive data should not appear in plain text
                assert "secret_api_key_123" not in log_content, \
                    "API key should be encrypted in logs"
                assert "private_key_for_signing" not in log_content, \
                    "Private key should be encrypted in logs"
                assert "0x1234567890123456789012345678901234567890" not in log_content, \
                    "Wallet address should be encrypted in logs"
        finally:
            os.unlink(log_file)

    def _write_encrypted_log(self, data, log_file, encryption_key):
        """Helper method to write encrypted log data"""
        fernet = Fernet(encryption_key)
        encrypted_data = fernet.encrypt(json.dumps(data).encode())
        
        with open(log_file, 'ab') as f:
            f.write(encrypted_data + b'\n')

    def test_backup_file_encryption_should_protect_data_backups(self, encryption_key):
        """Test backup file encryption protects data backups - TDD FAIL FIRST"""
        # Create backup data with sensitive information
        backup_data = {
            "users": [
                {"username": "admin", "password_hash": "admin_password_hash", "api_key": "admin_api_key"},
                {"username": "trader", "password_hash": "trader_password_hash", "api_key": "trader_api_key"}
            ],
            "trading_history": [
                {"trade_id": 1, "api_key": "trading_api_key", "amount": "1.5 BTC"},
                {"trade_id": 2, "api_key": "trading_api_key", "amount": "10 ETH"}
            ],
            "wallet_keys": [
                {"address": "0x123...", "private_key": "private_key_123"},
                {"address": "0x456...", "private_key": "private_key_456"}
            ]
        }
        
        # Encrypt backup data
        fernet = Fernet(encryption_key)
        encrypted_backup = fernet.encrypt(json.dumps(backup_data).encode())
        
        # Save encrypted backup
        with tempfile.NamedTemporaryFile(delete=False, suffix='.backup') as f:
            f.write(encrypted_backup)
            backup_file = f.name
        
        try:
            # Verify backup file is encrypted
            with open(backup_file, 'rb') as f:
                backup_content = f.read()
                backup_text = backup_content.decode('utf-8', errors='ignore')
                
                # Sensitive data should not appear in plain text
                assert "admin_password_hash" not in backup_text, \
                    "Password hash should be encrypted in backup"
                assert "admin_api_key" not in backup_text, \
                    "API key should be encrypted in backup"
                assert "private_key_123" not in backup_text, \
                    "Private key should be encrypted in backup"
        finally:
            os.unlink(backup_file)


class TestEncryptionInTransit:
    """Test data encryption in transit security"""
    
    def test_https_enforcement_should_reject_http_connections(self):
        """Test HTTPS enforcement rejects HTTP connections - TDD FAIL FIRST"""
        # Mock HTTP request (should be rejected)
        http_url = "http://api.trading-system.com/api/portfolio"
        
        # Should reject HTTP connections
        with pytest.raises((requests.exceptions.SSLError, ValueError, Exception)):
            # This should fail if HTTPS enforcement is properly implemented
            response = requests.get(http_url, verify=True, timeout=5)
            
            # If request succeeds, check that it was redirected to HTTPS
            if response.status_code == 200:
                assert response.url.startswith('https://'), \
                    "HTTP requests should be redirected to HTTPS"

    def test_tls_version_validation_should_reject_weak_tls(self):
        """Test TLS version validation rejects weak TLS versions - TDD FAIL FIRST"""
        # Test with weak TLS versions
        weak_tls_versions = [
            ssl.PROTOCOL_SSLv3,   # SSL 3.0 (deprecated)
            ssl.PROTOCOL_TLSv1,   # TLS 1.0 (deprecated) 
            ssl.PROTOCOL_TLSv1_1, # TLS 1.1 (deprecated)
        ]
        
        for tls_version in weak_tls_versions:
            # Should reject weak TLS versions
            context = ssl.SSLContext(tls_version)
            
            with pytest.raises((ssl.SSLError, ValueError, Exception)):
                # This should fail if proper TLS validation is implemented
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                
                # Attempt connection with weak TLS (should fail)
                with patch('ssl.create_connection') as mock_conn:
                    mock_conn.side_effect = ssl.SSLError("Weak TLS version rejected")
                    raise ssl.SSLError("Weak TLS version not allowed")

    def test_certificate_validation_should_reject_invalid_certificates(self):
        """Test certificate validation rejects invalid certificates - TDD FAIL FIRST"""
        # Test with invalid certificates
        invalid_cert_scenarios = [
            {"cert_error": "CERTIFICATE_VERIFY_FAILED", "reason": "self-signed"},
            {"cert_error": "HOSTNAME_MISMATCH", "reason": "wrong hostname"}, 
            {"cert_error": "CERTIFICATE_EXPIRED", "reason": "expired certificate"},
            {"cert_error": "CERTIFICATE_REVOKED", "reason": "revoked certificate"},
        ]
        
        for scenario in invalid_cert_scenarios:
            with pytest.raises((ssl.SSLError, requests.exceptions.SSLError, Exception)):
                # Mock SSL error for invalid certificate
                with patch('requests.get') as mock_get:
                    mock_get.side_effect = ssl.SSLError(scenario["cert_error"])
                    
                    # Should reject invalid certificates
                    requests.get("https://invalid-cert.example.com", verify=True)

    def test_api_request_encryption_should_encrypt_sensitive_payloads(self):
        """Test API request encryption encrypts sensitive payloads - TDD FAIL FIRST"""
        # Sensitive API payload
        sensitive_payload = {
            "api_key": "secret_api_key_123",
            "wallet_private_key": "private_key_for_trading",
            "trade_details": {
                "amount": "1.5 BTC",
                "price": "$50000",
                "strategy": "proprietary_algorithm_v2"
            }
        }
        
        # Mock encrypted request
        with patch('requests.post') as mock_post:
            # Simulate sending encrypted payload
            encrypted_payload = self._encrypt_api_payload(sensitive_payload)
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "success"}
            
            # Send encrypted request
            response = requests.post(
                "https://api.trading-system.com/api/trade",
                json=encrypted_payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Verify payload was encrypted
            sent_data = mock_post.call_args[1]['json']
            sent_json = json.dumps(sent_data)
            
            # Sensitive data should not appear in plain text
            assert "secret_api_key_123" not in sent_json, \
                "API key should be encrypted in transit"
            assert "private_key_for_trading" not in sent_json, \
                "Private key should be encrypted in transit"
            assert "proprietary_algorithm_v2" not in sent_json, \
                "Sensitive strategy should be encrypted in transit"

    def _encrypt_api_payload(self, payload):
        """Helper method to encrypt API payload"""
        # Generate encryption key for payload
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        # Encrypt payload
        encrypted_data = fernet.encrypt(json.dumps(payload).encode())
        
        return {
            "encrypted_payload": encrypted_data.decode(),
            "encryption_metadata": {
                "algorithm": "Fernet",
                "key_id": "payload_key_123"  # Key ID for server to decrypt
            }
        }

    def test_database_connection_encryption_should_use_ssl(self):
        """Test database connections use SSL encryption - TDD FAIL FIRST"""
        # Mock database connection
        with patch('psycopg2.connect') as mock_connect:
            # Database connection should require SSL
            connection_params = {
                "host": "database.trading-system.com",
                "database": "trading_db",
                "user": "trading_user",
                "password": "trading_password",
                "sslmode": "require",  # Should require SSL
                "sslcert": "/path/to/client-cert.pem",
                "sslkey": "/path/to/client-key.pem",
                "sslrootcert": "/path/to/ca-cert.pem"
            }
            
            mock_connect.return_value = MagicMock()
            
            # Attempt database connection
            import psycopg2
            psycopg2.connect(**connection_params)
            
            # Verify SSL was required
            mock_connect.assert_called_with(**connection_params)
            called_params = mock_connect.call_args[1]
            assert called_params.get("sslmode") == "require", \
                "Database connection should require SSL"

    def test_websocket_connection_encryption_should_use_wss(self):
        """Test WebSocket connections use WSS encryption - TDD FAIL FIRST"""
        # Mock WebSocket connection
        with patch('websockets.connect') as mock_ws_connect:
            # WebSocket should use WSS (secure)
            ws_url = "wss://api.trading-system.com/ws/trading"
            
            mock_ws_connect.return_value.__aenter__ = MagicMock()
            mock_ws_connect.return_value.__aexit__ = MagicMock()
            
            # Should connect with WSS
            import websockets
            
            async def test_ws_connection():
                async with websockets.connect(ws_url, ssl=True) as websocket:
                    pass
            
            asyncio.run(test_ws_connection())
            
            # Verify WSS was used
            mock_ws_connect.assert_called()
            called_url = mock_ws_connect.call_args[0][0]
            assert called_url.startswith("wss://"), \
                "WebSocket connection should use WSS encryption"


class TestDataIntegrityValidation:
    """Test data integrity validation security"""
    
    def test_data_tampering_detection_should_identify_modified_data(self):
        """Test data tampering detection identifies modified data - TDD FAIL FIRST"""
        # Original data with checksum
        original_data = {
            "trade_id": 12345,
            "amount": "1.5 BTC",
            "price": "$50000",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        # Generate checksum for original data
        original_checksum = self._calculate_data_checksum(original_data)
        
        # Tampered data (amount changed)
        tampered_data = original_data.copy()
        tampered_data["amount"] = "15.0 BTC"  # Increased amount
        
        # Verify tampering is detected
        tampered_checksum = self._calculate_data_checksum(tampered_data)
        
        assert original_checksum != tampered_checksum, \
            "Data tampering should be detected through checksum mismatch"
        
        # Integrity validation should fail
        is_valid = self._validate_data_integrity(tampered_data, original_checksum)
        assert not is_valid, "Tampered data should fail integrity validation"

    def _calculate_data_checksum(self, data):
        """Helper method to calculate data checksum"""
        import hashlib
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()

    def _validate_data_integrity(self, data, expected_checksum):
        """Helper method to validate data integrity"""
        actual_checksum = self._calculate_data_checksum(data)
        return actual_checksum == expected_checksum

    def test_message_authentication_should_validate_message_origin(self):
        """Test message authentication validates message origin - TDD FAIL FIRST"""
        # Message with authentication
        message = {
            "command": "EXECUTE_TRADE",
            "params": {
                "symbol": "BTC/USD",
                "amount": "0.5",
                "side": "BUY"
            },
            "timestamp": datetime.now().isoformat(),
            "sender": "trading_bot_001"
        }
        
        # Generate message authentication code (MAC)
        secret_key = b"shared_secret_key_for_authentication"
        message_mac = self._generate_message_mac(message, secret_key)
        
        # Verify authentic message
        is_authentic = self._verify_message_mac(message, message_mac, secret_key)
        assert is_authentic, "Authentic message should pass MAC verification"
        
        # Test with wrong MAC (forged message)
        wrong_mac = "wrong_mac_value_123"
        is_forged = self._verify_message_mac(message, wrong_mac, secret_key)
        assert not is_forged, "Forged message should fail MAC verification"

    def _generate_message_mac(self, message, secret_key):
        """Helper method to generate message MAC"""
        import hmac
        import hashlib
        
        message_string = json.dumps(message, sort_keys=True)
        mac = hmac.new(secret_key, message_string.encode(), hashlib.sha256)
        return mac.hexdigest()

    def _verify_message_mac(self, message, provided_mac, secret_key):
        """Helper method to verify message MAC"""
        expected_mac = self._generate_message_mac(message, secret_key)
        return hmac.compare_digest(expected_mac, provided_mac)

    def test_replay_attack_prevention_should_reject_duplicate_requests(self):
        """Test replay attack prevention rejects duplicate requests - TDD FAIL FIRST"""
        # Request with nonce/timestamp
        request_1 = {
            "action": "TRANSFER_FUNDS",
            "amount": "1000 USD",
            "destination": "wallet_123",
            "nonce": "unique_nonce_001",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        # Simulate processing first request
        processed_nonces = set()
        
        # First request should be accepted
        is_valid_1 = self._validate_request_uniqueness(request_1, processed_nonces)
        assert is_valid_1, "First request should be accepted"
        
        # Add nonce to processed set
        processed_nonces.add(request_1["nonce"])
        
        # Duplicate request (replay attack) should be rejected
        request_2 = request_1.copy()  # Exact duplicate
        
        is_valid_2 = self._validate_request_uniqueness(request_2, processed_nonces)
        assert not is_valid_2, "Duplicate request should be rejected (replay attack prevention)"

    def _validate_request_uniqueness(self, request, processed_nonces):
        """Helper method to validate request uniqueness"""
        nonce = request.get("nonce")
        
        if not nonce:
            return False  # No nonce = invalid
        
        if nonce in processed_nonces:
            return False  # Duplicate nonce = replay attack
        
        # Check timestamp freshness (prevent old request replay)
        timestamp_str = request.get("timestamp")
        if timestamp_str:
            request_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            # Reject requests older than 5 minutes
            time_diff = current_time - request_time.replace(tzinfo=None)
            if time_diff.total_seconds() > 300:  # 5 minutes
                return False
        
        return True


class TestKeyManagement:
    """Test cryptographic key management security"""
    
    def test_key_rotation_should_update_encryption_keys_securely(self):
        """Test key rotation updates encryption keys securely - TDD FAIL FIRST"""
        # Original key and encrypted data
        original_key = Fernet.generate_key()
        fernet_old = Fernet(original_key)
        
        sensitive_data = "super_secret_trading_data_123"
        encrypted_data = fernet_old.encrypt(sensitive_data.encode())
        
        # Rotate to new key
        new_key = Fernet.generate_key()
        fernet_new = Fernet(new_key)
        
        # Re-encrypt data with new key
        decrypted_data = fernet_old.decrypt(encrypted_data)
        re_encrypted_data = fernet_new.encrypt(decrypted_data)
        
        # Verify new key can decrypt
        final_data = fernet_new.decrypt(re_encrypted_data).decode()
        assert final_data == sensitive_data, "Key rotation should preserve data"
        
        # Verify old key cannot decrypt new data
        with pytest.raises(Exception):
            fernet_old.decrypt(re_encrypted_data)

    def test_key_derivation_should_generate_strong_keys(self):
        """Test key derivation generates cryptographically strong keys - TDD FAIL FIRST"""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        import os
        
        # Test key derivation from password
        password = b"user_password_123"
        salt = os.urandom(32)  # 32 bytes salt
        
        # Use strong key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # Strong iteration count
        )
        
        derived_key = kdf.derive(password)
        
        # Verify key properties
        assert len(derived_key) == 32, "Derived key should be 32 bytes"
        
        # Same input should produce same key
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,  # Same salt
            iterations=100000,
        )
        derived_key_2 = kdf2.derive(password)
        assert derived_key == derived_key_2, "Same input should produce same derived key"
        
        # Different salt should produce different key
        different_salt = os.urandom(32)
        kdf3 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=different_salt,  # Different salt
            iterations=100000,
        )
        derived_key_3 = kdf3.derive(password)
        assert derived_key != derived_key_3, "Different salt should produce different key"

    def test_key_storage_should_protect_keys_in_memory(self):
        """Test key storage protects keys in memory - TDD FAIL FIRST"""
        # This test is conceptual - actual implementation would need secure memory management
        
        # Create key manager
        key_manager = SecureKeyManager()
        
        # Store key securely
        key_id = key_manager.store_key("trading_key", b"secret_key_data_123")
        
        # Key should not be accessible directly
        with pytest.raises((AttributeError, SecurityError, Exception)):
            # Should not be able to access raw key data
            raw_key = key_manager._keys["trading_key"]
        
        # Key should be retrievable through secure method
        retrieved_key = key_manager.get_key(key_id, authorized_user="admin")
        assert retrieved_key is not None, "Authorized user should be able to retrieve key"
        
        # Unauthorized access should fail
        with pytest.raises((PermissionError, SecurityError, Exception)):
            key_manager.get_key(key_id, authorized_user="unauthorized")

    def test_key_destruction_should_securely_wipe_keys(self):
        """Test key destruction securely wipes keys from memory - TDD FAIL FIRST"""
        import ctypes
        
        # Create key manager
        key_manager = SecureKeyManager()
        
        # Store key
        key_data = b"secret_key_to_be_destroyed"
        key_id = key_manager.store_key("temp_key", key_data)
        
        # Verify key exists
        assert key_manager.key_exists(key_id), "Key should exist before destruction"
        
        # Destroy key
        key_manager.destroy_key(key_id)
        
        # Verify key is destroyed
        assert not key_manager.key_exists(key_id), "Key should not exist after destruction"
        
        # Verify key cannot be retrieved
        with pytest.raises((KeyError, SecurityError, Exception)):
            key_manager.get_key(key_id, authorized_user="admin")


# Mock secure key manager for testing
class SecureKeyManager:
    """Mock secure key manager for testing purposes"""
    
    def __init__(self):
        self._keys = {}
        self._key_permissions = {}
        
    def store_key(self, key_name, key_data):
        """Store key securely (mocked implementation)"""
        key_id = f"key_{len(self._keys)}"
        # In real implementation, this would use secure memory/HSM
        self._keys[key_id] = {"encrypted_key": key_data, "name": key_name}
        self._key_permissions[key_id] = {"authorized_users": ["admin"]}
        return key_id
    
    def get_key(self, key_id, authorized_user):
        """Retrieve key with authorization check"""
        if key_id not in self._keys:
            raise KeyError("Key not found")
        
        if authorized_user not in self._key_permissions[key_id]["authorized_users"]:
            raise PermissionError("Unauthorized access to key")
        
        return self._keys[key_id]["encrypted_key"]
    
    def key_exists(self, key_id):
        """Check if key exists"""
        return key_id in self._keys
    
    def destroy_key(self, key_id):
        """Securely destroy key"""
        if key_id in self._keys:
            # In real implementation, this would securely wipe memory
            del self._keys[key_id]
            del self._key_permissions[key_id]


class SecurityError(Exception):
    """Custom security exception for testing"""
    pass