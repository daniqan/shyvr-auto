"""
Configuration encryption utilities for secure configuration loading
Provides encryption/decryption of sensitive configuration values
"""

import base64
import hashlib
import logging
import os
from typing import Any, Dict, List, Set

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ConfigEncryption:
    """Handles encryption and decryption of sensitive configuration values"""
    
    # Sensitive configuration keys that should be encrypted
    SENSITIVE_KEYS = {
        'password', 'secret', 'token', 'key', 'credentials',
        'private_key', 'webhook_secret', 'api_key', 'secret_key'
    }
    
    def __init__(self, master_key: str = None):
        """Initialize configuration encryption"""
        if master_key is None:
            master_key = self.derive_key_from_environment()
        
        self._fernet = self._create_fernet_from_key(master_key)
        logger.info("Configuration encryption initialized")
    
    def _create_fernet_from_key(self, master_key: str) -> Fernet:
        """Create Fernet encryption object from master key"""
        # Use PBKDF2 to derive a proper key from the master key
        salt = b'config_encryption_salt'  # In production, use a random salt per encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        # Handle both string and bytes input
        if isinstance(master_key, str):
            key_bytes = master_key.encode()
        else:
            key_bytes = master_key
        key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
        return Fernet(key)
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a single configuration value"""
        if not isinstance(value, str):
            raise ValueError("Only string values can be encrypted")
        
        encrypted_bytes = self._fernet.encrypt(value.encode())
        encrypted_b64 = base64.urlsafe_b64encode(encrypted_bytes).decode()
        return f"encrypted:{encrypted_b64}"
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a single configuration value"""
        if not encrypted_value.startswith('encrypted:'):
            raise ValueError("Value is not encrypted (missing 'encrypted:' prefix)")
        
        encrypted_b64 = encrypted_value[10:]  # Remove 'encrypted:' prefix
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_b64.encode())
        decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    
    def encrypt_sensitive_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive values in configuration dictionary"""
        result = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = self.encrypt_sensitive_config(value)
            elif isinstance(value, str) and self._is_sensitive_key(key):
                # Encrypt sensitive string values
                if not value.startswith('encrypted:'):
                    result[key] = self.encrypt_value(value)
                    logger.info(f"Encrypted sensitive configuration key: {key}")
                else:
                    result[key] = value  # Already encrypted
            else:
                # Keep non-sensitive values as-is
                result[key] = value
        
        return result
    
    def decrypt_config_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted values in configuration dictionary"""
        result = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = self.decrypt_config_values(value)
            elif isinstance(value, str) and value.startswith('encrypted:'):
                # Decrypt encrypted values
                try:
                    result[key] = self.decrypt_value(value)
                    logger.debug(f"Decrypted configuration key: {key}")
                except Exception as e:
                    logger.error(f"Failed to decrypt configuration key {key}: {e}")
                    raise ValueError(f"Failed to decrypt configuration key {key}: {e}")
            else:
                # Keep non-encrypted values as-is
                result[key] = value
        
        return result
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if configuration key contains sensitive data"""
        key_lower = key.lower()
        return any(sensitive_word in key_lower for sensitive_word in self.SENSITIVE_KEYS)
    
    def rotate_encryption_key(self) -> None:
        """Rotate the encryption key"""
        # Generate new master key
        new_master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        # Store old fernet for decryption
        old_fernet = self._fernet
        
        # Create new fernet with new key
        self._fernet = self._create_fernet_from_key(new_master_key)
        
        logger.warning("Encryption key rotated - old encrypted values will need re-encryption")
        return new_master_key
    
    def derive_key_from_environment(self) -> str:
        """Derive encryption key from environment variables"""
        # Try to get encryption key from environment
        encryption_key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        
        if encryption_key:
            logger.info("Using encryption key from CONFIG_ENCRYPTION_KEY environment variable")
            return encryption_key
        
        # Fallback: derive from other environment variables
        secret_key = os.environ.get('SECRET_KEY', '')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        if not secret_key and not db_password:
            logger.warning("No encryption key sources found in environment variables")
            # Generate a default key for development (not secure for production)
            default_key = "development-config-encryption-key-not-for-production"
            logger.warning("Using default development encryption key - NOT SECURE FOR PRODUCTION")
            return default_key
        
        # Combine available secrets to create encryption key
        combined = f"{secret_key}{db_password}"
        key_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        logger.info("Derived encryption key from environment variables")
        return key_hash
    
    def get_sensitive_keys_in_config(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Get list of sensitive keys found in configuration"""
        sensitive_keys = []
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                # Recursively check nested dictionaries
                sensitive_keys.extend(self.get_sensitive_keys_in_config(value, current_path))
            elif self._is_sensitive_key(key):
                sensitive_keys.append(current_path)
        
        return sensitive_keys