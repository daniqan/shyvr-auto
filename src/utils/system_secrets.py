"""
System Secrets handler for the RLTE application
Provides a centralized interface for accessing Google Cloud Secret Manager
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
import structlog
from google.cloud import secretmanager
from google.api_core import exceptions as gcp_exceptions


class SystemSecrets:
    """
    Centralized secret management using Google Cloud Secret Manager
    
    This class provides a clean interface for accessing all secrets needed
    by the application, with caching to minimize API calls.
    """
    
    def __init__(self, project_id: str = "shvyr-ai-bots"):
        """
        Initialize the Secret Manager
        
        Args:
            project_id: GCP project ID containing the secrets
        """
        self.project_id = project_id
        self.logger = structlog.get_logger().bind(component="SystemSecrets")
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            self._available = True
            self.logger.info("Secret Manager initialized", project=project_id)
        except Exception as e:
            self.client = None
            self._available = False
            self.logger.error("Failed to initialize Secret Manager client", error=str(e))
    
    @lru_cache(maxsize=32)
    def _get_secret(self, secret_name: str) -> Optional[str]:
        """
        Internal method to fetch a secret from Google Cloud Secret Manager
        
        Args:
            secret_name: Name of the secret in Secret Manager
            
        Returns:
            Secret value as string, or None if not found/error
        """
        if not self._available:
            self.logger.warning(f"Secret Manager not available, cannot fetch {secret_name}")
            return None
        
        try:
            # Build the resource name of the secret version
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            
            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})
            
            # Return the decoded payload
            secret_value = response.payload.data.decode("UTF-8")
            self.logger.debug(f"Successfully retrieved secret: {secret_name}")
            return secret_value
            
        except gcp_exceptions.NotFound:
            self.logger.warning(f"Secret not found: {secret_name}")
            return None
        except gcp_exceptions.PermissionDenied:
            self.logger.error(f"Permission denied accessing secret: {secret_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            return None
    
    # Database secrets
    @property
    def db_password(self) -> Optional[str]:
        """Get database password"""
        return self._get_secret("DB_PASSWORD")
    
    @property
    def db_host(self) -> Optional[str]:
        """Get database host"""
        return self._get_secret("DB_HOST")
    
    @property
    def database_url(self) -> Optional[str]:
        """Get full database URL"""
        return self._get_secret("DATABASE_URL")
    
    # API Keys
    @property
    def telegram_token(self) -> Optional[str]:
        """Get Telegram bot token"""
        return self._get_secret("TELEGRAM_TOKEN")
    
    @property
    def coingecko_api_key(self) -> Optional[str]:
        """Get CoinGecko API key"""
        return self._get_secret("COINGECKO_API_KEY")
    
    @property
    def lunarcrush_api_key(self) -> Optional[str]:
        """Get LunarCrush API key"""
        return self._get_secret("LUNARCRUSH_API_KEY")
    
    @property
    def helius_api_key(self) -> Optional[str]:
        """Get Helius API key"""
        return self._get_secret("HELIUS_API_KEY")
    
    @property
    def etherscan_api_key(self) -> Optional[str]:
        """Get Etherscan API key"""
        return self._get_secret("ETHERSCAN_API_KEY")
    
    @property
    def birdeye_api_key(self) -> Optional[str]:
        """Get Birdeye API key"""
        return self._get_secret("BIRDEYE_API_KEY")
    
    @property
    def quicknode_api_key(self) -> Optional[str]:
        """Get QuickNode API key"""
        return self._get_secret("QUICKNODE_API_KEY")
    
    @property
    def moralis_api_key(self) -> Optional[str]:
        """Get Moralis API key"""
        return self._get_secret("MORALIS_API_KEY")
    
    # Security secrets
    @property
    def secret_key(self) -> Optional[str]:
        """Get application secret key (used for JWT signing and other security operations)"""
        return self._get_secret("SECRET_KEY")
    
    # Cloud SQL connection
    @property
    def cloud_sql_connection_name(self) -> Optional[str]:
        """Get Cloud SQL connection name"""
        return self._get_secret("CLOUD_SQL_CONNECTION_NAME")
    
    # Telegram admin users
    @property
    def telegram_admin_users(self) -> Optional[str]:
        """Get Telegram admin users (comma-separated IDs)"""
        return self._get_secret("TELEGRAM_ADMIN_USERS")
    
    def get_telegram_admin_user_ids(self) -> list[int]:
        """Get Telegram admin user IDs as a list of integers"""
        admin_users = self.telegram_admin_users
        if not admin_users:
            return []
        
        try:
            return [int(uid.strip()) for uid in admin_users.split(",") if uid.strip()]
        except ValueError as e:
            self.logger.error(f"Error parsing TELEGRAM_ADMIN_USERS: {e}")
            return []
    
    # Generic secret access
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get any secret by name
        
        Args:
            secret_name: Name of the secret in Secret Manager
            
        Returns:
            Secret value as string, or None if not found
        """
        return self._get_secret(secret_name)
    
    def get_all_api_keys(self) -> Dict[str, Optional[str]]:
        """
        Get all API keys as a dictionary
        
        Returns:
            Dictionary of API key names to values
        """
        return {
            "coingecko": self.coingecko_api_key,
            "lunarcrush": self.lunarcrush_api_key,
            "helius": self.helius_api_key,
            "etherscan": self.etherscan_api_key,
            "birdeye": self.birdeye_api_key,
            "quicknode": self.quicknode_api_key,
            "moralis": self.moralis_api_key,
        }
    
    def get_database_config(self, use_local_proxy: bool = None) -> Dict[str, Optional[str]]:
        """
        Get database configuration as a dictionary
        
        Args:
            use_local_proxy: If True, use localhost:5433 for Cloud SQL proxy connection.
                           If None, auto-detect based on environment.
        
        Returns:
            Dictionary with database configuration
        """
        import os
        
        # Auto-detect if we should use local proxy
        if use_local_proxy is None:
            # Use local proxy if we're in development environment or if proxy is running
            use_local_proxy = (
                os.environ.get('ENVIRONMENT') == 'development' or
                os.path.exists('/tmp/.cloud-sql-proxy.lock') or
                os.system('lsof -i:5433 >/dev/null 2>&1') == 0  # Check if port 5433 is in use
            )
        
        # Get password from Secret Manager
        password = self.db_password
        if not password:
            return {"error": "No database password available"}
        
        # Determine host and port based on connection method
        if use_local_proxy:
            # Use localhost with Cloud SQL proxy
            host = "localhost"
            port = "5433"
        else:
            # Use direct connection (for production Cloud Run)
            database_url = self.database_url
            if database_url:
                # Parse production DATABASE_URL
                import re
                match = re.search(r'://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
                if match:
                    host = match.group(3)
                    port = match.group(4)
                else:
                    host = self.db_host or "34.60.72.85"
                    port = "5432"
            else:
                host = self.db_host or "34.60.72.85"
                port = "5432"
        
        return {
            "username": "rlte_prod_user",
            "password": password,
            "host": host,
            "port": port,
            "database": "shyvr_rlte_prod",
            "url": f"postgresql://rlte_prod_user:{password}@{host}:{port}/shyvr_rlte_prod"
        }
    
    def is_available(self) -> bool:
        """
        Check if Secret Manager is available
        
        Returns:
            True if Secret Manager client is initialized and available
        """
        return self._available
    
    def clear_cache(self):
        """Clear the secret cache to force refresh on next access"""
        self._get_secret.cache_clear()
        self.logger.info("Secret cache cleared")


# Global instance for easy access
_system_secrets: Optional[SystemSecrets] = None


def get_system_secrets(project_id: str = "shvyr-ai-bots") -> SystemSecrets:
    """
    Get or create the global SystemSecrets instance
    
    Args:
        project_id: GCP project ID
        
    Returns:
        SystemSecrets instance
    """
    global _system_secrets
    if _system_secrets is None:
        _system_secrets = SystemSecrets(project_id)
    return _system_secrets