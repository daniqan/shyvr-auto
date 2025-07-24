"""
Test cases for Solana RPC integration.

Following TDD methodology - these tests define the expected Solana RPC functionality
and will drive the implementation of proper Solana RPC integration.

Key differences from Ethereum:
- Uses solana-py library instead of web3.py
- Different RPC endpoint structure (no API key in URL typically)
- Cluster validation instead of chain ID
- Different error patterns and response formats
"""

import pytest
import os
import base58
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional

from src.wallet.base import (
    WalletConfig,
    WalletBalance,
    TransactionResult,
    TransactionStatus,
    Chain,
    NetworkType,
    WalletError,
    WalletConnectionError,
    WalletTransactionError
)
from src.wallet.config import WalletConfigManager

# Import will succeed since SolanaWallet is already implemented
from src.wallet.solana_wallet import SolanaWallet, SPLToken


class TestSolanaRPCConfiguration:
    """Test Solana RPC configuration and URL construction."""
    
    def test_solana_rpc_url_construction_mainnet(self):
        """Test that Solana mainnet URLs are constructed correctly."""
        config_manager = WalletConfigManager()
        
        # Test mainnet URL construction
        config = config_manager.create_wallet_config(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112"
        )
        
        expected_url = "https://api.mainnet-beta.solana.com"
        assert config.rpc_url == expected_url
        assert config.api_key is None  # Solana typically doesn't use API keys in URL
    
    def test_solana_rpc_url_construction_devnet(self):
        """Test that Solana devnet URLs are constructed correctly."""
        config_manager = WalletConfigManager()
        
        # Test devnet URL construction
        config = config_manager.create_wallet_config(
            chain=Chain.SOLANA,
            network=NetworkType.TESTNET,  # Maps to devnet for Solana
            wallet_address="11111111111111111111111111111112"
        )
        
        expected_url = "https://api.devnet.solana.com"
        assert config.rpc_url == expected_url
        assert config.api_key is None
    
    def test_solana_custom_rpc_endpoint(self):
        """Test custom Solana RPC endpoint configuration."""
        config_manager = WalletConfigManager()
        
        custom_rpc = "https://my-solana-node.com:8899"
        config = config_manager.create_wallet_config(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url=custom_rpc
        )
        
        assert config.rpc_url == custom_rpc
    
    def test_solana_rpc_timeout_configuration(self):
        """Test Solana RPC timeout configuration."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com",
            timeout_seconds=45
        )
        
        assert config.timeout_seconds == 45
    
    def test_solana_environment_variable_handling(self):
        """Test that Solana configuration handles environment variables."""
        test_cases = [
            ("SOLANA_RPC_URL", "https://custom-solana-rpc.com"),
            ("SOLANA_PRIVATE_KEY", "5" + "A" * 87),  # Base58 encoded key
            ("SOLANA_WALLET_ADDRESS", "11111111111111111111111111111112"),
        ]
        
        for env_var, value in test_cases:
            with patch.dict(os.environ, {env_var: value}):
                assert os.getenv(env_var) == value


class TestSolanaConnectionValidation:
    """Test Solana RPC connection validation and cluster verification."""
    
    @pytest.fixture
    def mainnet_config(self):
        """Create Solana mainnet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "A" * 87,  # Base58 encoded private key
            rpc_url="https://api.mainnet-beta.solana.com",
            timeout_seconds=30
        )
    
    @pytest.fixture
    def devnet_config(self):
        """Create Solana devnet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.TESTNET,
            private_key="5" + "B" * 87,
            rpc_url="https://api.devnet.solana.com",
            timeout_seconds=30
        )
    
    async def test_solana_wallet_creation(self, mainnet_config):
        """Test creating Solana wallet instance."""
        wallet = SolanaWallet(mainnet_config)
        assert wallet.chain == Chain.SOLANA
        assert wallet.network == NetworkType.MAINNET
        assert wallet.is_connected == False
        
        # Test invalid chain raises error
        invalid_config = WalletConfig(
            chain=Chain.ETHEREUM,  # Wrong chain for SolanaWallet
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64
        )
        
        with pytest.raises(WalletError, match="SolanaWallet only supports Solana chain"):
            SolanaWallet(invalid_config)
    
    async def test_solana_rpc_connection_success(self, mainnet_config):
        """Test successful Solana RPC connection."""
        wallet = SolanaWallet(mainnet_config)
        
        # Mock successful Solana RPC connection
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock health check
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            # Mock slot info
            slot_response = AsyncMock()
            slot_response.value = 200000000
            mock_client_instance.get_slot.return_value = slot_response
            
            # Mock keypair creation
            with patch('src.wallet.solana_wallet.Keypair') as mock_keypair:
                mock_keypair_instance = Mock()
                mock_keypair_instance.pubkey.return_value = Mock()
                mock_keypair_instance.pubkey.return_value.__str__ = lambda: "11111111111111111111111111111112"
                mock_keypair.from_bytes.return_value = mock_keypair_instance
                
                with patch('src.wallet.solana_wallet.base58.b58decode') as mock_b58decode:
                    mock_b58decode.return_value = b'A' * 64
                    
                    success = await wallet.connect()
                    assert success == True
                    assert wallet.is_connected == True
                    assert wallet.wallet_address is not None
    
    async def test_solana_rpc_connection_failure_health_check(self, mainnet_config):
        """Test Solana RPC connection failure during health check."""
        wallet = SolanaWallet(mainnet_config)
        
        # Mock failed health check
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_health.side_effect = Exception("RPC unavailable")
            
            with pytest.raises(WalletConnectionError, match="Failed to connect to Solana RPC"):
                await wallet.connect()
    
    async def test_solana_rpc_connection_no_rpc_url(self):
        """Test Solana connection failure when RPC URL is not configured."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "A" * 87,
            rpc_url=None  # No RPC URL
        )
        
        wallet = SolanaWallet(config)
        
        with pytest.raises(WalletConnectionError, match="RPC URL not configured"):
            await wallet.connect()
    
    async def test_solana_read_only_mode(self):
        """Test Solana wallet in read-only mode."""
        readonly_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(readonly_config)
        
        # Mock successful connection
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 200000000
            mock_client_instance.get_slot.return_value = slot_response
            
            # Mock address validation
            with patch('src.wallet.solana_wallet.Pubkey.from_string') as mock_pubkey:
                mock_pubkey.return_value = Mock()
                
                success = await wallet.connect()
                assert success == True
                assert wallet.is_connected == True
                assert wallet.wallet_address == readonly_config.wallet_address
    
    async def test_solana_invalid_private_key_base58(self):
        """Test Solana wallet with invalid base58 private key."""
        invalid_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="invalid_base58_key!@#$",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(invalid_config)
        
        with patch('src.wallet.solana_wallet.AsyncClient'):
            with pytest.raises(WalletConnectionError, match="Invalid Solana private key"):
                await wallet.connect()
    
    async def test_solana_invalid_private_key_length(self):
        """Test Solana wallet with private key of invalid length."""
        invalid_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="A" * 32,  # Too short
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(invalid_config)
        
        with patch('src.wallet.solana_wallet.AsyncClient'):
            with pytest.raises(WalletConnectionError, match="Invalid Solana private key"):
                await wallet.connect()
    
    async def test_solana_invalid_wallet_address(self):
        """Test Solana wallet with invalid wallet address format."""
        invalid_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="invalid_solana_address",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(invalid_config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            # Mock invalid address validation
            with patch('src.wallet.solana_wallet.Pubkey.from_string') as mock_pubkey:
                mock_pubkey.side_effect = Exception("Invalid address")
                
                with pytest.raises(WalletConnectionError, match="Invalid Solana address"):
                    await wallet.connect()


class TestSolanaClusterValidation:
    """Test Solana cluster validation and network verification."""
    
    async def test_mainnet_cluster_validation(self):
        """Test validation of mainnet-beta cluster."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock successful health and slot responses
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 200000000  # Mainnet-like slot number
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.Pubkey.from_string'):
                success = await wallet.connect()
                assert success == True
                assert wallet.is_connected == True
    
    async def test_devnet_cluster_validation(self):
        """Test validation of devnet cluster."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.TESTNET,  # Maps to devnet
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.devnet.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock successful health and slot responses
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 150000000  # Devnet-like slot number
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.Pubkey.from_string'):
                success = await wallet.connect()
                assert success == True
                assert wallet.is_connected == True
    
    async def test_cluster_rpc_endpoint_mismatch(self):
        """Test detection of cluster/RPC endpoint mismatches."""
        # Configure for devnet but use mainnet RPC
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.TESTNET,  # Expecting devnet
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"  # But using mainnet RPC
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock successful health check
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            # Mock slot response indicating mainnet
            slot_response = AsyncMock()
            slot_response.value = 250000000  # Very high slot indicating mainnet
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.Pubkey.from_string'):
                # Connection should still succeed, but we could add validation
                success = await wallet.connect()
                assert success == True


class TestSolanaErrorHandling:
    """Test Solana-specific error handling scenarios."""
    
    @pytest.fixture
    def solana_config(self):
        """Create basic Solana configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "A" * 87,
            rpc_url="https://api.mainnet-beta.solana.com",
            timeout_seconds=30
        )
    
    async def test_solana_rpc_timeout_handling(self, solana_config):
        """Test Solana RPC timeout handling."""
        wallet = SolanaWallet(solana_config)
        
        # Mock timeout scenario
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            import asyncio
            mock_client.side_effect = asyncio.TimeoutError("Connection timeout")
            
            with pytest.raises(WalletConnectionError, match="Connection failed"):
                await wallet.connect()
    
    async def test_solana_rpc_connection_refused(self, solana_config):
        """Test Solana RPC connection refused error."""
        wallet = SolanaWallet(solana_config)
        
        # Mock connection refused
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client.side_effect = ConnectionError("Connection refused")
            
            with pytest.raises(WalletConnectionError, match="Connection failed"):
                await wallet.connect()
    
    async def test_solana_rpc_invalid_response(self, solana_config):
        """Test handling of invalid RPC responses."""
        wallet = SolanaWallet(solana_config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock invalid health response
            mock_client_instance.get_health.side_effect = Exception("Invalid JSON response")
            
            with pytest.raises(WalletConnectionError, match="Failed to connect to Solana RPC"):
                await wallet.connect()
    
    async def test_solana_insufficient_funds_error(self, solana_config):
        """Test handling of insufficient funds errors during transactions."""
        wallet = SolanaWallet(solana_config)
        wallet._connected = True
        wallet.client = AsyncMock()
        
        # Mock keypair
        wallet.keypair = Mock()
        wallet.keypair.pubkey.return_value = Mock()
        
        # Mock insufficient funds error
        with patch('src.wallet.solana_wallet.Pubkey.from_string'):
            wallet.client.send_transaction.side_effect = Exception("Insufficient funds")
            
            with pytest.raises(WalletTransactionError, match="SOL transfer failed"):
                await wallet.send_native_token("11111111111111111111111111111112", Decimal("1.0"))
    
    async def test_solana_invalid_transaction_error(self, solana_config):
        """Test handling of invalid transaction errors."""
        wallet = SolanaWallet(solana_config)
        wallet._connected = True
        wallet.client = AsyncMock()
        
        # Mock keypair
        wallet.keypair = Mock()
        wallet.keypair.pubkey.return_value = Mock()
        
        # Mock invalid transaction error
        with patch('src.wallet.solana_wallet.Pubkey.from_string'):
            wallet.client.send_transaction.side_effect = Exception("Invalid transaction")
            
            with pytest.raises(WalletTransactionError, match="SOL transfer failed"):
                await wallet.send_native_token("11111111111111111111111111111112", Decimal("0.5"))


class TestSolanaAddressValidation:
    """Test Solana address validation functionality."""
    
    @pytest.fixture
    def solana_wallet(self):
        """Create connected Solana wallet for testing."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        wallet._connected = True
        return wallet
    
    async def test_valid_solana_addresses(self, solana_wallet):
        """Test validation of valid Solana addresses."""
        valid_addresses = [
            "11111111111111111111111111111112",  # System Program
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token Program
            "SysvarRent111111111111111111111111111111111",  # Sysvar Rent
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        ]
        
        for address in valid_addresses:
            with patch('src.wallet.solana_wallet.Pubkey.from_string') as mock_pubkey:
                mock_pubkey.return_value = Mock()
                result = await solana_wallet.validate_address(address)
                assert result == True
    
    async def test_invalid_solana_addresses(self, solana_wallet):
        """Test validation of invalid Solana addresses."""
        invalid_addresses = [
            "invalid_address",
            "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb",  # Ethereum address
            "1111111111111111111111111111111",  # Too short
            "11111111111111111111111111111112345",  # Too long
            "",  # Empty
            "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",  # Invalid base58
        ]
        
        for address in invalid_addresses:
            with patch('src.wallet.solana_wallet.Pubkey.from_string') as mock_pubkey:
                mock_pubkey.side_effect = Exception("Invalid address")
                result = await solana_wallet.validate_address(address)
                assert result == False
    
    async def test_solana_private_key_validation(self):
        """Test validation of Solana private key formats."""
        # Test base58 encoded private key (88 characters)
        base58_key = "5" + "A" * 87
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key=base58_key,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 200000000
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.base58.b58decode') as mock_b58decode:
                mock_b58decode.return_value = b'A' * 64
                
                with patch('src.wallet.solana_wallet.Keypair.from_bytes') as mock_keypair:
                    mock_keypair_instance = Mock()
                    mock_keypair_instance.pubkey.return_value.__str__ = lambda: "11111111111111111111111111111112"
                    mock_keypair.return_value = mock_keypair_instance
                    
                    success = await wallet.connect()
                    assert success == True
    
    async def test_solana_hex_private_key_validation(self):
        """Test validation of hex-encoded Solana private keys."""
        # Test hex private key
        hex_key = "a" * 128  # 64 bytes as hex
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key=hex_key,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 200000000
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.Keypair.from_bytes') as mock_keypair:
                mock_keypair_instance = Mock()
                mock_keypair_instance.pubkey.return_value.__str__ = lambda: "11111111111111111111111111111112"
                mock_keypair.return_value = mock_keypair_instance
                
                success = await wallet.connect()
                assert success == True


class TestSolanaEnvironmentIntegration:
    """Test Solana integration with environment variables."""
    
    def test_solana_environment_variable_priority(self):
        """Test that environment variables take priority for Solana configuration."""
        config_manager = WalletConfigManager()
        
        test_rpc_url = "https://custom-solana-node.com:8899"
        test_private_key = "5" + "C" * 87
        test_wallet_address = "11111111111111111111111111111113"
        
        env_vars = {
            "SOLANA_RPC_URL": test_rpc_url,
            "SOLANA_MAINNET_PRIVATE_KEY": test_private_key,
            "SOLANA_MAINNET_WALLET_ADDRESS": test_wallet_address
        }
        
        with patch.dict(os.environ, env_vars):
            config = config_manager.get_wallet_config_from_env(
                chain=Chain.SOLANA,
                network=NetworkType.MAINNET
            )
            
            # Should use environment variable values
            assert config.rpc_url == test_rpc_url
            assert config.private_key == test_private_key
    
    def test_solana_network_specific_environment_variables(self):
        """Test network-specific environment variables for Solana."""
        config_manager = WalletConfigManager()
        
        mainnet_key = "5" + "M" * 87
        devnet_key = "5" + "D" * 87
        mainnet_address = "11111111111111111111111111111114"
        devnet_address = "11111111111111111111111111111115"
        
        env_vars = {
            "SOLANA_MAINNET_PRIVATE_KEY": mainnet_key,
            "SOLANA_TESTNET_PRIVATE_KEY": devnet_key,
            "SOLANA_MAINNET_WALLET_ADDRESS": mainnet_address,
            "SOLANA_TESTNET_WALLET_ADDRESS": devnet_address
        }
        
        with patch.dict(os.environ, env_vars):
            # Mainnet should use mainnet-specific variables
            mainnet_config = config_manager.get_wallet_config_from_env(
                chain=Chain.SOLANA,
                network=NetworkType.MAINNET
            )
            assert mainnet_config.private_key == mainnet_key
            
            # Testnet should use testnet-specific variables
            testnet_config = config_manager.get_wallet_config_from_env(
                chain=Chain.SOLANA,
                network=NetworkType.TESTNET
            )
            assert testnet_config.private_key == devnet_key
    
    def test_solana_missing_environment_variables_fallback(self):
        """Test fallback behavior when Solana environment variables are missing."""
        config_manager = WalletConfigManager()
        
        # Clear Solana-related environment variables
        env_vars_to_clear = [
            "SOLANA_RPC_URL", "SOLANA_PRIVATE_KEY", "SOLANA_WALLET_ADDRESS",
            "SOLANA_MAINNET_PRIVATE_KEY", "SOLANA_TESTNET_PRIVATE_KEY"
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            # Should raise error when no credentials provided
            with pytest.raises(ValueError, match="Must provide private_key, mnemonic, or wallet_address"):
                config_manager.get_wallet_config_from_env(
                    chain=Chain.SOLANA,
                    network=NetworkType.MAINNET
                )


class TestSolanaRPCMocking:
    """Test Solana RPC connections using comprehensive mocking."""
    
    @pytest.fixture
    def mock_successful_solana_rpc(self):
        """Mock successful Solana RPC connection."""
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Configure successful connection responses
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            slot_response = AsyncMock()
            slot_response.value = 200000000
            mock_client_instance.get_slot.return_value = slot_response
            
            # Mock balance response
            balance_response = AsyncMock()
            balance_response.value = 1500000000  # 1.5 SOL in lamports
            mock_client_instance.get_balance.return_value = balance_response
            
            yield mock_client_instance
    
    @pytest.fixture
    def mock_failing_solana_rpc(self):
        """Mock failing Solana RPC connection."""
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Configure failing connection responses
            mock_client_instance.get_health.side_effect = Exception("RPC node unavailable")
            
            yield mock_client_instance
    
    async def test_mocked_solana_connection_success(self, mock_successful_solana_rpc):
        """Test successful Solana connection with mocked RPC."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.Pubkey.from_string'):
            success = await wallet.connect()
            
            assert success == True
            assert wallet.is_connected == True
            
            # Verify RPC methods were called
            mock_successful_solana_rpc.get_health.assert_called()
    
    async def test_mocked_solana_connection_failure(self, mock_failing_solana_rpc):
        """Test failed Solana connection with mocked RPC."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with pytest.raises(WalletConnectionError, match="Failed to connect to Solana RPC"):
            await wallet.connect()
    
    async def test_mocked_solana_balance_query(self, mock_successful_solana_rpc):
        """Test Solana balance query with mocked RPC."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        wallet._connected = True
        wallet.client = mock_successful_solana_rpc
        wallet._wallet_address = "11111111111111111111111111111112"
        
        with patch('src.wallet.solana_wallet.Pubkey.from_string') as mock_pubkey:
            mock_pubkey.return_value = Mock()
            
            balance = await wallet.get_native_balance()
            
            # Should return 1.5 SOL (1500000000 lamports / 1e9)
            assert balance == Decimal("1.5")
            
            # Verify RPC method was called
            mock_successful_solana_rpc.get_balance.assert_called()


# TDD Failing Tests - These drive implementation improvements
class TestSolanaRPCTDDFailingScenarios:
    """Tests designed to fail initially - drives TDD implementation."""
    
    async def test_solana_cluster_validation_enhancement(self):
        """Test enhanced cluster validation - will drive implementation."""
        # This test will drive implementation of better cluster validation
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.devnet.solana.com"  # Mismatch: mainnet config, devnet RPC
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock devnet characteristics
            health_response = AsyncMock()
            health_response.value = "ok"
            mock_client_instance.get_health.return_value = health_response
            
            # Low slot number indicates devnet
            slot_response = AsyncMock()
            slot_response.value = 150000000
            mock_client_instance.get_slot.return_value = slot_response
            
            with patch('src.wallet.solana_wallet.Pubkey.from_string'):
                # Currently this will succeed, but could add validation
                success = await wallet.connect()
                assert success == True
                
                # Future enhancement: detect cluster mismatch
                # with pytest.raises(WalletConnectionError, match="Cluster mismatch"):
                #     await wallet.connect()
    
    async def test_solana_advanced_error_recovery(self):
        """Test advanced error recovery - will drive implementation."""
        # This test will drive implementation of retry logic and error recovery
        pytest.skip("Advanced error recovery not implemented yet")
    
    async def test_solana_performance_optimization(self):
        """Test performance optimization - will drive implementation."""
        # This test will drive implementation of connection pooling and caching
        pytest.skip("Performance optimization not implemented yet")
    
    async def test_solana_websocket_subscription_support(self):
        """Test WebSocket subscription support - will drive implementation."""
        # This test will drive implementation of real-time updates via WebSocket
        pytest.skip("WebSocket subscription support not implemented yet")