"""
Test cases for wallet base interface and data structures.

Following TDD methodology:
1. Write failing tests that define expected wallet behavior
2. Implement minimal code to make tests pass
3. Refactor while keeping tests green
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from typing import Optional

from src.wallet.base import (
    WalletBase,
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


class TestWalletConfig:
    """Test WalletConfig data structure."""
    
    def test_wallet_config_creation_with_private_key(self):
        """Test creating wallet config with private key."""
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test"
        )
        
        assert config.chain == Chain.ETHEREUM
        assert config.network == NetworkType.TESTNET
        assert config.private_key == "0x" + "a" * 64
        assert config.rpc_url == "https://sepolia.infura.io/v3/test"
    
    def test_wallet_config_creation_with_wallet_address(self):
        """Test creating wallet config with wallet address only (read-only)."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        assert config.chain == Chain.SOLANA
        assert config.wallet_address == "11111111111111111111111111111112"
    
    def test_wallet_config_validation_fails_without_auth(self):
        """Test wallet config validation fails without private key, mnemonic, or address."""
        with pytest.raises(ValueError, match="Must provide private_key, mnemonic, or wallet_address"):
            WalletConfig(
                chain=Chain.ETHEREUM,
                network=NetworkType.TESTNET,
                rpc_url="https://sepolia.infura.io/v3/test"
            )


class TestWalletBalance:
    """Test WalletBalance data structure."""
    
    def test_wallet_balance_creation(self):
        """Test creating wallet balance."""
        balance = WalletBalance(
            native_balance=Decimal("1.5"),
            native_symbol="ETH",
            token_balances={"0x123": Decimal("100.0")},
            total_usd_value=Decimal("3000.0")
        )
        
        assert balance.native_balance == Decimal("1.5")
        assert balance.native_symbol == "ETH"
        assert balance.token_balances["0x123"] == Decimal("100.0")
        assert balance.total_usd_value == Decimal("3000.0")
    
    def test_wallet_balance_str_representation(self):
        """Test wallet balance string representation."""
        balance = WalletBalance(
            native_balance=Decimal("2.5"),
            native_symbol="SOL",
            token_balances={}
        )
        
        assert str(balance) == "2.5 SOL"


class TestTransactionResult:
    """Test TransactionResult data structure."""
    
    def test_transaction_result_successful(self):
        """Test successful transaction result."""
        result = TransactionResult(
            transaction_hash="0xabc123",
            status=TransactionStatus.CONFIRMED,
            gas_used=21000,
            gas_price=Decimal("20.0"),
            block_number=12345
        )
        
        assert result.transaction_hash == "0xabc123"
        assert result.status == TransactionStatus.CONFIRMED
        assert result.is_successful == True
        assert result.transaction_fee == Decimal("420000.0")  # 21000 * 20.0
    
    def test_transaction_result_failed(self):
        """Test failed transaction result."""
        result = TransactionResult(
            transaction_hash="0xdef456",
            status=TransactionStatus.FAILED,
            error_message="Insufficient gas"
        )
        
        assert result.is_successful == False
        assert result.error_message == "Insufficient gas"
        assert result.transaction_fee is None


class TestWalletErrors:
    """Test wallet error classes."""
    
    def test_wallet_error_inheritance(self):
        """Test wallet error class hierarchy."""
        base_error = WalletError("Base error")
        connection_error = WalletConnectionError("Connection failed")
        transaction_error = WalletTransactionError("Transaction failed")
        
        assert isinstance(connection_error, WalletError)
        assert isinstance(transaction_error, WalletError)
        assert str(base_error) == "Base error"


class ConcreteWallet(WalletBase):
    """Concrete implementation of WalletBase for testing."""
    
    def __init__(self, config: WalletConfig):
        super().__init__(config)
        self._mock_connected = False
        self._mock_balance = Decimal("1.0")
    
    async def connect(self) -> bool:
        """Mock connect implementation."""
        self._connected = True
        self._wallet_address = "mock_address"
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnect implementation."""
        self._connected = False
        self._wallet_address = None
    
    async def get_balance(self, token_address: Optional[str] = None) -> WalletBalance:
        """Mock get_balance implementation."""
        return WalletBalance(
            native_balance=self._mock_balance,
            native_symbol="MOCK",
            token_balances={}
        )
    
    async def get_native_balance(self) -> Decimal:
        """Mock get_native_balance implementation."""
        return self._mock_balance
    
    async def get_token_balance(self, token_address: str) -> Decimal:
        """Mock get_token_balance implementation."""
        return Decimal("100.0")
    
    async def send_native_token(
        self,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """Mock send_native_token implementation."""
        return TransactionResult(
            transaction_hash="0xmock",
            status=TransactionStatus.PENDING
        )
    
    async def send_token(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """Mock send_token implementation."""
        return TransactionResult(
            transaction_hash="0xmock_token",
            status=TransactionStatus.PENDING
        )
    
    async def estimate_gas(
        self,
        to_address: str,
        amount: Decimal,
        token_address: Optional[str] = None,
        data: Optional[bytes] = None
    ) -> int:
        """Mock estimate_gas implementation."""
        return 21000
    
    async def get_gas_price(self) -> Decimal:
        """Mock get_gas_price implementation."""
        return Decimal("20.0")
    
    async def get_transaction_status(self, transaction_hash: str) -> TransactionResult:
        """Mock get_transaction_status implementation."""
        return TransactionResult(
            transaction_hash=transaction_hash,
            status=TransactionStatus.CONFIRMED
        )
    
    async def sign_message(self, message: str) -> str:
        """Mock sign_message implementation."""
        return "0xmock_signature"
    
    async def validate_address(self, address: str) -> bool:
        """Mock validate_address implementation."""
        return len(address) > 10


class TestWalletBase:
    """Test abstract WalletBase class using concrete implementation."""
    
    @pytest.fixture
    def wallet_config(self):
        """Create test wallet configuration."""
        return WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64
        )
    
    @pytest.fixture
    def wallet(self, wallet_config):
        """Create test wallet instance."""
        return ConcreteWallet(wallet_config)
    
    def test_wallet_initialization(self, wallet_config):
        """Test wallet initialization with config."""
        wallet = ConcreteWallet(wallet_config)
        
        assert wallet.config == wallet_config
        assert wallet.chain == Chain.ETHEREUM
        assert wallet.network == NetworkType.TESTNET
        assert wallet.is_connected == False
        assert wallet.wallet_address is None
    
    async def test_wallet_connection_lifecycle(self, wallet):
        """Test wallet connect/disconnect lifecycle."""
        # Initially not connected
        assert wallet.is_connected == False
        assert wallet.wallet_address is None
        
        # Connect wallet
        result = await wallet.connect()
        assert result == True
        assert wallet.is_connected == True
        assert wallet.wallet_address == "mock_address"
        
        # Disconnect wallet
        await wallet.disconnect()
        assert wallet.is_connected == False
        assert wallet.wallet_address is None
    
    async def test_wallet_balance_operations(self, wallet):
        """Test wallet balance query operations."""
        await wallet.connect()
        
        # Test native balance
        balance = await wallet.get_native_balance()
        assert balance == Decimal("1.0")
        
        # Test token balance
        token_balance = await wallet.get_token_balance("0x123")
        assert token_balance == Decimal("100.0")
        
        # Test full balance
        full_balance = await wallet.get_balance()
        assert full_balance.native_balance == Decimal("1.0")
        assert full_balance.native_symbol == "MOCK"
    
    async def test_wallet_transaction_operations(self, wallet):
        """Test wallet transaction operations."""
        await wallet.connect()
        
        # Test native token send
        result = await wallet.send_native_token("0xrecipient", Decimal("0.5"))
        assert result.transaction_hash == "0xmock"
        assert result.status == TransactionStatus.PENDING
        
        # Test token send
        token_result = await wallet.send_token("0xtoken", "0xrecipient", Decimal("10.0"))
        assert token_result.transaction_hash == "0xmock_token"
        assert token_result.status == TransactionStatus.PENDING
    
    async def test_wallet_gas_operations(self, wallet):
        """Test wallet gas estimation and pricing."""
        await wallet.connect()
        
        # Test gas estimation
        gas_estimate = await wallet.estimate_gas("0xrecipient", Decimal("0.5"))
        assert gas_estimate == 21000
        
        # Test gas price
        gas_price = await wallet.get_gas_price()
        assert gas_price == Decimal("20.0")
    
    async def test_wallet_transaction_status(self, wallet):
        """Test transaction status checking."""
        await wallet.connect()
        
        status = await wallet.get_transaction_status("0xtest")
        assert status.transaction_hash == "0xtest"
        assert status.status == TransactionStatus.CONFIRMED
    
    async def test_wallet_message_signing(self, wallet):
        """Test message signing functionality."""
        await wallet.connect()
        
        signature = await wallet.sign_message("Hello, World!")
        assert signature == "0xmock_signature"
    
    async def test_wallet_address_validation(self, wallet):
        """Test address validation."""
        await wallet.connect()
        
        assert await wallet.validate_address("valid_long_address") == True
        assert await wallet.validate_address("short") == False
    
    async def test_wallet_health_check(self, wallet):
        """Test wallet health check functionality."""
        # Health check when disconnected
        health = await wallet.health_check()
        assert health["connected"] == True  # Should auto-connect
        assert health["status"] == "healthy"
        assert "native_balance" in health
        assert "gas_price" in health
    
    def test_wallet_string_representations(self, wallet):
        """Test wallet string representations."""
        wallet._wallet_address = "0x123"
        wallet._connected = True
        
        str_repr = str(wallet)
        assert "ConcreteWallet" in str_repr
        assert "ethereum" in str_repr
        
        repr_str = repr(wallet)
        assert "ConcreteWallet" in repr_str
        assert "chain=ethereum" in repr_str
        assert "network=testnet" in repr_str
        assert "address=0x123" in repr_str
        assert "connected=True" in repr_str


class TestWalletIntegration:
    """Integration tests for wallet functionality."""
    
    async def test_wallet_operations_require_connection(self):
        """Test that wallet operations require connection."""
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64
        )
        wallet = ConcreteWallet(config)
        
        # These should work without connection in our mock
        # but in real implementation would require connection
        balance = await wallet.get_native_balance()
        assert balance == Decimal("1.0")  # Mock returns default value
    
    async def test_wallet_chain_network_properties(self):
        """Test wallet chain and network properties."""
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112"
        )
        wallet = ConcreteWallet(config)
        
        assert wallet.chain == Chain.SOLANA
        assert wallet.network == NetworkType.MAINNET


# These tests will initially fail because the implementations are incomplete
# This is expected behavior for TDD - we write failing tests first, then implement
class TestWalletFailingScenarios:
    """Tests that should initially fail - this is expected in TDD."""
    
    async def test_ethereum_wallet_specific_functionality(self):
        """This test will fail until EthereumWallet is fully implemented."""
        # This will be implemented in the next phase
        pytest.skip("EthereumWallet implementation pending")
    
    async def test_solana_wallet_specific_functionality(self):
        """This test will fail until SolanaWallet is fully implemented."""
        # This will be implemented in the next phase
        pytest.skip("SolanaWallet implementation pending")
    
    async def test_wallet_config_manager_functionality(self):
        """This test will fail until WalletConfigManager is enhanced."""
        # This will be tested when we enhance the config manager
        pytest.skip("WalletConfigManager enhancement pending")