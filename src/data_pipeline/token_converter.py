"""
Token Converter Utility
Provides conversion between different token representations
"""

from datetime import datetime
from typing import Optional
import pandas as pd

from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain


def token_config_to_discovered(config, 
                              price_data: Optional[pd.DataFrame] = None) -> DiscoveredToken:
    """
    Convert TokenConfig to DiscoveredToken for compatibility with ML/RL systems
    
    Args:
        config: TokenConfig from corpus collection
        price_data: Optional OHLCV data to extract current price/volume
        
    Returns:
        DiscoveredToken compatible with ML/RL components
    """
    # Map network string to Chain enum
    chain = Chain.ETHEREUM  # default
    if config.network:
        chain_map = {
            'eth': Chain.ETHEREUM,
            'ethereum': Chain.ETHEREUM,
            'sol': Chain.SOLANA,
            'solana': Chain.SOLANA,
            'base': Chain.BASE,
            'bnb': Chain.BNB_CHAIN,
            'polygon': Chain.POLYGON,
            'arbitrum': Chain.ARBITRUM,
            'optimism': Chain.OPTIMISM,
            'avalanche': Chain.AVALANCHE,
        }
        chain = chain_map.get(config.network.lower(), Chain.ETHEREUM)
    
    # Extract current market data if available
    current_price = None
    volume_24h = None
    price_change_24h = None
    
    if price_data is not None and not price_data.empty:
        current_price = float(price_data['close'].iloc[-1])
        
        # Calculate 24h volume if we have enough data
        if len(price_data) >= 24:
            volume_24h = float(price_data['volume'].tail(24).sum())
            price_24h_ago = float(price_data['close'].iloc[-24])
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    # Create DiscoveredToken
    return DiscoveredToken(
        address=config.contract_address or config.coingecko_id,
        chain=chain,
        symbol=config.symbol,
        name=config.symbol,  # Use symbol as name for corpus tokens
        discovered_at=datetime.now(),  # Use current time for corpus collection
        discovery_source='corpus',  # Special source identifier for corpus tokens
        status=TokenStatus.VALIDATED,  # Corpus tokens are pre-validated
        price_usd=current_price,
        volume_24h=volume_24h,
        price_change_24h=price_change_24h,
        tags=['corpus', config.category],
        metadata={
            'coingecko_id': config.coingecko_id,
            'extract_stablecoin_features': config.extract_stablecoin_features,
            'original_network': config.network,
        }
    )


def discovered_to_token_config(token: DiscoveredToken):
    """
    Convert DiscoveredToken to TokenConfig for corpus collection
    
    Args:
        token: DiscoveredToken from discovery system
        
    Returns:
        TokenConfig compatible with corpus collection
    """
    # Import here to avoid circular dependency
    from src.data_pipeline.multi_granularity_collector import TokenConfig
    
    # Determine category from tags or default
    category = 'crypto'  # default
    if token.tags:
        if 'stablecoin' in token.tags:
            category = 'stablecoin'
        elif 'defi' in token.tags:
            category = 'defi'
        elif 'wrapped' in token.tags:
            category = 'wrapped_crypto'
    
    # Extract stablecoin features for known stablecoins
    extract_stablecoin = category == 'stablecoin' or (
        token.symbol and token.symbol.upper() in ['USDC', 'USDT', 'DAI', 'BUSD']
    )
    
    return TokenConfig(
        symbol=token.symbol,
        contract_address=token.address if token.address != token.symbol else None,
        network=token.chain.value,
        coingecko_id=token.metadata.get('coingecko_id', token.symbol.lower()),
        category=category,
        extract_stablecoin_features=extract_stablecoin
    )