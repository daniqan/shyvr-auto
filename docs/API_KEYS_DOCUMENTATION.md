# API Keys Documentation - Shyvr RLTE

This document provides comprehensive information about all API keys and secrets required by the Shyvr RLTE trading system.

## Core System Secrets (Required)

These secrets are essential for basic system operation and must be configured for any deployment.

### TELEGRAM_TOKEN
- **Purpose**: Bot authentication with Telegram API
- **Source**: @BotFather on Telegram
- **Format**: `bot1234567890:ABCDEF1234567890abcdef1234567890ABC`
- **Required**: Yes
- **Used by**: `src/telegram/bot.py`, webhook handling
- **Security**: Medium - allows bot control but not account access

### WEBHOOK_SECRET
- **Purpose**: Secure webhook endpoint authentication
- **Source**: Generated random string (recommended 32+ characters)
- **Format**: Alphanumeric string
- **Required**: Yes
- **Used by**: Webhook validation middleware
- **Security**: High - prevents unauthorized webhook calls

### DB_PASSWORD
- **Purpose**: PostgreSQL database authentication
- **Source**: Database admin or Cloud SQL
- **Format**: String (avoid special characters for compatibility)
- **Required**: Yes
- **Used by**: Database connections throughout application
- **Security**: High - grants database access

### DATABASE_URL
- **Purpose**: Complete PostgreSQL connection string
- **Source**: Constructed from database parameters
- **Format**: `postgresql://user:password@host:port/database`
- **Required**: Yes
- **Used by**: Database connection pooling
- **Security**: High - contains credentials and connection info

## Blockchain & RPC Secrets

These secrets enable blockchain data access and transaction capabilities.

### HELIUS_API_KEY
- **Purpose**: Solana blockchain data and RPC access
- **Source**: [Helius Labs](https://helius.dev)
- **Format**: UUID-like string
- **Required**: For Solana operations
- **Used by**: `src/blockchain/solana_client.py`, on-chain analytics
- **Rate Limits**: Varies by plan (10-1000 req/sec)
- **Security**: Medium - read-only blockchain access

### BIRDEYE_API_KEY
- **Purpose**: DeFi data aggregation and DEX analytics
- **Source**: [Birdeye](https://birdeye.so/developers)
- **Format**: API key string
- **Required**: For DeFi analytics
- **Used by**: `src/ml_analysis/dex_analytics.py`
- **Rate Limits**: 100-1000 requests/minute depending on plan
- **Security**: Low - public market data access

### ETHERSCAN_API_KEY
- **Purpose**: Ethereum blockchain explorer data
- **Source**: [Etherscan](https://etherscan.io/apis)
- **Format**: 34-character alphanumeric string
- **Required**: For Ethereum operations
- **Used by**: `src/blockchain/ethereum_client.py`
- **Rate Limits**: 5 calls/second for free tier
- **Security**: Low - public blockchain data

### ALCHEMY_API_KEY
- **Purpose**: General Ethereum/Polygon RPC provider
- **Source**: [Alchemy](https://www.alchemy.com)
- **Format**: 32-character alphanumeric string
- **Required**: For multi-chain operations
- **Used by**: `src/wallet/config.py`, RPC failover
- **Rate Limits**: Varies by plan
- **Security**: Medium - can submit transactions if private keys available

### ETHEREUM_API_KEY / BASE_API_KEY
- **Purpose**: Chain-specific RPC access
- **Source**: Various providers (Infura, Alchemy, QuickNode)
- **Format**: Provider-specific format
- **Required**: For specific chain operations
- **Used by**: Chain-specific wallet operations
- **Security**: Medium - blockchain access with transaction capability

## Trading & Wallet Secrets (CRITICAL)

**⚠️ WARNING**: These secrets control actual cryptocurrency wallets and should only be configured for production live trading. Improper handling can result in loss of funds.

### SOLANA_PRIVATE_KEY
- **Purpose**: Solana wallet private key for transaction signing
- **Source**: Solana wallet generation (Phantom, Solflare, CLI)
- **Format**: Base58 encoded string (~88 characters)
- **Required**: Only for live Solana trading
- **Used by**: `src/modes/live_mode.py`, transaction signing
- **Security**: CRITICAL - grants full wallet control
- **Storage**: Store in Secret Manager with restricted access

### ETHEREUM_PRIVATE_KEY
- **Purpose**: Ethereum wallet private key for transaction signing
- **Source**: Ethereum wallet generation (MetaMask, hardware wallet)
- **Format**: 64-character hexadecimal string (with or without 0x prefix)
- **Required**: Only for live Ethereum trading
- **Used by**: `src/modes/live_mode.py`, transaction signing
- **Security**: CRITICAL - grants full wallet control
- **Storage**: Store in Secret Manager with restricted access

### HYPERLIQUID_PRIVATE_KEY
- **Purpose**: Hyperliquid exchange private key
- **Source**: Hyperliquid exchange account setup
- **Format**: Exchange-specific format
- **Required**: Only for Hyperliquid trading
- **Used by**: `src/trading/hyperliquid_client.py`
- **Security**: CRITICAL - grants trading access
- **Storage**: Store in Secret Manager with restricted access

### HYPERLIQUID_API_KEY
- **Purpose**: Hyperliquid exchange API authentication
- **Source**: Hyperliquid API settings
- **Format**: API key string
- **Required**: For Hyperliquid operations
- **Used by**: `src/trading/hyperliquid_client.py`
- **Security**: High - grants trading API access

### WALLET_PRIVATE_KEY
- **Purpose**: Primary wallet private key (fallback)
- **Source**: Primary wallet generation
- **Format**: Chain-specific format
- **Required**: For live trading mode
- **Used by**: Wallet initialization fallback
- **Security**: CRITICAL - grants full wallet control

## AI & ML API Secrets

These secrets enable AI-powered analysis and decision making.

### XAI_API_KEY
- **Purpose**: xAI/Grok API for advanced analysis
- **Source**: [xAI Platform](https://x.ai)
- **Format**: API key string
- **Required**: For XAI features
- **Used by**: `src/ai/xai_client.py`, explainable AI features
- **Rate Limits**: Varies by plan
- **Security**: Medium - API usage charges apply

### OPENAI_API_KEY
- **Purpose**: OpenAI GPT API for natural language processing
- **Source**: [OpenAI Platform](https://platform.openai.com)
- **Format**: `sk-...` prefixed string
- **Required**: For AI analysis features
- **Used by**: `src/ai/openai_client.py`
- **Rate Limits**: Token-based billing
- **Security**: Medium - API usage charges apply

### AGENT_API_KEY
- **Purpose**: Custom AI agent API authentication
- **Source**: Internal agent system
- **Format**: Custom format
- **Required**: For agent integration
- **Used by**: `src/agents/agent_client.py`
- **Security**: Medium - controls agent access

## Market Data & Analytics Secrets

These secrets provide market data and analytics capabilities.

### LUNARCRUSH_API_KEY
- **Purpose**: Social sentiment and market analytics
- **Source**: [LunarCrush](https://lunarcrush.com/developers)
- **Format**: API key string
- **Required**: For social sentiment analysis
- **Used by**: `src/ml_analysis/market_data.py`
- **Rate Limits**: 100-1000 requests/hour depending on plan
- **Security**: Low - public market data

### COINGECKO_API_KEY
- **Purpose**: Free CoinGecko API access
- **Source**: [CoinGecko](https://www.coingecko.com/en/api)
- **Format**: API key string
- **Required**: For market data
- **Used by**: `src/ml_analysis/lstm_model.py`, historical data
- **Rate Limits**: 10-50 calls/minute for free tier
- **Security**: Low - public market data

### COINGECKO_PRO_API_KEY
- **Purpose**: Premium CoinGecko API access
- **Source**: [CoinGecko Pro](https://www.coingecko.com/en/api/pricing)
- **Format**: API key string
- **Required**: For enhanced market data
- **Used by**: Premium market data features
- **Rate Limits**: Higher limits than free tier
- **Security**: Low - public market data

### GLASSNODE_API_KEY
- **Purpose**: On-chain analytics and metrics
- **Source**: [Glassnode](https://glassnode.com)
- **Format**: API key string
- **Required**: For on-chain analysis
- **Used by**: `src/ml_analysis/onchain_analytics.py`
- **Rate Limits**: Varies by subscription
- **Security**: Low - public blockchain analytics

### MESSARI_API_KEY
- **Purpose**: Crypto research and fundamental data
- **Source**: [Messari](https://messari.io/api)
- **Format**: API key string
- **Required**: For fundamental analysis
- **Used by**: Research and analysis features
- **Rate Limits**: Varies by plan
- **Security**: Low - public research data

### JUPITER_API_KEY
- **Purpose**: Jupiter DEX aggregator on Solana
- **Source**: [Jupiter](https://docs.jup.ag)
- **Format**: API key string
- **Required**: For Solana DEX operations
- **Used by**: `src/trading/jupiter_client.py`
- **Rate Limits**: High throughput for DEX operations
- **Security**: Medium - can execute trades if wallet keys available

## Social Media & External APIs

These secrets enable social media monitoring and external data access.

### X_BEARER_TOKEN
- **Purpose**: X (Twitter) API v2 bearer token authentication
- **Source**: [X Developer Portal](https://developer.twitter.com)
- **Format**: Bearer token string
- **Required**: For X API access
- **Used by**: `src/social/x_client.py`, sentiment analysis
- **Rate Limits**: Varies by endpoint and plan
- **Security**: Medium - social media access

### X_API_KEY
- **Purpose**: X (Twitter) API key
- **Source**: X Developer Portal
- **Format**: API key string
- **Required**: For X API operations
- **Used by**: X API authentication
- **Security**: Medium - social media access

### X_API_SECRET
- **Purpose**: X (Twitter) API secret
- **Source**: X Developer Portal
- **Format**: API secret string
- **Required**: For X API operations
- **Used by**: X API authentication
- **Security**: High - paired with API key for full access

## Environment Variable Mapping

The following table shows how secrets map to environment variables in the application:

| Secret Manager Name | Environment Variable | Module Usage |
|-------------------|---------------------|-------------|
| TELEGRAM_TOKEN | TELEGRAM_TOKEN | Telegram bot authentication |
| WEBHOOK_SECRET | WEBHOOK_SECRET | Webhook validation |
| DB_PASSWORD | DB_PASSWORD | Database connections |
| DATABASE_URL | DATABASE_URL | Database URL construction |
| HELIUS_API_KEY | HELIUS_API_KEY | Solana blockchain access |
| BIRDEYE_API_KEY | BIRDEYE_API_KEY | DeFi analytics |
| ETHERSCAN_API_KEY | ETHERSCAN_API_KEY | Ethereum blockchain |
| LUNARCRUSH_API_KEY | LUNARCRUSH_API_KEY | Social sentiment |
| COINGECKO_API_KEY | COINGECKO_API_KEY | Market data |
| XAI_API_KEY | XAI_API_KEY | AI analysis |
| OPENAI_API_KEY | OPENAI_API_KEY | GPT analysis |
| X_BEARER_TOKEN | X_BEARER_TOKEN | X/Twitter access |
| SOLANA_PRIVATE_KEY | SOLANA_PRIVATE_KEY | Solana trading |
| ETHEREUM_PRIVATE_KEY | ETHEREUM_PRIVATE_KEY | Ethereum trading |
| HYPERLIQUID_API_KEY | HYPERLIQUID_API_KEY | Exchange trading |

## Security Best Practices

### Secret Categories by Risk Level

**CRITICAL (Trading Keys)**
- Store in separate Secret Manager project if possible
- Implement additional access controls
- Enable audit logging
- Use hardware security modules in production
- Regular rotation (monthly)

**HIGH (Database, Infrastructure)**
- Restrict access to essential services only
- Enable audit logging
- Regular rotation (quarterly)
- Use strong, unique passwords

**MEDIUM (API Keys with Write Access)**
- Monitor usage patterns
- Set up billing alerts
- Regular rotation (semi-annually)
- Use least-privilege API permissions

**LOW (Public Data APIs)**
- Monitor for rate limit violations
- Basic access controls
- Rotation as needed for security

### Access Control

1. **Service Accounts**: Use dedicated service accounts for each environment
2. **IAM Policies**: Apply least-privilege access principles
3. **Audit Logs**: Enable and monitor Secret Manager audit logs
4. **Network Security**: Restrict secret access to authorized networks

### Monitoring and Alerting

1. **Usage Monitoring**: Track secret access patterns
2. **Billing Alerts**: Monitor API usage costs
3. **Rate Limit Alerts**: Alert on API rate limit violations
4. **Security Alerts**: Monitor for unauthorized access attempts

### Development vs Production

**Development Environment**:
- Use separate project/secrets for development
- Mock external APIs where possible
- Never use production trading keys
- Use dummy/test API keys

**Production Environment**:
- Separate secret management from development
- Enable all monitoring and alerting
- Use production-grade security measures
- Regular security audits

## Troubleshooting

### Common Issues

1. **Secret Not Found**: Verify secret exists in correct project
2. **Access Denied**: Check IAM permissions for service account
3. **Rate Limits**: Monitor API usage and implement backoff
4. **Invalid Keys**: Verify key format and expiration

### Validation Commands

```bash
# Validate all secrets
python deploy/validate_secrets.py

# Test specific secret access
gcloud secrets versions access latest --secret="TELEGRAM_TOKEN"

# List all secrets
gcloud secrets list
```

### Emergency Procedures

1. **Compromised Key**: Immediately rotate affected secrets
2. **Service Outage**: Check secret access and IAM permissions
3. **Billing Issues**: Review API usage patterns and rate limits
4. **Security Breach**: Audit access logs and rotate all critical secrets

## References

- [Google Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Cloud Run Secret Integration](https://cloud.google.com/run/docs/configuring/secrets)
- [IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
- [Security Key Management](https://cloud.google.com/security-key-management)