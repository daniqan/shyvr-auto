# API Research: Moralis vs CoinGecko for Historical OHLCV Data

## Research Summary

This document contains the comprehensive research and analysis conducted to select the appropriate API for historical OHLCV (Open, High, Low, Close, Volume) cryptocurrency data for the LSTM model's `_prepare_training_data` method in the Shyvr AI trading bot.

## Comparison Matrix

| Criteria | Moralis API | CoinGecko API | Winner |
|----------|-------------|---------------|---------|
| **Historical Data Coverage** | Limited to DEX data | 10+ years (since 2014) | CoinGecko |
| **Rate Limits (Free)** | Limited free tier | 30 calls/min, 10K/month | CoinGecko |
| **Rate Limits (Paid)** | Millions/month (custom) | 500 calls/min, 500K/month | Moralis |
| **Pricing (Entry Level)** | Enterprise pricing | $129/month | CoinGecko |
| **Cross-chain Support** | 100+ chains (Web3 focus) | 243 networks (market focus) | Tie |
| **Integration Complexity** | Complex but powerful | Simple and straightforward | CoinGecko |
| **Data Quality** | Real-time, on-chain focus | Aggregated market data | Tie |
| **Existing Integration** | None | Already implemented | CoinGecko |
| **OHLCV Endpoints** | Advanced DEX OHLCV | Simple `/coins/{id}/ohlc` | CoinGecko |

## Detailed Analysis

### 1. API Capabilities for Historical Price Data

**Moralis API:**
- Comprehensive OHLCV data across 100+ endpoints
- Cross-chain support (Ethereum, Solana, BSC, Polygon, etc.)
- Real-time updates and custom timeframes
- Advanced token insights and DeFi positions
- Primarily DEX-focused data

**CoinGecko API:**
- Over 10 years of historical data (since 2014)
- Coverage of 10M+ tokens across 243 networks
- Aggregated data from both CEX and DEX sources
- Simple OHLC endpoint with automatic granularity
- Market-focused comprehensive data

### 2. Rate Limits and Pricing Models

**Moralis API:**
- Enterprise-grade with customizable plans
- Millions of monthly requests possible
- SOC 2 Type 2 & ISO 27001 certified
- More expensive but comprehensive

**CoinGecko API:**
- Free: 30 calls/min, 10K calls/month
- Paid: $129/month for 500 calls/min, 500K calls/month
- Cost-effective for historical price data needs

### 3. Integration Readiness

**Existing CoinGecko Integration:**
- `CoinGeckoClient` already implemented in `src/ml_analysis/market_data.py`
- Environment variables configured (`COINGECKO_API_KEY`, `COINGECKO_PRO_API_KEY`)
- Caching and error handling infrastructure in place

**Moralis Integration:**
- Would require new client implementation
- Additional infrastructure setup needed
- No existing integration in codebase

## Decision: CoinGecko API Selected

### Primary Reasons:

1. **Perfect Fit for LSTM Requirements**: 10+ years of historical data with automatic granularity handling
2. **Cost Effectiveness**: $129/month vs enterprise Moralis pricing for basic historical data needs
3. **Existing Integration**: CoinGecko client already implemented in codebase
4. **Data Quality**: Aggregated market data provides comprehensive representation for ML training
5. **Simplicity**: Straightforward integration with clear documentation

### Implementation Strategy:

1. **Extend Existing Client**: Add OHLCV endpoints to `CoinGeckoClient` class
2. **Historical Data Method**: Implement `/coins/{id}/market_chart` for price history
3. **OHLCV Integration**: Add `/coins/{id}/ohlc` for candlestick data
4. **Leverage Infrastructure**: Use existing caching and error handling
5. **Gradual Enhancement**: Start with basic implementation, expand as needed

## Next Steps

1. Implement CoinGecko OHLCV endpoints in existing client
2. Update `_prepare_training_data` method in LSTM model
3. Replace placeholder implementation with real data fetching
4. Add comprehensive tests (80%+ coverage target)
5. Ensure required API keys are documented in environment setup
6. Commit implementation with proper testing

## Technical Implementation Notes

### Required Endpoints:
- `/coins/{id}/market_chart` - Historical price data
- `/coins/{id}/ohlc` - OHLC candlestick data
- `/global` - Market dominance and correlation data (already implemented)

### Data Processing:
- Convert CoinGecko timestamps to proper datetime objects
- Handle missing data points gracefully
- Implement proper data validation and sanitization
- Cache historical data to minimize API calls

### Error Handling:
- Rate limit management (existing infrastructure)
- Network failure retry logic (existing infrastructure)
- Data validation and fallback mechanisms
- API key authentication handling (existing infrastructure)

---

*Research conducted on 2025-07-28 as part of Phase 1 GCP Production Deployment preparation.*