"""
Enhanced Initial Corpus Collector with improved granularity
Collects hourly data by chunking long time ranges
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import structlog

from src.ml_analysis.market_data import CoinGeckoClient
from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector


class EnhancedCorpusCollector(InitialCorpusCollector):
    """Enhanced corpus collector that gets better data granularity"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = structlog.get_logger().bind(component=self.__class__.__name__)
    
    async def _collect_token_ohlcv_with_retries(self, token: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Override to use chunked approach for better granularity
        
        CoinGecko granularity by range:
        - 1-2 days: 5-minute to hourly data
        - 3-30 days: hourly data  
        - 31-90 days: 4-hour data
        - 91+ days: 4-day data (96 hours)
        
        Strategy: For ranges > 30 days, split into 30-day chunks to get hourly data
        """
        total_days = (end_date - start_date).days
        
        if total_days <= 30:
            # For 30 days or less, use the original method
            return await super()._collect_token_ohlcv_with_retries(token, start_date, end_date)
        
        # For longer periods, use chunked approach
        self.logger.info(f"Using chunked approach for {token}: {total_days} days split into 30-day chunks")
        
        all_data = []
        chunk_size = 30  # days
        current_end = end_date
        
        # Work backwards from end_date
        while current_end > start_date:
            current_start = max(current_end - timedelta(days=chunk_size), start_date)
            
            self.logger.info(f"Collecting chunk for {token}: {current_start.date()} to {current_end.date()}")
            
            try:
                # Collect chunk data
                chunk_data = await self._collect_single_chunk(token, current_start, current_end)
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    self.logger.info(f"Chunk collected: {len(chunk_data)} records")
                
                # Rate limiting between chunks
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect chunk for {token}: {e}")
            
            # Move to next chunk
            current_end = current_start
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_values('timestamp')
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
        combined_df = combined_df.reset_index(drop=True)
        
        self.logger.info(f"Total records collected for {token}: {len(combined_df)}")
        
        # If we still don't have enough granularity, try to interpolate
        if len(combined_df) < total_days * 6:  # Less than 6 records per day (4-hour granularity)
            combined_df = await self._enhance_with_market_chart(token, start_date, end_date, combined_df)
        
        return combined_df
    
    async def _collect_single_chunk(self, token: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect a single chunk of data with retries"""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                days = (end_date - start_date).days
                
                data = await self.coingecko_client.get_ohlcv_data(
                    coin_id=token,
                    days=days,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                self.logger.warning(f"Chunk collection attempt {attempt + 1} failed: {e}")
                last_error = e
                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
        
        if last_error:
            raise last_error
        return pd.DataFrame()
    
    async def _enhance_with_market_chart(self, token: str, start_date: datetime, end_date: datetime, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance OHLC data using market_chart endpoint for better granularity
        This constructs synthetic OHLC from price data when we have sparse candles
        """
        self.logger.info(f"Enhancing {token} data with market_chart endpoint")
        
        try:
            # Get price data from market_chart endpoint
            params = {
                "vs_currency": "usd",
                "from": int(start_date.timestamp()),
                "to": int(end_date.timestamp())
            }
            
            url = f"{self.coingecko_client.base_url}/coins/{token}/market_chart/range"
            data = await self.coingecko_client._make_request(url, params=params, headers=self.coingecko_client.headers)
            
            if not data or 'prices' not in data:
                return ohlc_df
            
            # Convert price data to DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            price_df = pd.DataFrame(prices, columns=['timestamp_ms', 'price'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp_ms'], unit='ms')
            
            # Create hourly OHLC from price data
            price_df.set_index('timestamp', inplace=True)
            
            # Resample to hourly
            hourly_ohlc = price_df['price'].resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Add volume if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp_ms', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp_ms'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                hourly_volume = volume_df['volume'].resample('1H').sum()
                hourly_ohlc['volume'] = hourly_volume
            else:
                hourly_ohlc['volume'] = 0
            
            # Reset index to get timestamp as column
            hourly_ohlc = hourly_ohlc.reset_index()
            
            # If we have existing OHLC data, prefer it over synthetic
            if not ohlc_df.empty:
                # Use existing OHLC timestamps as reference
                existing_times = set(ohlc_df['timestamp'])
                
                # Add synthetic data only for missing timestamps
                synthetic_data = hourly_ohlc[~hourly_ohlc['timestamp'].isin(existing_times)]
                
                if not synthetic_data.empty:
                    self.logger.info(f"Adding {len(synthetic_data)} synthetic hourly records to {token}")
                    combined = pd.concat([ohlc_df, synthetic_data], ignore_index=True)
                    combined = combined.sort_values('timestamp').reset_index(drop=True)
                    return combined
                
                return ohlc_df
            else:
                # No existing OHLC, use all synthetic data
                self.logger.info(f"Using {len(hourly_ohlc)} synthetic hourly records for {token}")
                return hourly_ohlc
                
        except Exception as e:
            self.logger.warning(f"Failed to enhance with market_chart data: {e}")
            return ohlc_df
    
    async def collect_standardized_corpus(self, tokens: List[str], start_date: datetime, end_date: datetime) -> Dict:
        """
        Override to provide better progress reporting
        """
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED CORPUS COLLECTION")
        self.logger.info("=" * 60)
        self.logger.info(f"Tokens: {', '.join(tokens)}")
        self.logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Total days: {(end_date - start_date).days}")
        self.logger.info("Strategy: Using 30-day chunks for optimal hourly granularity")
        self.logger.info("=" * 60)
        
        # Call parent implementation
        result = await super().collect_standardized_corpus(tokens, start_date, end_date)
        
        if result['success']:
            self.logger.info("=" * 60)
            self.logger.info("COLLECTION SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"✅ Version: {result.get('version_name')}")
            self.logger.info(f"✅ Total records: {result.get('total_records'):,}")
            
            # Calculate actual vs expected
            expected_hourly = len(tokens) * (end_date - start_date).days * 24
            actual_ohlcv = result.get('total_records', 0)
            coverage = (actual_ohlcv / expected_hourly * 100) if expected_hourly > 0 else 0
            
            self.logger.info(f"📊 Data coverage:")
            self.logger.info(f"   Expected hourly records: {expected_hourly:,}")
            self.logger.info(f"   Actual OHLCV records: {actual_ohlcv:,}")
            self.logger.info(f"   Coverage: {coverage:.1f}%")
            
            if coverage < 25:
                self.logger.warning("⚠️  Low coverage! Consider:")
                self.logger.warning("   1. Using shorter time ranges")
                self.logger.warning("   2. Checking API rate limits")
                self.logger.warning("   3. Using continuous collection over time")
            elif coverage < 50:
                self.logger.info("📈 Moderate coverage - suitable for initial training")
            else:
                self.logger.info("✨ Excellent coverage for training!")
            
            self.logger.info("=" * 60)
        
        return result