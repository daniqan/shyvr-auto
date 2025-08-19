"""
Paginated Data Collector for handling large-scale historical data collection

This module provides the main interface for collecting data with intelligent
pagination, retry logic, and gap filling capabilities.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import structlog

from src.data_pipeline.pagination_manager import (
    ChunkManager, CollectionProgress, CollectionMonitor, DataValidator, ChunkConfig
)
from src.ml_analysis.market_data import CoinGeckoClient

logger = structlog.get_logger(__name__)


class PaginatedDataCollector:
    """
    Dynamic pagination handler for CoinGecko Pro API
    
    Key features:
    - Automatically determines optimal chunk size based on timeframe
    - Handles before_timestamp pagination seamlessly  
    - Retries failed chunks
    - Validates data completeness
    - Merges overlapping data
    - Supports resume from checkpoint
    """
    
    def __init__(self, 
                 coingecko_client: CoinGeckoClient,
                 pagination_config: Dict[str, Any] = None,
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize paginated collector
        
        Args:
            coingecko_client: Initialized CoinGecko client
            pagination_config: Pagination configuration
            checkpoint_dir: Directory for checkpoints
        """
        self.client = coingecko_client
        self.pagination_config = pagination_config or self._get_default_config()
        self.progress_tracker = CollectionProgress(checkpoint_dir)
        self.monitor = CollectionMonitor()
        self.validator = DataValidator(self.pagination_config.get('validation', {}))
        
        logger.info("PaginatedDataCollector initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pagination configuration"""
        return {
            'api_limits': {
                'daily': {
                    'max_candles': 180,
                    'optimal_chunk_days': 150
                },
                'hourly': {
                    'max_candles': 1000,
                    'optimal_chunk_hours': 900
                },
                'four_hour': {
                    'max_candles': 1000,
                    'optimal_chunk_periods': 900
                }
            },
            'strategy': {
                'direction': 'backward',
                'overlap_periods': 1,
                'max_retries': 3,
                'retry_delay': 2
            },
            'validation': {
                'min_completeness': 0.85,
                'gap_tolerance_hours': 48,
                'deduplication': True
            }
        }
    
    async def collect_with_pagination(self,
                                     contract_address: str,
                                     network: str,
                                     token_symbol: str,
                                     timeframe: str,
                                     days: int,
                                     resume: bool = True) -> pd.DataFrame:
        """
        Collect data with intelligent pagination
        
        Args:
            contract_address: Token contract address
            network: Network (eth, bsc, etc.)
            token_symbol: Token symbol for logging
            timeframe: 'daily', 'hourly', or 'four_hour'
            days: Number of days to collect
            resume: Whether to resume from checkpoint
            
        Returns:
            DataFrame with collected OHLCV data
        """
        logger.info(
            f"Starting paginated collection for {token_symbol}",
            timeframe=timeframe,
            days=days,
            network=network
        )
        
        # Initialize chunk manager
        chunk_manager = ChunkManager(timeframe, days, self.pagination_config)
        
        # Check for resume point
        if resume and self.progress_tracker.can_resume(token_symbol, timeframe):
            logger.info(f"Resuming collection for {token_symbol}/{timeframe}")
            resume_point = self.progress_tracker.get_resume_point(token_symbol, timeframe)
            chunk_manager.load_state(resume_point.get('chunk_state', {}))
            
            # Load existing data
            if 'data_file' in resume_point:
                existing_data = pd.read_parquet(resume_point['data_file'])
                chunk_manager.collected_data.append(existing_data)
                logger.info(f"Loaded {len(existing_data)} existing records from checkpoint")
        
        # Collection loop
        retry_queue = []
        
        while chunk_manager.has_more_chunks() or retry_queue:
            # Get next chunk
            if retry_queue:
                chunk = retry_queue.pop(0)
                logger.info(f"Retrying chunk {chunk.chunk_id}")
            else:
                chunk = chunk_manager.get_next_chunk()
                if not chunk:
                    break
            
            try:
                # Collect chunk data
                chunk_data = await self._fetch_chunk(
                    contract_address=contract_address,
                    network=network,
                    chunk=chunk,
                    timeframe=timeframe
                )
                
                # Validate chunk data
                if chunk_data is not None and len(chunk_data) > 0:
                    # Update before_timestamp for next chunk
                    if len(chunk_data) > 0 and 'timestamp' in chunk_data.columns:
                        oldest_timestamp = chunk_data['timestamp'].min()
                        if isinstance(oldest_timestamp, pd.Timestamp):
                            oldest_timestamp = int(oldest_timestamp.timestamp())
                        chunk_manager.update_before_timestamp(chunk, oldest_timestamp)
                    
                    # Record success
                    chunk_manager.record_chunk_result(chunk, chunk_data, success=True)
                    
                    # Update progress
                    progress = chunk_manager.get_progress_summary()
                    self.monitor.log_progress(
                        token=token_symbol,
                        timeframe=timeframe,
                        chunks_done=progress['chunks_completed'],
                        chunks_total=progress['chunks_total'],
                        candles_collected=progress['candles_collected'],
                        candles_expected=progress['candles_expected']
                    )
                    
                    # Save checkpoint
                    self.progress_tracker.save_checkpoint(
                        token_symbol, 
                        timeframe, 
                        chunk_manager,
                        chunk_manager.merge_data()
                    )
                else:
                    # Empty or invalid data
                    raise ValueError(f"No data returned for chunk {chunk.chunk_id}")
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk {chunk.chunk_id}: {e}")
                chunk_manager.record_chunk_result(chunk, None, success=False)
                
                # Add to retry queue if retries remaining
                if chunk.retry_count < self.pagination_config['strategy']['max_retries']:
                    retry_queue.append(chunk)
                    await asyncio.sleep(self.pagination_config['strategy']['retry_delay'])
            
            # Move to next chunk
            chunk_manager.advance_to_next_chunk()
        
        # Merge all collected data
        final_data = chunk_manager.merge_data()
        
        # Fill gaps if needed
        gaps = chunk_manager.get_gaps()
        if gaps and self.pagination_config.get('strategy', {}).get('fill_gaps', True):
            logger.info(f"Found {len(gaps)} gaps, attempting to fill")
            gap_data = await self._fill_gaps(
                contract_address=contract_address,
                network=network,
                token_symbol=token_symbol,
                gaps=gaps,
                timeframe=timeframe
            )
            if not gap_data.empty:
                final_data = pd.concat([final_data, gap_data], ignore_index=True)
                final_data = final_data.drop_duplicates('timestamp').sort_values('timestamp')
        
        # Validate final data
        validation_report = self.validator.generate_validation_report(
            final_data, timeframe, days
        )
        
        if validation_report['valid']:
            logger.info(f"Data validation passed for {token_symbol}/{timeframe}")
        else:
            logger.warning(
                f"Data validation issues for {token_symbol}/{timeframe}",
                report=validation_report
            )
        
        # Clear checkpoint on successful completion
        if validation_report['validations']['candle_count']['valid']:
            self.progress_tracker.clear_checkpoint(token_symbol, timeframe)
        
        # Generate final summary
        self.monitor.generate_summary()
        
        return final_data
    
    async def _fetch_chunk(self,
                          contract_address: str,
                          network: str,
                          chunk: ChunkConfig,
                          timeframe: str) -> pd.DataFrame:
        """
        Fetch a single chunk of data
        
        Args:
            contract_address: Token contract address
            network: Network identifier
            chunk: Chunk configuration
            timeframe: Timeframe for data
            
        Returns:
            DataFrame with chunk data
        """
        try:
            # Map timeframe to API format
            if timeframe in ['day', 'daily']:
                api_timeframe = 'day'
                aggregate = 1
            elif timeframe in ['hour', 'hourly']:
                api_timeframe = 'hour'
                aggregate = 1
            elif timeframe == 'four_hour':
                api_timeframe = 'hour'
                aggregate = 4
            else:
                api_timeframe = 'hour'
                aggregate = 1
            
            # Build API URL
            url = f"{self.client.base_url}/onchain/networks/{network}/tokens/{contract_address}/ohlcv/{api_timeframe}"
            
            # Build parameters
            params = {
                'aggregate': aggregate,
                'include_empty_intervals': 'false',
                'limit': 1000  # Max candles per request
            }
            
            # Add before_timestamp for pagination
            if chunk.before_timestamp:
                params['before_timestamp'] = chunk.before_timestamp
                logger.debug(f"Using before_timestamp: {chunk.before_timestamp}")
            
            # Make API request
            logger.debug(f"Fetching chunk {chunk.chunk_id}: {chunk.chunk_size_days} days")
            response = await self.client._make_request(url, params=params, headers=self.client.headers)
            
            # Parse response
            ohlcv_list = self._parse_response(response)
            
            if not ohlcv_list:
                logger.warning(f"No data in chunk {chunk.chunk_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(ohlcv_list)
            
            logger.info(
                f"Chunk {chunk.chunk_id} fetched",
                records=len(df),
                expected=chunk.expected_candles,
                date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}" if len(df) > 0 else "N/A"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk.chunk_id}: {e}")
            raise
    
    def _parse_response(self, response: Any) -> List:
        """Parse API response to extract OHLCV data"""
        if isinstance(response, dict) and 'data' in response:
            # Handle nested response structure
            if 'attributes' in response['data']:
                return response['data']['attributes'].get('ohlcv_list', [])
            elif 'ohlcv_list' in response['data']:
                return response['data']['ohlcv_list']
            else:
                return []
        elif isinstance(response, list):
            # Direct list response
            return response
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            return []
    
    def _convert_to_dataframe(self, ohlcv_list: List) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame"""
        if not ohlcv_list:
            return pd.DataFrame()
        
        data = []
        for entry in ohlcv_list:
            if len(entry) >= 6:
                # Full OHLCV with volume
                data.append({
                    'timestamp': pd.to_datetime(entry[0], unit='s'),
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': float(entry[5])
                })
            elif len(entry) >= 5:
                # OHLC without volume
                data.append({
                    'timestamp': pd.to_datetime(entry[0], unit='s'),
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': 0.0
                })
        
        df = pd.DataFrame(data)
        
        # Sort by timestamp and remove duplicates
        if not df.empty:
            df = df.sort_values('timestamp')
            df = df.drop_duplicates('timestamp')
            df = df.reset_index(drop=True)
        
        return df
    
    async def _fill_gaps(self,
                        contract_address: str,
                        network: str,
                        token_symbol: str,
                        gaps: List,
                        timeframe: str) -> pd.DataFrame:
        """
        Attempt to fill gaps in collected data
        
        Args:
            contract_address: Token contract address
            network: Network identifier
            token_symbol: Token symbol
            gaps: List of TimeRange gaps
            timeframe: Timeframe
            
        Returns:
            DataFrame with gap data
        """
        gap_data = []
        
        for gap in gaps[:5]:  # Limit to 5 gaps to avoid excessive API calls
            try:
                logger.info(
                    f"Attempting to fill gap for {token_symbol}",
                    start=gap.start,
                    end=gap.end,
                    expected_candles=gap.expected_candles
                )
                
                # Create a custom chunk for the gap
                gap_chunk = ChunkConfig(
                    chunk_id=999,  # Special ID for gap chunks
                    start_timestamp=int(gap.start.timestamp()),
                    end_timestamp=int(gap.end.timestamp()),
                    before_timestamp=int(gap.end.timestamp()),
                    expected_candles=gap.expected_candles,
                    chunk_size_days=(gap.end - gap.start).days
                )
                
                # Fetch gap data
                df = await self._fetch_chunk(
                    contract_address=contract_address,
                    network=network,
                    chunk=gap_chunk,
                    timeframe=timeframe
                )
                
                if not df.empty:
                    gap_data.append(df)
                    logger.info(f"Filled gap with {len(df)} candles")
                
            except Exception as e:
                logger.error(f"Failed to fill gap: {e}")
                continue
        
        if gap_data:
            return pd.concat(gap_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    async def collect_multiple_tokens(self,
                                     tokens: List[Dict[str, Any]],
                                     timeframes: List[Dict[str, Any]],
                                     parallel: bool = True,
                                     max_concurrent: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple tokens and timeframes
        
        Args:
            tokens: List of token configurations
            timeframes: List of timeframe configurations
            parallel: Whether to collect tokens in parallel
            max_concurrent: Max concurrent collections
            
        Returns:
            Dictionary mapping token_timeframe to DataFrames
        """
        results = {}
        
        if parallel:
            # Parallel collection with semaphore
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            
            for token in tokens:
                for timeframe in timeframes:
                    if not timeframe.get('enabled', True):
                        continue
                    
                    task = self._collect_with_semaphore(
                        semaphore=semaphore,
                        token=token,
                        timeframe=timeframe
                    )
                    tasks.append((f"{token['symbol']}_{timeframe['name']}", task))
            
            # Execute all tasks
            for key, task in tasks:
                try:
                    results[key] = await task
                except Exception as e:
                    logger.error(f"Failed to collect {key}: {e}")
                    results[key] = pd.DataFrame()
        else:
            # Sequential collection
            for token in tokens:
                for timeframe in timeframes:
                    if not timeframe.get('enabled', True):
                        continue
                    
                    key = f"{token['symbol']}_{timeframe['name']}"
                    try:
                        results[key] = await self.collect_with_pagination(
                            contract_address=token['contract_address'],
                            network=token['network'],
                            token_symbol=token['symbol'],
                            timeframe=timeframe['name'],
                            days=timeframe['days_back']
                        )
                    except Exception as e:
                        logger.error(f"Failed to collect {key}: {e}")
                        results[key] = pd.DataFrame()
        
        return results
    
    async def _collect_with_semaphore(self,
                                     semaphore: asyncio.Semaphore,
                                     token: Dict[str, Any],
                                     timeframe: Dict[str, Any]) -> pd.DataFrame:
        """Collect data with semaphore for rate limiting"""
        async with semaphore:
            return await self.collect_with_pagination(
                contract_address=token['contract_address'],
                network=token['network'],
                token_symbol=token['symbol'],
                timeframe=timeframe['name'],
                days=timeframe['days_back']
            )