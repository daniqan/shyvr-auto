"""
Dynamic Pagination Manager for Multi-Granularity Data Collection

This module provides intelligent pagination handling for collecting large-scale
historical data from APIs with request limits (e.g., CoinGecko Pro).
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for a single data chunk"""
    chunk_id: int
    start_timestamp: int
    end_timestamp: int
    before_timestamp: Optional[int]
    expected_candles: int
    chunk_size_days: float
    retry_count: int = 0
    completed: bool = False
    candles_collected: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TimeRange:
    """Represents a time range for gap filling"""
    start: datetime
    end: datetime
    expected_candles: int
    

class ChunkManager:
    """Manages chunk state and progress for paginated data collection"""
    
    def __init__(self, 
                 timeframe: str,
                 total_days: int,
                 pagination_config: Dict[str, Any]):
        """
        Initialize chunk manager
        
        Args:
            timeframe: 'day', 'hour', or 'four_hour'
            total_days: Total days to collect
            pagination_config: Pagination configuration from YAML
        """
        self.timeframe = timeframe
        self.total_days = total_days
        self.pagination_config = pagination_config
        
        # Initialize state
        self.chunks = self._calculate_chunks()
        self.completed_chunks = []
        self.failed_chunks = []
        self.collected_data = []
        self.current_chunk_index = 0
        
        logger.info(
            "ChunkManager initialized",
            timeframe=timeframe,
            total_days=total_days,
            total_chunks=len(self.chunks)
        )
    
    def _calculate_chunks(self) -> List[ChunkConfig]:
        """Calculate optimal chunks based on timeframe and API limits"""
        chunks = []
        
        # Get API limits for this timeframe
        api_limits = self.pagination_config.get('api_limits', {})
        timeframe_limits = api_limits.get(self.timeframe, {})
        
        # Determine chunk size
        if self.timeframe == 'day' or self.timeframe == 'daily':
            max_candles = timeframe_limits.get('max_candles', 180)
            optimal_chunk_days = timeframe_limits.get('optimal_chunk_days', 150)
            candles_per_day = 1
        elif self.timeframe == 'hour' or self.timeframe == 'hourly':
            max_candles = timeframe_limits.get('max_candles', 1000)
            optimal_chunk_days = timeframe_limits.get('optimal_chunk_hours', 900) / 24  # Convert hours to days
            candles_per_day = 24
        elif self.timeframe == 'four_hour':
            max_candles = timeframe_limits.get('max_candles', 1000)
            optimal_chunk_days = timeframe_limits.get('optimal_chunk_periods', 900) / 6  # 6 four-hour periods per day
            candles_per_day = 6
        else:
            # Default fallback
            max_candles = 1000
            optimal_chunk_days = 30
            candles_per_day = 24
        
        # Calculate chunks from newest to oldest (backward pagination)
        end_time = datetime.now(timezone.utc)
        remaining_days = self.total_days
        chunk_id = 0
        
        while remaining_days > 0:
            # Calculate this chunk's size
            chunk_days = min(remaining_days, optimal_chunk_days)
            
            # Calculate timestamps
            start_time = end_time - timedelta(days=chunk_days)
            
            # Create chunk config
            chunk = ChunkConfig(
                chunk_id=chunk_id,
                start_timestamp=int(start_time.timestamp()),
                end_timestamp=int(end_time.timestamp()),
                before_timestamp=int(end_time.timestamp()) if chunk_id > 0 else None,
                expected_candles=int(chunk_days * candles_per_day),
                chunk_size_days=chunk_days
            )
            
            chunks.append(chunk)
            
            # Move to next chunk
            end_time = start_time
            remaining_days -= chunk_days
            chunk_id += 1
        
        logger.info(
            f"Calculated {len(chunks)} chunks for {self.total_days} days",
            timeframe=self.timeframe,
            chunk_sizes=[c.chunk_size_days for c in chunks]
        )
        
        return chunks
    
    def get_next_chunk(self) -> Optional[ChunkConfig]:
        """Returns next chunk to fetch"""
        if self.current_chunk_index >= len(self.chunks):
            return None
        
        chunk = self.chunks[self.current_chunk_index]
        
        # Skip completed chunks
        while chunk.completed and self.current_chunk_index < len(self.chunks) - 1:
            self.current_chunk_index += 1
            chunk = self.chunks[self.current_chunk_index]
        
        if chunk.completed:
            return None
        
        return chunk
    
    def record_chunk_result(self, chunk: ChunkConfig, data: Optional[pd.DataFrame], success: bool):
        """Records chunk completion"""
        if success and data is not None:
            chunk.completed = True
            chunk.candles_collected = len(data)
            self.completed_chunks.append(chunk)
            self.collected_data.append(data)
            
            logger.info(
                f"Chunk {chunk.chunk_id} completed",
                candles_collected=chunk.candles_collected,
                expected=chunk.expected_candles,
                completeness=f"{(chunk.candles_collected/chunk.expected_candles)*100:.1f}%"
            )
        else:
            chunk.retry_count += 1
            if chunk.retry_count >= self.pagination_config.get('strategy', {}).get('max_retries', 3):
                self.failed_chunks.append(chunk)
                logger.warning(f"Chunk {chunk.chunk_id} failed after {chunk.retry_count} retries")
            else:
                logger.info(f"Chunk {chunk.chunk_id} failed, will retry (attempt {chunk.retry_count})")
    
    def has_more_chunks(self) -> bool:
        """Check if there are more chunks to process"""
        return self.current_chunk_index < len(self.chunks) and not all(c.completed for c in self.chunks)
    
    def advance_to_next_chunk(self):
        """Move to next chunk"""
        self.current_chunk_index += 1
    
    def get_gaps(self) -> List[TimeRange]:
        """Identifies gaps in collected data"""
        if not self.collected_data:
            return []
        
        # Merge all data and sort by timestamp
        all_data = pd.concat(self.collected_data, ignore_index=True)
        all_data = all_data.sort_values('timestamp').drop_duplicates('timestamp')
        
        gaps = []
        gap_tolerance = self.pagination_config.get('validation', {}).get('gap_tolerance_hours', 48)
        
        # Check for gaps
        if 'timestamp' in all_data.columns:
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
            time_diffs = all_data['timestamp'].diff()
            
            # Expected time difference based on timeframe
            if self.timeframe in ['day', 'daily']:
                expected_diff = pd.Timedelta(days=1)
            elif self.timeframe in ['hour', 'hourly']:
                expected_diff = pd.Timedelta(hours=1)
            elif self.timeframe == 'four_hour':
                expected_diff = pd.Timedelta(hours=4)
            else:
                expected_diff = pd.Timedelta(hours=1)
            
            # Find gaps larger than tolerance
            gap_indices = time_diffs[time_diffs > pd.Timedelta(hours=gap_tolerance)].index
            
            for idx in gap_indices:
                gap_start = all_data.loc[idx - 1, 'timestamp']
                gap_end = all_data.loc[idx, 'timestamp']
                gap_duration = (gap_end - gap_start).total_seconds() / 3600  # hours
                
                if self.timeframe in ['day', 'daily']:
                    expected_candles = int(gap_duration / 24)
                elif self.timeframe in ['hour', 'hourly']:
                    expected_candles = int(gap_duration)
                else:
                    expected_candles = int(gap_duration / 4)
                
                gaps.append(TimeRange(
                    start=gap_start,
                    end=gap_end,
                    expected_candles=expected_candles
                ))
                
                logger.warning(
                    f"Gap detected: {gap_duration:.1f} hours",
                    start=gap_start.isoformat(),
                    end=gap_end.isoformat()
                )
        
        return gaps
    
    def merge_data(self) -> pd.DataFrame:
        """Merges all chunks into single DataFrame"""
        if not self.collected_data:
            return pd.DataFrame()
        
        # Concatenate all data
        merged = pd.concat(self.collected_data, ignore_index=True)
        
        # Remove duplicates and sort
        if 'timestamp' in merged.columns:
            merged = merged.drop_duplicates('timestamp')
            merged = merged.sort_values('timestamp')
            merged = merged.reset_index(drop=True)
        
        logger.info(
            f"Merged {len(self.collected_data)} chunks",
            total_candles=len(merged),
            date_range=f"{merged['timestamp'].min()} to {merged['timestamp'].max()}" if 'timestamp' in merged.columns else "N/A"
        )
        
        return merged
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total_candles = sum(chunk.candles_collected for chunk in self.completed_chunks)
        expected_total = sum(chunk.expected_candles for chunk in self.chunks)
        
        return {
            'chunks_completed': len(self.completed_chunks),
            'chunks_total': len(self.chunks),
            'chunks_failed': len(self.failed_chunks),
            'candles_collected': total_candles,
            'candles_expected': expected_total,
            'completeness_percentage': (total_candles / expected_total * 100) if expected_total > 0 else 0,
            'current_chunk': self.current_chunk_index
        }
    
    def update_before_timestamp(self, chunk: ChunkConfig, last_timestamp: int):
        """Update before_timestamp for next pagination"""
        # Find next chunk
        next_chunk_index = chunk.chunk_id + 1
        if next_chunk_index < len(self.chunks):
            self.chunks[next_chunk_index].before_timestamp = last_timestamp - 1
            logger.debug(f"Updated before_timestamp for chunk {next_chunk_index}: {last_timestamp - 1}")
    
    def retry_chunk(self, chunk: ChunkConfig):
        """Mark chunk for retry"""
        chunk.completed = False
        chunk.retry_count += 1
        logger.info(f"Chunk {chunk.chunk_id} marked for retry (attempt {chunk.retry_count})")
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint"""
        self.current_chunk_index = state.get('current_chunk', 0)
        self.completed_chunks = []
        self.failed_chunks = []
        
        # Restore chunk states
        chunk_states = state.get('chunks', [])
        for i, chunk_state in enumerate(chunk_states):
            if i < len(self.chunks):
                self.chunks[i].completed = chunk_state.get('completed', False)
                self.chunks[i].candles_collected = chunk_state.get('candles_collected', 0)
                self.chunks[i].retry_count = chunk_state.get('retry_count', 0)
                
                if self.chunks[i].completed:
                    self.completed_chunks.append(self.chunks[i])
        
        logger.info(
            f"Loaded state: {len(self.completed_chunks)}/{len(self.chunks)} chunks completed",
            current_chunk=self.current_chunk_index
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpoint"""
        return {
            'current_chunk': self.current_chunk_index,
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'completed_count': len(self.completed_chunks),
            'failed_count': len(self.failed_chunks)
        }


class CollectionProgress:
    """Tracks and persists collection progress for recovery"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize progress tracker
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "corpus_collection_progress.json"
        self.progress_data = {}
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load existing checkpoint if available"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.progress_data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                self.progress_data = {}
        else:
            logger.info("No checkpoint found, starting fresh")
    
    def save_checkpoint(self, token: str, timeframe: str, chunk_manager: ChunkManager, data_collected: pd.DataFrame = None):
        """Save progress for recovery"""
        key = f"{token}_{timeframe}"
        
        self.progress_data[key] = {
            'token': token,
            'timeframe': timeframe,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chunk_state': chunk_manager.get_state(),
            'progress': chunk_manager.get_progress_summary()
        }
        
        # Save collected data if provided
        if data_collected is not None and len(data_collected) > 0:
            data_file = self.checkpoint_dir / f"{key}_data.parquet"
            data_collected.to_parquet(data_file)
            self.progress_data[key]['data_file'] = str(data_file)
            logger.debug(f"Saved {len(data_collected)} records to {data_file}")
        
        # Write checkpoint
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
            logger.debug(f"Checkpoint saved for {key}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def can_resume(self, token: str, timeframe: str) -> bool:
        """Check if partial data exists for resume"""
        key = f"{token}_{timeframe}"
        if key not in self.progress_data:
            return False
        
        # Check if data file exists
        if 'data_file' in self.progress_data[key]:
            data_file = Path(self.progress_data[key]['data_file'])
            return data_file.exists()
        
        return False
    
    def get_resume_point(self, token: str, timeframe: str) -> Dict[str, Any]:
        """Returns where to resume collection"""
        key = f"{token}_{timeframe}"
        return self.progress_data.get(key, {})
    
    def clear_checkpoint(self, token: str = None, timeframe: str = None):
        """Clear checkpoint data"""
        if token and timeframe:
            key = f"{token}_{timeframe}"
            if key in self.progress_data:
                # Remove data file if exists
                if 'data_file' in self.progress_data[key]:
                    data_file = Path(self.progress_data[key]['data_file'])
                    if data_file.exists():
                        data_file.unlink()
                
                del self.progress_data[key]
                self.save_checkpoint(token, timeframe, None)
                logger.info(f"Cleared checkpoint for {key}")
        else:
            # Clear all checkpoints
            for key in list(self.progress_data.keys()):
                if 'data_file' in self.progress_data[key]:
                    data_file = Path(self.progress_data[key]['data_file'])
                    if data_file.exists():
                        data_file.unlink()
            
            self.progress_data = {}
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            logger.info("Cleared all checkpoints")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints"""
        summary = {
            'total_checkpoints': len(self.progress_data),
            'tokens': {}
        }
        
        for key, data in self.progress_data.items():
            token = data.get('token', 'unknown')
            if token not in summary['tokens']:
                summary['tokens'][token] = []
            
            summary['tokens'][token].append({
                'timeframe': data.get('timeframe'),
                'progress': data.get('progress', {}).get('completeness_percentage', 0),
                'timestamp': data.get('timestamp')
            })
        
        return summary


class CollectionMonitor:
    """Real-time collection monitoring and reporting"""
    
    def __init__(self):
        """Initialize collection monitor"""
        self.start_time = datetime.now()
        self.token_progress = {}
        
    def log_progress(self, token: str, timeframe: str, chunks_done: int, chunks_total: int, candles_collected: int, candles_expected: int):
        """Log collection progress"""
        key = f"{token}/{timeframe}"
        
        # Calculate metrics
        chunk_progress = (chunks_done / chunks_total * 100) if chunks_total > 0 else 0
        candle_progress = (candles_collected / candles_expected * 100) if candles_expected > 0 else 0
        
        # Estimate time remaining
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if chunks_done > 0:
            avg_time_per_chunk = elapsed / chunks_done
            remaining_chunks = chunks_total - chunks_done
            eta_seconds = avg_time_per_chunk * remaining_chunks
            eta_str = f"{int(eta_seconds)}s"
        else:
            eta_str = "calculating..."
        
        # Store progress
        self.token_progress[key] = {
            'chunks_done': chunks_done,
            'chunks_total': chunks_total,
            'candles_collected': candles_collected,
            'candles_expected': candles_expected,
            'chunk_progress': chunk_progress,
            'candle_progress': candle_progress
        }
        
        # Log progress
        logger.info(
            f"[{key}] Progress: {chunks_done}/{chunks_total} chunks | "
            f"{candles_collected}/{candles_expected} candles ({candle_progress:.1f}%) | "
            f"ETA: {eta_str}"
        )
    
    def log_gap_detected(self, token: str, timeframe: str, gap_start: datetime, gap_end: datetime, missing_candles: int):
        """Log detected gaps for transparency"""
        gap_duration = (gap_end - gap_start).total_seconds() / 3600  # hours
        
        logger.warning(
            f"[{token}/{timeframe}] Gap detected",
            duration_hours=f"{gap_duration:.1f}",
            missing_candles=missing_candles,
            start=gap_start.isoformat(),
            end=gap_end.isoformat()
        )
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate final collection summary"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'duration_seconds': total_duration,
            'duration_formatted': f"{int(total_duration/60)}m {int(total_duration%60)}s",
            'tokens': {}
        }
        
        for key, progress in self.token_progress.items():
            token, timeframe = key.split('/')
            if token not in summary['tokens']:
                summary['tokens'][token] = {}
            
            summary['tokens'][token][timeframe] = {
                'completeness': f"{progress['candle_progress']:.1f}%",
                'candles_collected': progress['candles_collected'],
                'candles_expected': progress['candles_expected'],
                'chunks_processed': f"{progress['chunks_done']}/{progress['chunks_total']}"
            }
        
        # Log summary
        logger.info("Collection Summary", **summary)
        
        return summary


class DataValidator:
    """Validates collected data completeness and quality"""
    
    def __init__(self, validation_config: Dict[str, Any] = None):
        """
        Initialize data validator
        
        Args:
            validation_config: Validation configuration from YAML
        """
        self.validation_config = validation_config or {}
        self.min_completeness = self.validation_config.get('min_completeness', 0.85)
        self.gap_tolerance_hours = self.validation_config.get('gap_tolerance_hours', 48)
    
    def validate_timeframe_coverage(self, df: pd.DataFrame, expected_days: int) -> Tuple[bool, str]:
        """Check if date range covers expected period"""
        if df.empty:
            return False, "No data collected"
        
        if 'timestamp' not in df.columns:
            return False, "No timestamp column found"
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        actual_days = (df['timestamp'].max() - df['timestamp'].min()).days
        
        coverage_ratio = actual_days / expected_days if expected_days > 0 else 0
        
        if coverage_ratio >= self.min_completeness:
            return True, f"Coverage OK: {actual_days}/{expected_days} days ({coverage_ratio*100:.1f}%)"
        else:
            return False, f"Insufficient coverage: {actual_days}/{expected_days} days ({coverage_ratio*100:.1f}%)"
    
    def validate_candle_count(self, df: pd.DataFrame, timeframe: str, days: int) -> Tuple[bool, str]:
        """Verify expected number of candles"""
        if df.empty:
            return False, "No data collected"
        
        # Calculate expected candles
        if timeframe in ['day', 'daily']:
            expected = days
        elif timeframe in ['hour', 'hourly']:
            expected = days * 24
        elif timeframe == 'four_hour':
            expected = days * 6
        else:
            expected = days * 24
        
        actual = len(df)
        completeness = actual / expected if expected > 0 else 0
        
        if completeness >= self.min_completeness:
            return True, f"Candle count OK: {actual}/{expected} ({completeness*100:.1f}%)"
        else:
            return False, f"Insufficient candles: {actual}/{expected} ({completeness*100:.1f}%)"
    
    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict[str, Any]]:
        """Identify gaps larger than threshold"""
        if df.empty or 'timestamp' not in df.columns:
            return []
        
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Expected time difference
        if timeframe in ['day', 'daily']:
            expected_diff = pd.Timedelta(days=1)
        elif timeframe in ['hour', 'hourly']:
            expected_diff = pd.Timedelta(hours=1)
        elif timeframe == 'four_hour':
            expected_diff = pd.Timedelta(hours=4)
        else:
            expected_diff = pd.Timedelta(hours=1)
        
        # Find gaps
        time_diffs = df['timestamp'].diff()
        gap_threshold = pd.Timedelta(hours=self.gap_tolerance_hours)
        
        gaps = []
        gap_indices = time_diffs[time_diffs > gap_threshold].index
        
        for idx in gap_indices:
            gap_start = df.loc[idx - 1, 'timestamp']
            gap_end = df.loc[idx, 'timestamp']
            gap_duration = (gap_end - gap_start).total_seconds() / 3600
            
            gaps.append({
                'start': gap_start.isoformat(),
                'end': gap_end.isoformat(),
                'duration_hours': gap_duration,
                'missing_periods': int(gap_duration / (expected_diff.total_seconds() / 3600))
            })
        
        return gaps
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check for invalid values"""
        issues = []
        
        if df.empty:
            return False, ["No data to validate"]
        
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            issues.append(f"NaN values found: {nan_cols.to_dict()}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Negative values in {col}: {negative_count} rows")
        
        # Check OHLC consistency (high >= low, high >= open/close, etc.)
        if all(col in df.columns for col in price_cols):
            invalid_high_low = (df['high'] < df['low']).sum()
            if invalid_high_low > 0:
                issues.append(f"High < Low in {invalid_high_low} rows")
            
            invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
            if invalid_high > 0:
                issues.append(f"High not highest in {invalid_high} rows")
            
            invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
            if invalid_low > 0:
                issues.append(f"Low not lowest in {invalid_low} rows")
        
        # Check for zero volume (might be OK for some tokens)
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > len(df) * 0.5:  # More than 50% zero volume
                issues.append(f"Excessive zero volume: {zero_volume}/{len(df)} rows")
        
        return len(issues) == 0, issues
    
    def generate_validation_report(self, df: pd.DataFrame, timeframe: str, expected_days: int) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timeframe': timeframe,
            'expected_days': expected_days,
            'actual_rows': len(df),
            'validations': {}
        }
        
        # Coverage validation
        coverage_valid, coverage_msg = self.validate_timeframe_coverage(df, expected_days)
        report['validations']['coverage'] = {
            'valid': coverage_valid,
            'message': coverage_msg
        }
        
        # Candle count validation
        count_valid, count_msg = self.validate_candle_count(df, timeframe, expected_days)
        report['validations']['candle_count'] = {
            'valid': count_valid,
            'message': count_msg
        }
        
        # Gap detection
        gaps = self.detect_gaps(df, timeframe)
        report['validations']['gaps'] = {
            'count': len(gaps),
            'gaps': gaps
        }
        
        # Data quality
        quality_valid, quality_issues = self.validate_data_quality(df)
        report['validations']['quality'] = {
            'valid': quality_valid,
            'issues': quality_issues
        }
        
        # Overall validation
        report['valid'] = all([
            coverage_valid,
            count_valid,
            quality_valid,
            len(gaps) == 0
        ])
        
        return report