"""
Log Retention and Archival Policies for Phase 6.2

This module provides comprehensive log retention, archival, and lifecycle management
for compliance with financial regulations and operational requirements.
"""

import asyncio
import gzip
import json
import shutil
import tarfile
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import os
import hashlib

import structlog

logger = structlog.get_logger()


class RetentionPeriod(Enum):
    """Standard retention periods for different log types."""
    IMMEDIATE = "immediate"  # 0 days
    DAILY = "daily"  # 1 day
    WEEKLY = "weekly"  # 7 days
    MONTHLY = "monthly"  # 30 days
    QUARTERLY = "quarterly"  # 90 days
    YEARLY = "yearly"  # 365 days
    FINANCIAL_RECORDS = "financial_records"  # 7 years (regulatory requirement)
    SECURITY_LOGS = "security_logs"  # 3 years
    AUDIT_TRAIL = "audit_trail"  # 10 years
    PERMANENT = "permanent"  # Never delete


class ArchivalFormat(Enum):
    """Supported archival formats."""
    GZIP = "gzip"
    TAR_GZ = "tar.gz"
    TAR_BZIP2 = "tar.bz2"
    ZIP = "zip"


class StorageTier(Enum):
    """Storage tiers for different access patterns."""
    HOT = "hot"  # Frequently accessed, local storage
    WARM = "warm"  # Occasionally accessed, compressed local
    COLD = "cold"  # Rarely accessed, archived to cloud
    FROZEN = "frozen"  # Compliance storage, minimal access


@dataclass
class RetentionPolicy:
    """Definition of log retention policy."""
    name: str
    retention_period: RetentionPeriod
    archive_after_days: int = 30
    compress_after_days: int = 7
    storage_tier_progression: List[StorageTier] = field(default_factory=lambda: [
        StorageTier.HOT, StorageTier.WARM, StorageTier.COLD
    ])
    archival_format: ArchivalFormat = ArchivalFormat.GZIP
    checksum_validation: bool = True
    encryption_required: bool = False
    compliance_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_retention_days(self) -> int:
        """Get retention period in days."""
        retention_map = {
            RetentionPeriod.IMMEDIATE: 0,
            RetentionPeriod.DAILY: 1,
            RetentionPeriod.WEEKLY: 7,
            RetentionPeriod.MONTHLY: 30,
            RetentionPeriod.QUARTERLY: 90,
            RetentionPeriod.YEARLY: 365,
            RetentionPeriod.FINANCIAL_RECORDS: 365 * 7,  # 7 years
            RetentionPeriod.SECURITY_LOGS: 365 * 3,  # 3 years
            RetentionPeriod.AUDIT_TRAIL: 365 * 10,  # 10 years
            RetentionPeriod.PERMANENT: -1  # Never delete
        }
        return retention_map[self.retention_period]


@dataclass
class ArchivalMetadata:
    """Metadata for archived log files."""
    original_path: str
    archive_path: str
    original_size: int
    compressed_size: int
    checksum: str
    compression_ratio: float
    archived_at: datetime
    retention_policy: str
    storage_tier: StorageTier
    expiry_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "original_path": self.original_path,
            "archive_path": self.archive_path,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "checksum": self.checksum,
            "compression_ratio": self.compression_ratio,
            "archived_at": self.archived_at.isoformat(),
            "retention_policy": self.retention_policy,
            "storage_tier": self.storage_tier.value,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None
        }


class LogRetentionManager:
    """Manager for log retention and archival operations."""
    
    def __init__(self,
                 log_directory: Path,
                 archive_directory: Path,
                 policies: Optional[Dict[str, RetentionPolicy]] = None,
                 metadata_file: Optional[Path] = None):
        """Initialize log retention manager."""
        self.log_directory = Path(log_directory)
        self.archive_directory = Path(archive_directory)
        self.policies = policies or self._get_default_policies()
        self.metadata_file = metadata_file or (self.archive_directory / "archival_metadata.json")
        
        # Ensure directories exist
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.archival_metadata: Dict[str, ArchivalMetadata] = self._load_metadata()
        
        logger.info("LogRetentionManager initialized",
                   log_directory=str(self.log_directory),
                   archive_directory=str(self.archive_directory),
                   policies_count=len(self.policies))
    
    def _get_default_policies(self) -> Dict[str, RetentionPolicy]:
        """Get default retention policies for different log types."""
        return {
            "financial_transaction": RetentionPolicy(
                name="financial_transaction",
                retention_period=RetentionPeriod.FINANCIAL_RECORDS,
                archive_after_days=30,
                compress_after_days=7,
                encryption_required=True,
                compliance_metadata={
                    "regulation": "SOX, MiFID II",
                    "data_classification": "financial",
                    "retention_reason": "regulatory_compliance"
                }
            ),
            "security_event": RetentionPolicy(
                name="security_event",
                retention_period=RetentionPeriod.SECURITY_LOGS,
                archive_after_days=90,
                compress_after_days=30,
                encryption_required=True,
                compliance_metadata={
                    "regulation": "SOC 2, GDPR",
                    "data_classification": "security",
                    "retention_reason": "security_compliance"
                }
            ),
            "audit_trail": RetentionPolicy(
                name="audit_trail",
                retention_period=RetentionPeriod.AUDIT_TRAIL,
                archive_after_days=180,
                compress_after_days=30,
                encryption_required=True,
                compliance_metadata={
                    "regulation": "SOX, GDPR, MiFID II",
                    "data_classification": "audit",
                    "retention_reason": "audit_compliance"
                }
            ),
            "system_operational": RetentionPolicy(
                name="system_operational",
                retention_period=RetentionPeriod.QUARTERLY,
                archive_after_days=30,
                compress_after_days=7,
                encryption_required=False,
                compliance_metadata={
                    "data_classification": "operational",
                    "retention_reason": "operational_analysis"
                }
            ),
            "algorithmic_decision": RetentionPolicy(
                name="algorithmic_decision",
                retention_period=RetentionPeriod.FINANCIAL_RECORDS,
                archive_after_days=30,
                compress_after_days=7,
                encryption_required=True,
                compliance_metadata={
                    "regulation": "MiFID II, GDPR",
                    "data_classification": "algorithmic",
                    "retention_reason": "algorithmic_compliance"
                }
            ),
            "performance_metrics": RetentionPolicy(
                name="performance_metrics",
                retention_period=RetentionPeriod.YEARLY,
                archive_after_days=90,
                compress_after_days=30,
                encryption_required=False,
                compliance_metadata={
                    "data_classification": "performance",
                    "retention_reason": "performance_analysis"
                }
            )
        }
    
    def _load_metadata(self) -> Dict[str, ArchivalMetadata]:
        """Load archival metadata from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for file_path, meta_dict in data.items():
                metadata[file_path] = ArchivalMetadata(
                    original_path=meta_dict["original_path"],
                    archive_path=meta_dict["archive_path"],
                    original_size=meta_dict["original_size"],
                    compressed_size=meta_dict["compressed_size"],
                    checksum=meta_dict["checksum"],
                    compression_ratio=meta_dict["compression_ratio"],
                    archived_at=datetime.fromisoformat(meta_dict["archived_at"]),
                    retention_policy=meta_dict["retention_policy"],
                    storage_tier=StorageTier(meta_dict["storage_tier"]),
                    expiry_date=datetime.fromisoformat(meta_dict["expiry_date"]) if meta_dict["expiry_date"] else None
                )
            
            return metadata
            
        except Exception as e:
            logger.error("Failed to load archival metadata", error=str(e))
            return {}
    
    def _save_metadata(self):
        """Save archival metadata to file."""
        try:
            data = {
                file_path: metadata.to_dict()
                for file_path, metadata in self.archival_metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error("Failed to save archival metadata", error=str(e))
    
    def add_policy(self, policy: RetentionPolicy):
        """Add or update a retention policy."""
        self.policies[policy.name] = policy
        logger.info("Retention policy added", policy_name=policy.name)
    
    def get_policy(self, policy_name: str) -> Optional[RetentionPolicy]:
        """Get retention policy by name."""
        return self.policies.get(policy_name)
    
    def classify_log_file(self, file_path: Path) -> str:
        """Classify log file to determine appropriate retention policy."""
        file_content = self._read_log_sample(file_path)
        
        # Analyze content to determine classification
        if any(word in file_content.lower() for word in ["trade", "transaction", "order", "financial"]):
            return "financial_transaction"
        elif any(word in file_content.lower() for word in ["security", "auth", "login", "breach"]):
            return "security_event"
        elif any(word in file_content.lower() for word in ["audit", "compliance"]):
            return "audit_trail"
        elif any(word in file_content.lower() for word in ["model", "prediction", "algorithm", "ml"]):
            return "algorithmic_decision"
        elif any(word in file_content.lower() for word in ["performance", "latency", "cpu", "memory"]):
            return "performance_metrics"
        else:
            return "system_operational"
    
    def _read_log_sample(self, file_path: Path, sample_size: int = 1024) -> str:
        """Read a sample of log file for classification."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(sample_size)
        except Exception:
            return ""
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error("Failed to calculate checksum", file=str(file_path), error=str(e))
            return ""
    
    async def compress_file(self, file_path: Path, archive_format: ArchivalFormat = ArchivalFormat.GZIP) -> Optional[Path]:
        """Compress a log file."""
        try:
            if archive_format == ArchivalFormat.GZIP:
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                return compressed_path
            
            elif archive_format == ArchivalFormat.TAR_GZ:
                compressed_path = file_path.with_suffix('.tar.gz')
                
                with tarfile.open(compressed_path, 'w:gz') as tar:
                    tar.add(file_path, arcname=file_path.name)
                
                return compressed_path
            
            else:
                logger.warning("Unsupported archive format", format=archive_format.value)
                return None
                
        except Exception as e:
            logger.error("Failed to compress file", file=str(file_path), error=str(e))
            return None
    
    async def archive_file(self, file_path: Path, policy_name: str) -> bool:
        """Archive a log file according to retention policy."""
        policy = self.get_policy(policy_name)
        if not policy:
            logger.error("Unknown retention policy", policy=policy_name)
            return False
        
        try:
            # Calculate original file stats
            original_size = file_path.stat().st_size
            original_checksum = self._calculate_checksum(file_path)
            
            # Create archive directory structure
            archive_date = datetime.now(timezone.utc)
            archive_subdir = self.archive_directory / policy_name / archive_date.strftime("%Y/%m")
            archive_subdir.mkdir(parents=True, exist_ok=True)
            
            # Compress file
            compressed_path = await self.compress_file(file_path, policy.archival_format)
            if not compressed_path:
                return False
            
            # Move compressed file to archive
            archive_path = archive_subdir / compressed_path.name
            shutil.move(str(compressed_path), str(archive_path))
            
            # Calculate compressed stats
            compressed_size = archive_path.stat().st_size
            compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0
            
            # Calculate expiry date
            retention_days = policy.get_retention_days()
            expiry_date = None if retention_days == -1 else archive_date + timedelta(days=retention_days)
            
            # Create metadata
            metadata = ArchivalMetadata(
                original_path=str(file_path),
                archive_path=str(archive_path),
                original_size=original_size,
                compressed_size=compressed_size,
                checksum=original_checksum,
                compression_ratio=compression_ratio,
                archived_at=archive_date,
                retention_policy=policy_name,
                storage_tier=StorageTier.WARM,
                expiry_date=expiry_date
            )
            
            # Store metadata
            self.archival_metadata[str(file_path)] = metadata
            self._save_metadata()
            
            # Remove original file
            file_path.unlink()
            
            logger.info("File archived successfully",
                       original_file=str(file_path),
                       archive_path=str(archive_path),
                       compression_ratio=f"{compression_ratio:.2%}",
                       policy=policy_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to archive file", file=str(file_path), error=str(e))
            return False
    
    async def restore_file(self, original_path: str, restore_directory: Optional[Path] = None) -> Optional[Path]:
        """Restore an archived file."""
        metadata = self.archival_metadata.get(original_path)
        if not metadata:
            logger.error("No archival metadata found", file=original_path)
            return None
        
        try:
            archive_path = Path(metadata.archive_path)
            if not archive_path.exists():
                logger.error("Archived file not found", archive_path=str(archive_path))
                return None
            
            # Determine restore location
            if restore_directory:
                restore_path = restore_directory / Path(original_path).name
            else:
                restore_path = Path(original_path)
            
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Decompress file
            if archive_path.suffix == '.gz':
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(restore_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Handle other formats
                shutil.copy2(archive_path, restore_path)
            
            # Verify checksum
            restored_checksum = self._calculate_checksum(restore_path)
            if restored_checksum != metadata.checksum:
                logger.error("Checksum mismatch after restore",
                           expected=metadata.checksum,
                           actual=restored_checksum)
                restore_path.unlink()
                return None
            
            logger.info("File restored successfully",
                       original_path=original_path,
                       restore_path=str(restore_path))
            
            return restore_path
            
        except Exception as e:
            logger.error("Failed to restore file", file=original_path, error=str(e))
            return None
    
    async def cleanup_expired_files(self) -> Dict[str, int]:
        """Clean up files that have exceeded their retention period."""
        results = {
            "scanned": 0,
            "expired": 0,
            "deleted": 0,
            "errors": 0
        }
        
        now = datetime.now(timezone.utc)
        expired_files = []
        
        # Check archived files
        for file_path, metadata in self.archival_metadata.items():
            results["scanned"] += 1
            
            if metadata.expiry_date and metadata.expiry_date <= now:
                expired_files.append((file_path, metadata))
                results["expired"] += 1
        
        # Delete expired files
        for file_path, metadata in expired_files:
            try:
                archive_path = Path(metadata.archive_path)
                if archive_path.exists():
                    archive_path.unlink()
                
                # Remove from metadata
                del self.archival_metadata[file_path]
                results["deleted"] += 1
                
                logger.info("Expired file deleted",
                           file=file_path,
                           expiry_date=metadata.expiry_date.isoformat())
                
            except Exception as e:
                logger.error("Failed to delete expired file",
                           file=file_path,
                           error=str(e))
                results["errors"] += 1
        
        # Save updated metadata
        if expired_files:
            self._save_metadata()
        
        logger.info("Retention cleanup completed", **results)
        return results
    
    async def process_log_directory(self) -> Dict[str, int]:
        """Process all log files in the directory according to retention policies."""
        results = {
            "scanned": 0,
            "compressed": 0,
            "archived": 0,
            "errors": 0
        }
        
        now = datetime.now(timezone.utc)
        
        # Scan all log files
        for file_path in self.log_directory.rglob("*.log"):
            if not file_path.is_file():
                continue
            
            results["scanned"] += 1
            
            try:
                # Get file age
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                age_days = (now - file_mtime).days
                
                # Classify file to get policy
                policy_name = self.classify_log_file(file_path)
                policy = self.get_policy(policy_name)
                
                if not policy:
                    logger.warning("No policy found for file", file=str(file_path))
                    continue
                
                # Check if file should be archived
                if age_days >= policy.archive_after_days:
                    success = await self.archive_file(file_path, policy_name)
                    if success:
                        results["archived"] += 1
                    else:
                        results["errors"] += 1
                
                # Check if file should be compressed (but not archived yet)
                elif age_days >= policy.compress_after_days and file_path.suffix != '.gz':
                    compressed_path = await self.compress_file(file_path, policy.archival_format)
                    if compressed_path:
                        file_path.unlink()  # Remove original
                        results["compressed"] += 1
                    else:
                        results["errors"] += 1
                
            except Exception as e:
                logger.error("Error processing file", file=str(file_path), error=str(e))
                results["errors"] += 1
        
        logger.info("Log directory processing completed", **results)
        return results
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get statistics about retained logs."""
        stats = {
            "total_archived_files": len(self.archival_metadata),
            "total_archive_size": 0,
            "policies": {},
            "storage_tiers": {tier.value: 0 for tier in StorageTier},
            "expiring_soon": 0  # Files expiring in next 30 days
        }
        
        now = datetime.now(timezone.utc)
        thirty_days = now + timedelta(days=30)
        
        for metadata in self.archival_metadata.values():
            stats["total_archive_size"] += metadata.compressed_size
            
            # Policy stats
            policy = metadata.retention_policy
            if policy not in stats["policies"]:
                stats["policies"][policy] = {"count": 0, "size": 0}
            stats["policies"][policy]["count"] += 1
            stats["policies"][policy]["size"] += metadata.compressed_size
            
            # Storage tier stats
            stats["storage_tiers"][metadata.storage_tier.value] += 1
            
            # Expiring soon
            if metadata.expiry_date and metadata.expiry_date <= thirty_days:
                stats["expiring_soon"] += 1
        
        return stats
    
    async def validate_archive_integrity(self) -> Dict[str, Any]:
        """Validate integrity of archived files."""
        results = {
            "checked": 0,
            "valid": 0,
            "corrupted": 0,
            "missing": 0,
            "errors": []
        }
        
        for file_path, metadata in self.archival_metadata.items():
            results["checked"] += 1
            
            try:
                archive_path = Path(metadata.archive_path)
                
                if not archive_path.exists():
                    results["missing"] += 1
                    results["errors"].append(f"Missing archive: {archive_path}")
                    continue
                
                # For this implementation, we'll check file size as a basic integrity check
                # In production, you might want to decompress and verify checksums
                current_size = archive_path.stat().st_size
                if current_size == metadata.compressed_size:
                    results["valid"] += 1
                else:
                    results["corrupted"] += 1
                    results["errors"].append(f"Size mismatch: {archive_path}")
                
            except Exception as e:
                results["corrupted"] += 1
                results["errors"].append(f"Error checking {file_path}: {str(e)}")
        
        logger.info("Archive integrity validation completed", **{k: v for k, v in results.items() if k != "errors"})
        
        return results


class RetentionScheduler:
    """Scheduler for automated retention tasks."""
    
    def __init__(self, retention_manager: LogRetentionManager):
        """Initialize retention scheduler."""
        self.retention_manager = retention_manager
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self,
                   process_interval_hours: int = 24,
                   cleanup_interval_hours: int = 168):  # Weekly cleanup
        """Start automated retention tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Schedule periodic processing
        self._tasks.append(
            asyncio.create_task(
                self._periodic_task(
                    self.retention_manager.process_log_directory,
                    process_interval_hours * 3600,
                    "log_processing"
                )
            )
        )
        
        # Schedule periodic cleanup
        self._tasks.append(
            asyncio.create_task(
                self._periodic_task(
                    self.retention_manager.cleanup_expired_files,
                    cleanup_interval_hours * 3600,
                    "expired_cleanup"
                )
            )
        )
        
        logger.info("Retention scheduler started",
                   process_interval_hours=process_interval_hours,
                   cleanup_interval_hours=cleanup_interval_hours)
    
    async def stop(self):
        """Stop automated retention tasks."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        logger.info("Retention scheduler stopped")
    
    async def _periodic_task(self, func: Callable, interval_seconds: int, task_name: str):
        """Run a function periodically."""
        while self.is_running:
            try:
                logger.debug("Starting periodic task", task=task_name)
                result = await func()
                logger.info("Periodic task completed", task=task_name, result=result)
                
            except Exception as e:
                logger.error("Periodic task failed", task=task_name, error=str(e))
            
            # Wait for next interval
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
    
    async def force_cleanup(self) -> Dict[str, Any]:
        """Force immediate cleanup of expired files."""
        logger.info("Forcing immediate cleanup")
        return await self.retention_manager.cleanup_expired_files()
    
    async def force_processing(self) -> Dict[str, Any]:
        """Force immediate processing of log directory."""
        logger.info("Forcing immediate log processing")
        return await self.retention_manager.process_log_directory()


# Convenience functions for common operations
async def setup_default_retention(log_directory: Path, archive_directory: Path) -> LogRetentionManager:
    """Set up log retention with default policies."""
    manager = LogRetentionManager(log_directory, archive_directory)
    return manager


async def archive_logs_by_pattern(retention_manager: LogRetentionManager,
                                 pattern: str,
                                 policy_name: str) -> int:
    """Archive all log files matching a pattern."""
    archived_count = 0
    
    for file_path in retention_manager.log_directory.rglob(pattern):
        if file_path.is_file():
            success = await retention_manager.archive_file(file_path, policy_name)
            if success:
                archived_count += 1
    
    return archived_count


def get_compliance_report(retention_manager: LogRetentionManager) -> Dict[str, Any]:
    """Generate compliance report for retention policies."""
    stats = retention_manager.get_retention_stats()
    
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": stats["total_archived_files"],
        "total_size_mb": stats["total_archive_size"] / (1024 * 1024),
        "policies_compliance": {},
        "retention_status": "compliant",
        "recommendations": []
    }
    
    # Check each policy for compliance
    for policy_name, policy in retention_manager.policies.items():
        policy_stats = stats["policies"].get(policy_name, {"count": 0, "size": 0})
        
        compliance_info = {
            "files_count": policy_stats["count"],
            "total_size_mb": policy_stats["size"] / (1024 * 1024),
            "retention_period": policy.retention_period.value,
            "encryption_required": policy.encryption_required,
            "compliance_metadata": policy.compliance_metadata
        }
        
        report["policies_compliance"][policy_name] = compliance_info
    
    # Add recommendations
    if stats["expiring_soon"] > 100:
        report["recommendations"].append(
            f"High number of files ({stats['expiring_soon']}) expiring soon. Consider reviewing retention policies."
        )
    
    return report