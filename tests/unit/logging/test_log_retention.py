"""
Test suite for log retention and archival system (Phase 6.2)

This module provides comprehensive TDD tests for the log retention,
archival policies, and lifecycle management features.
"""

import pytest
import asyncio
import json
import tempfile
import gzip
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the log retention system
from src.logging.log_retention import (
    RetentionPeriod, ArchivalFormat, StorageTier, RetentionPolicy,
    ArchivalMetadata, LogRetentionManager, RetentionScheduler,
    setup_default_retention, archive_logs_by_pattern, get_compliance_report
)


class TestRetentionPeriod:
    """Test retention period enumeration."""
    
    def test_retention_periods_have_correct_values(self):
        """Test that retention periods have expected values."""
        assert RetentionPeriod.IMMEDIATE.value == "immediate"
        assert RetentionPeriod.DAILY.value == "daily"
        assert RetentionPeriod.WEEKLY.value == "weekly"
        assert RetentionPeriod.MONTHLY.value == "monthly"
        assert RetentionPeriod.QUARTERLY.value == "quarterly"
        assert RetentionPeriod.YEARLY.value == "yearly"
        assert RetentionPeriod.FINANCIAL_RECORDS.value == "financial_records"
        assert RetentionPeriod.SECURITY_LOGS.value == "security_logs"
        assert RetentionPeriod.AUDIT_TRAIL.value == "audit_trail"
        assert RetentionPeriod.PERMANENT.value == "permanent"


class TestArchivalFormat:
    """Test archival format enumeration."""
    
    def test_archival_formats_have_correct_values(self):
        """Test that archival formats have expected values."""
        assert ArchivalFormat.GZIP.value == "gzip"
        assert ArchivalFormat.TAR_GZ.value == "tar.gz"
        assert ArchivalFormat.TAR_BZIP2.value == "tar.bz2"
        assert ArchivalFormat.ZIP.value == "zip"


class TestStorageTier:
    """Test storage tier enumeration."""
    
    def test_storage_tiers_have_correct_values(self):
        """Test that storage tiers have expected values."""
        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"
        assert StorageTier.FROZEN.value == "frozen"


class TestRetentionPolicy:
    """Test retention policy functionality."""
    
    def test_retention_policy_creation(self):
        """Test retention policy creation with defaults."""
        policy = RetentionPolicy(
            name="test_policy",
            retention_period=RetentionPeriod.MONTHLY
        )
        
        assert policy.name == "test_policy"
        assert policy.retention_period == RetentionPeriod.MONTHLY
        assert policy.archive_after_days == 30
        assert policy.compress_after_days == 7
        assert policy.archival_format == ArchivalFormat.GZIP
        assert policy.checksum_validation is True
        assert policy.encryption_required is False
        assert isinstance(policy.compliance_metadata, dict)
    
    def test_retention_policy_with_custom_values(self):
        """Test retention policy creation with custom values."""
        policy = RetentionPolicy(
            name="financial_policy",
            retention_period=RetentionPeriod.FINANCIAL_RECORDS,
            archive_after_days=90,
            compress_after_days=30,
            archival_format=ArchivalFormat.TAR_GZ,
            encryption_required=True,
            compliance_metadata={
                "regulation": "SOX",
                "classification": "financial"
            }
        )
        
        assert policy.name == "financial_policy"
        assert policy.retention_period == RetentionPeriod.FINANCIAL_RECORDS
        assert policy.archive_after_days == 90
        assert policy.compress_after_days == 30
        assert policy.archival_format == ArchivalFormat.TAR_GZ
        assert policy.encryption_required is True
        assert policy.compliance_metadata["regulation"] == "SOX"
    
    def test_get_retention_days(self):
        """Test retention period conversion to days."""
        test_cases = [
            (RetentionPeriod.IMMEDIATE, 0),
            (RetentionPeriod.DAILY, 1),
            (RetentionPeriod.WEEKLY, 7),
            (RetentionPeriod.MONTHLY, 30),
            (RetentionPeriod.QUARTERLY, 90),
            (RetentionPeriod.YEARLY, 365),
            (RetentionPeriod.FINANCIAL_RECORDS, 365 * 7),
            (RetentionPeriod.SECURITY_LOGS, 365 * 3),
            (RetentionPeriod.AUDIT_TRAIL, 365 * 10),
            (RetentionPeriod.PERMANENT, -1)
        ]
        
        for period, expected_days in test_cases:
            policy = RetentionPolicy(name="test", retention_period=period)
            assert policy.get_retention_days() == expected_days


class TestArchivalMetadata:
    """Test archival metadata functionality."""
    
    def test_archival_metadata_creation(self):
        """Test archival metadata creation."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=365)
        
        metadata = ArchivalMetadata(
            original_path="/logs/test.log",
            archive_path="/archive/test.log.gz",
            original_size=1024,
            compressed_size=256,
            checksum="abc123",
            compression_ratio=0.75,
            archived_at=now,
            retention_policy="financial_records",
            storage_tier=StorageTier.WARM,
            expiry_date=expiry
        )
        
        assert metadata.original_path == "/logs/test.log"
        assert metadata.archive_path == "/archive/test.log.gz"
        assert metadata.original_size == 1024
        assert metadata.compressed_size == 256
        assert metadata.checksum == "abc123"
        assert metadata.compression_ratio == 0.75
        assert metadata.archived_at == now
        assert metadata.retention_policy == "financial_records"
        assert metadata.storage_tier == StorageTier.WARM
        assert metadata.expiry_date == expiry
    
    def test_archival_metadata_to_dict(self):
        """Test archival metadata conversion to dictionary."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=365)
        
        metadata = ArchivalMetadata(
            original_path="/logs/test.log",
            archive_path="/archive/test.log.gz",
            original_size=1024,
            compressed_size=256,
            checksum="abc123",
            compression_ratio=0.75,
            archived_at=now,
            retention_policy="financial_records",
            storage_tier=StorageTier.WARM,
            expiry_date=expiry
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result["original_path"] == "/logs/test.log"
        assert result["archive_path"] == "/archive/test.log.gz"
        assert result["original_size"] == 1024
        assert result["compressed_size"] == 256
        assert result["checksum"] == "abc123"
        assert result["compression_ratio"] == 0.75
        assert result["archived_at"] == now.isoformat()
        assert result["retention_policy"] == "financial_records"
        assert result["storage_tier"] == "warm"
        assert result["expiry_date"] == expiry.isoformat()
    
    def test_archival_metadata_to_dict_with_none_expiry(self):
        """Test archival metadata conversion with None expiry date."""
        now = datetime.now(timezone.utc)
        
        metadata = ArchivalMetadata(
            original_path="/logs/test.log",
            archive_path="/archive/test.log.gz",
            original_size=1024,
            compressed_size=256,
            checksum="abc123",
            compression_ratio=0.75,
            archived_at=now,
            retention_policy="permanent",
            storage_tier=StorageTier.FROZEN
        )
        
        result = metadata.to_dict()
        assert result["expiry_date"] is None


class TestLogRetentionManager:
    """Test log retention manager functionality."""
    
    def test_log_retention_manager_initialization(self):
        """Test log retention manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            assert manager.log_directory == log_dir
            assert manager.archive_directory == archive_dir
            assert len(manager.policies) > 0  # Should have default policies
            assert log_dir.exists()
            assert archive_dir.exists()
            assert isinstance(manager.archival_metadata, dict)
    
    def test_log_retention_manager_with_custom_policies(self):
        """Test log retention manager with custom policies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            custom_policies = {
                "test_policy": RetentionPolicy(
                    name="test_policy",
                    retention_period=RetentionPeriod.WEEKLY
                )
            }
            
            manager = LogRetentionManager(log_dir, archive_dir, policies=custom_policies)
            
            assert len(manager.policies) == 1
            assert "test_policy" in manager.policies
            assert manager.policies["test_policy"].retention_period == RetentionPeriod.WEEKLY
    
    def test_get_default_policies(self):
        """Test that default policies are comprehensive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Check that all expected default policies exist
            expected_policies = [
                "financial_transaction",
                "security_event", 
                "audit_trail",
                "system_operational",
                "algorithmic_decision",
                "performance_metrics"
            ]
            
            for policy_name in expected_policies:
                assert policy_name in manager.policies
                policy = manager.policies[policy_name]
                assert isinstance(policy, RetentionPolicy)
                assert policy.name == policy_name
    
    def test_add_policy(self):
        """Test adding a new retention policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            initial_count = len(manager.policies)
            
            new_policy = RetentionPolicy(
                name="custom_policy",
                retention_period=RetentionPeriod.MONTHLY
            )
            
            manager.add_policy(new_policy)
            
            assert len(manager.policies) == initial_count + 1
            assert "custom_policy" in manager.policies
            assert manager.policies["custom_policy"] == new_policy
    
    def test_get_policy(self):
        """Test getting retention policy by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Get existing policy
            policy = manager.get_policy("financial_transaction")
            assert policy is not None
            assert policy.name == "financial_transaction"
            
            # Get non-existent policy
            missing_policy = manager.get_policy("non_existent")
            assert missing_policy is None
    
    def test_classify_log_file_financial(self):
        """Test log file classification for financial content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create a financial log file
            log_file = log_dir / "financial.log"
            log_file.write_text("Trade executed successfully for BTC/USD order")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            classification = manager.classify_log_file(log_file)
            
            assert classification == "financial_transaction"
    
    def test_classify_log_file_security(self):
        """Test log file classification for security content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create a security log file
            log_file = log_dir / "security.log"
            log_file.write_text("User authentication failed from IP 192.168.1.100")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            classification = manager.classify_log_file(log_file)
            
            assert classification == "security_event"
    
    def test_classify_log_file_ml(self):
        """Test log file classification for ML content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create an ML log file
            log_file = log_dir / "ml.log"
            log_file.write_text("Model prediction completed with 95% confidence")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            classification = manager.classify_log_file(log_file)
            
            assert classification == "algorithmic_decision"
    
    def test_classify_log_file_default(self):
        """Test log file classification for general content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create a general log file
            log_file = log_dir / "general.log"
            log_file.write_text("System startup completed successfully")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            classification = manager.classify_log_file(log_file)
            
            assert classification == "system_operational"
    
    @pytest.mark.asyncio
    async def test_compress_file_gzip(self):
        """Test GZIP file compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test file
            test_file = log_dir / "test.log"
            test_content = "This is test log content\n" * 100
            test_file.write_text(test_content)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            compressed_path = await manager.compress_file(test_file, ArchivalFormat.GZIP)
            
            assert compressed_path is not None
            assert compressed_path.suffix == ".gz"
            assert compressed_path.exists()
            
            # Verify compressed content
            with gzip.open(compressed_path, 'rt') as f:
                decompressed_content = f.read()
            
            assert decompressed_content == test_content
    
    @pytest.mark.asyncio
    async def test_compress_file_tar_gz(self):
        """Test TAR.GZ file compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test file
            test_file = log_dir / "test.log"
            test_content = "This is test log content\n"
            test_file.write_text(test_content)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            compressed_path = await manager.compress_file(test_file, ArchivalFormat.TAR_GZ)
            
            assert compressed_path is not None
            assert compressed_path.suffix == ".gz"
            assert compressed_path.exists()
            
            # Verify tar.gz content
            with tarfile.open(compressed_path, 'r:gz') as tar:
                members = tar.getnames()
                assert test_file.name in members
    
    @pytest.mark.asyncio
    async def test_archive_file(self):
        """Test file archival process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test file
            test_file = log_dir / "financial.log"
            test_content = "Trade executed: BTC/USD order filled"
            test_file.write_text(test_content)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Archive the file
            result = await manager.archive_file(test_file, "financial_transaction")
            
            assert result is True
            assert not test_file.exists()  # Original should be removed
            assert len(manager.archival_metadata) == 1
            
            # Check metadata
            metadata = list(manager.archival_metadata.values())[0]
            assert metadata.original_path == str(test_file)
            assert metadata.retention_policy == "financial_transaction"
            assert metadata.storage_tier == StorageTier.WARM
            assert metadata.checksum is not None
    
    @pytest.mark.asyncio
    async def test_archive_file_with_unknown_policy(self):
        """Test archival with unknown retention policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test file
            test_file = log_dir / "test.log"
            test_file.write_text("Test content")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Try to archive with unknown policy
            result = await manager.archive_file(test_file, "unknown_policy")
            
            assert result is False
            assert test_file.exists()  # Original should remain
            assert len(manager.archival_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_restore_file(self):
        """Test file restoration from archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            restore_dir = Path(temp_dir) / "restore"
            log_dir.mkdir(parents=True)
            restore_dir.mkdir(parents=True)
            
            # Create and archive test file
            test_file = log_dir / "test.log"
            test_content = "Test log content for restoration"
            test_file.write_text(test_content)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Archive the file
            archive_result = await manager.archive_file(test_file, "system_operational")
            assert archive_result is True
            
            # Restore the file
            restored_path = await manager.restore_file(str(test_file), restore_dir)
            
            assert restored_path is not None
            assert restored_path.exists()
            assert restored_path.read_text() == test_content
    
    @pytest.mark.asyncio
    async def test_restore_file_checksum_mismatch(self):
        """Test file restoration with checksum validation failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test file
            test_file = log_dir / "test.log"
            test_file.write_text("Test content")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Archive the file
            await manager.archive_file(test_file, "system_operational")
            
            # Corrupt the metadata checksum
            original_path = str(test_file)
            metadata = manager.archival_metadata[original_path]
            metadata.checksum = "corrupted_checksum"
            
            # Try to restore - should fail due to checksum mismatch
            restored_path = await manager.restore_file(original_path)
            
            assert restored_path is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_files(self):
        """Test cleanup of expired archived files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Create mock expired metadata
            now = datetime.now(timezone.utc)
            expired_time = now - timedelta(days=1)
            
            archive_path = archive_dir / "expired.log.gz"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text("Expired archived content")
            
            metadata = ArchivalMetadata(
                original_path="/logs/expired.log",
                archive_path=str(archive_path),
                original_size=100,
                compressed_size=50,
                checksum="test_checksum",
                compression_ratio=0.5,
                archived_at=expired_time,
                retention_policy="test_policy",
                storage_tier=StorageTier.COLD,
                expiry_date=expired_time  # Already expired
            )
            
            manager.archival_metadata["/logs/expired.log"] = metadata
            
            # Run cleanup
            results = await manager.cleanup_expired_files()
            
            assert results["scanned"] == 1
            assert results["expired"] == 1
            assert results["deleted"] == 1
            assert results["errors"] == 0
            assert len(manager.archival_metadata) == 0
            assert not archive_path.exists()
    
    @pytest.mark.asyncio
    async def test_process_log_directory(self):
        """Test processing of entire log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test log files with different ages
            now = datetime.now(timezone.utc)
            
            # Old file (should be archived)
            old_file = log_dir / "old_financial.log"
            old_file.write_text("Old trade data")
            old_time = (now - timedelta(days=35)).timestamp()
            import os
            os.utime(old_file, (old_time, old_time))
            
            # Recent file (should be compressed only)
            recent_file = log_dir / "recent_system.log"
            recent_file.write_text("Recent system data")
            recent_time = (now - timedelta(days=10)).timestamp()
            os.utime(recent_file, (recent_time, recent_time))
            
            # Very recent file (should remain unchanged)
            new_file = log_dir / "new_performance.log"
            new_file.write_text("New performance data")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Process directory
            results = await manager.process_log_directory()
            
            assert results["scanned"] == 3
            assert results["archived"] >= 1  # Old file should be archived
            assert results["errors"] == 0
    
    def test_get_retention_stats(self):
        """Test retention statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Add mock archived files
            now = datetime.now(timezone.utc)
            
            metadata1 = ArchivalMetadata(
                original_path="/logs/file1.log",
                archive_path="/archive/file1.log.gz",
                original_size=1000,
                compressed_size=300,
                checksum="checksum1",
                compression_ratio=0.7,
                archived_at=now,
                retention_policy="financial_transaction",
                storage_tier=StorageTier.WARM,
                expiry_date=now + timedelta(days=30)
            )
            
            metadata2 = ArchivalMetadata(
                original_path="/logs/file2.log",
                archive_path="/archive/file2.log.gz",
                original_size=500,
                compressed_size=150,
                checksum="checksum2",
                compression_ratio=0.7,
                archived_at=now,
                retention_policy="security_event",
                storage_tier=StorageTier.COLD
            )
            
            manager.archival_metadata["/logs/file1.log"] = metadata1
            manager.archival_metadata["/logs/file2.log"] = metadata2
            
            stats = manager.get_retention_stats()
            
            assert stats["total_archived_files"] == 2
            assert stats["total_archive_size"] == 450  # 300 + 150
            assert "financial_transaction" in stats["policies"]
            assert "security_event" in stats["policies"]
            assert stats["policies"]["financial_transaction"]["count"] == 1
            assert stats["policies"]["security_event"]["count"] == 1
            assert stats["storage_tiers"]["warm"] == 1
            assert stats["storage_tiers"]["cold"] == 1
            assert stats["expiring_soon"] == 1  # file1 expires in 30 days
    
    @pytest.mark.asyncio
    async def test_validate_archive_integrity(self):
        """Test archive integrity validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            archive_dir.mkdir(parents=True)
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Create mock archive file
            archive_file = archive_dir / "test.log.gz"
            archive_file.write_text("Compressed content")
            
            # Add metadata
            metadata = ArchivalMetadata(
                original_path="/logs/test.log",
                archive_path=str(archive_file),
                original_size=100,
                compressed_size=len(archive_file.read_text()),
                checksum="test_checksum",
                compression_ratio=0.5,
                archived_at=datetime.now(timezone.utc),
                retention_policy="test_policy",
                storage_tier=StorageTier.WARM
            )
            
            manager.archival_metadata["/logs/test.log"] = metadata
            
            # Validate integrity
            results = await manager.validate_archive_integrity()
            
            assert results["checked"] == 1
            assert results["valid"] == 1
            assert results["corrupted"] == 0
            assert results["missing"] == 0
            assert len(results["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_archive_integrity_missing_file(self):
        """Test archive integrity validation with missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Add metadata for missing file
            metadata = ArchivalMetadata(
                original_path="/logs/missing.log",
                archive_path="/archive/missing.log.gz",
                original_size=100,
                compressed_size=50,
                checksum="test_checksum",
                compression_ratio=0.5,
                archived_at=datetime.now(timezone.utc),
                retention_policy="test_policy",
                storage_tier=StorageTier.WARM
            )
            
            manager.archival_metadata["/logs/missing.log"] = metadata
            
            # Validate integrity
            results = await manager.validate_archive_integrity()
            
            assert results["checked"] == 1
            assert results["valid"] == 0
            assert results["corrupted"] == 0
            assert results["missing"] == 1
            assert len(results["errors"]) == 1
            assert "Missing archive" in results["errors"][0]


class TestRetentionScheduler:
    """Test retention scheduler functionality."""
    
    @pytest.mark.asyncio
    async def test_retention_scheduler_initialization(self):
        """Test retention scheduler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            scheduler = RetentionScheduler(manager)
            
            assert scheduler.retention_manager == manager
            assert scheduler.is_running is False
            assert len(scheduler._tasks) == 0
    
    @pytest.mark.asyncio
    async def test_retention_scheduler_start_stop(self):
        """Test retention scheduler start and stop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            scheduler = RetentionScheduler(manager)
            
            # Start scheduler
            await scheduler.start(process_interval_hours=1, cleanup_interval_hours=2)
            
            assert scheduler.is_running is True
            assert len(scheduler._tasks) == 2  # Process and cleanup tasks
            
            # Stop scheduler
            await scheduler.stop()
            
            assert scheduler.is_running is False
            assert len(scheduler._tasks) == 0
    
    @pytest.mark.asyncio
    async def test_retention_scheduler_force_operations(self):
        """Test forced retention operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            scheduler = RetentionScheduler(manager)
            
            # Test force cleanup
            cleanup_results = await scheduler.force_cleanup()
            assert isinstance(cleanup_results, dict)
            assert "scanned" in cleanup_results
            
            # Test force processing
            processing_results = await scheduler.force_processing()
            assert isinstance(processing_results, dict)
            assert "scanned" in processing_results


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_setup_default_retention(self):
        """Test setup_default_retention function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = await setup_default_retention(log_dir, archive_dir)
            
            assert isinstance(manager, LogRetentionManager)
            assert manager.log_directory == log_dir
            assert manager.archive_directory == archive_dir
            assert len(manager.policies) > 0
    
    @pytest.mark.asyncio
    async def test_archive_logs_by_pattern(self):
        """Test archive_logs_by_pattern function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create test files
            test_file1 = log_dir / "test1.log"
            test_file2 = log_dir / "test2.log"
            test_file3 = log_dir / "other.txt"
            
            test_file1.write_text("Test content 1")
            test_file2.write_text("Test content 2")
            test_file3.write_text("Other content")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Archive by pattern
            archived_count = await archive_logs_by_pattern(
                manager, "*.log", "system_operational"
            )
            
            assert archived_count == 2
            assert not test_file1.exists()
            assert not test_file2.exists()
            assert test_file3.exists()  # Should not be affected
    
    def test_get_compliance_report(self):
        """Test compliance report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Add mock metadata
            now = datetime.now(timezone.utc)
            metadata = ArchivalMetadata(
                original_path="/logs/financial.log",
                archive_path="/archive/financial.log.gz",
                original_size=1000,
                compressed_size=300,
                checksum="test_checksum",
                compression_ratio=0.7,
                archived_at=now,
                retention_policy="financial_transaction",
                storage_tier=StorageTier.WARM,
                expiry_date=now + timedelta(days=30)
            )
            
            manager.archival_metadata["/logs/financial.log"] = metadata
            
            # Generate report
            report = get_compliance_report(manager)
            
            assert "generated_at" in report
            assert report["total_files"] == 1
            assert report["total_size_mb"] > 0
            assert "policies_compliance" in report
            assert "financial_transaction" in report["policies_compliance"]
            assert report["retention_status"] == "compliant"
            assert isinstance(report["recommendations"], list)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_compression_with_invalid_format(self):
        """Test compression with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            test_file = log_dir / "test.log"
            test_file.write_text("Test content")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Try unsupported format
            result = await manager.compress_file(test_file, ArchivalFormat.ZIP)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_archive_file_with_permission_error(self):
        """Test archival with file permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            test_file = log_dir / "test.log"
            test_file.write_text("Test content")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Mock permission error during compression
            with patch.object(manager, 'compress_file', return_value=None):
                result = await manager.archive_file(test_file, "system_operational")
                
                assert result is False
                assert test_file.exists()  # Original should remain
    
    def test_metadata_loading_with_corrupted_file(self):
        """Test metadata loading with corrupted metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            archive_dir.mkdir(parents=True)
            
            # Create corrupted metadata file
            metadata_file = archive_dir / "archival_metadata.json"
            metadata_file.write_text("{invalid json content")
            
            # Should handle corrupted file gracefully
            manager = LogRetentionManager(log_dir, archive_dir, metadata_file=metadata_file)
            
            assert len(manager.archival_metadata) == 0
    
    def test_file_classification_with_unreadable_file(self):
        """Test file classification with unreadable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create empty/unreadable file
            test_file = log_dir / "empty.log"
            test_file.touch()
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Should handle unreadable file gracefully
            classification = manager.classify_log_file(test_file)
            
            assert classification == "system_operational"  # Default classification
    
    @pytest.mark.asyncio
    async def test_cleanup_with_file_deletion_error(self):
        """Test cleanup with file deletion errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Add expired metadata with non-existent file
            now = datetime.now(timezone.utc)
            expired_time = now - timedelta(days=1)
            
            metadata = ArchivalMetadata(
                original_path="/logs/nonexistent.log",
                archive_path="/archive/nonexistent.log.gz",
                original_size=100,
                compressed_size=50,
                checksum="test_checksum",
                compression_ratio=0.5,
                archived_at=expired_time,
                retention_policy="test_policy",
                storage_tier=StorageTier.COLD,
                expiry_date=expired_time
            )
            
            manager.archival_metadata["/logs/nonexistent.log"] = metadata
            
            # Run cleanup - should handle missing file gracefully
            results = await manager.cleanup_expired_files()
            
            assert results["scanned"] == 1
            assert results["expired"] == 1
            assert results["deleted"] == 1  # Should report as deleted even if file missing
            assert results["errors"] == 0


class TestIntegrationWithEnhancedLogging:
    """Test integration with enhanced logging system."""
    
    def test_retention_policies_match_log_categories(self):
        """Test that retention policies cover all log categories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Check that key log categories have corresponding retention policies
            policy_names = set(manager.policies.keys())
            
            # Financial operations should have financial_transaction policy
            assert "financial_transaction" in policy_names
            
            # Security events should have security_event policy
            assert "security_event" in policy_names
            
            # Audit trails should have audit_trail policy
            assert "audit_trail" in policy_names
            
            # System operations should have system_operational policy
            assert "system_operational" in policy_names
            
            # ML/algorithmic decisions should have algorithmic_decision policy
            assert "algorithmic_decision" in policy_names
    
    def test_compliance_metadata_alignment(self):
        """Test that retention policies include proper compliance metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Check financial transaction policy compliance
            financial_policy = manager.get_policy("financial_transaction")
            assert financial_policy is not None
            assert financial_policy.encryption_required is True
            assert "SOX" in financial_policy.compliance_metadata.get("regulation", "")
            assert financial_policy.get_retention_days() == 365 * 7  # 7 years
            
            # Check security event policy compliance
            security_policy = manager.get_policy("security_event")
            assert security_policy is not None
            assert security_policy.encryption_required is True
            assert "SOC 2" in security_policy.compliance_metadata.get("regulation", "")
            assert security_policy.get_retention_days() == 365 * 3  # 3 years
            
            # Check audit trail policy compliance
            audit_policy = manager.get_policy("audit_trail")
            assert audit_policy is not None
            assert audit_policy.encryption_required is True
            assert "GDPR" in audit_policy.compliance_metadata.get("regulation", "")
            assert audit_policy.get_retention_days() == 365 * 10  # 10 years


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""
    
    @pytest.mark.asyncio
    async def test_large_batch_archival_performance(self):
        """Test performance with large batches of files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            log_dir.mkdir(parents=True)
            
            # Create many small log files
            file_count = 50
            for i in range(file_count):
                log_file = log_dir / f"log_{i:03d}.log"
                log_file.write_text(f"Log entry {i}")
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Measure processing time
            start_time = datetime.now()
            results = await manager.process_log_directory()
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Should complete within reasonable time (less than 5 seconds for 50 files)
            assert processing_time < 5.0
            assert results["scanned"] == file_count
            assert results["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_metadata_handling_with_many_files(self):
        """Test metadata handling with large numbers of archived files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            archive_dir = Path(temp_dir) / "archive"
            
            manager = LogRetentionManager(log_dir, archive_dir)
            
            # Add many metadata entries
            now = datetime.now(timezone.utc)
            entry_count = 1000
            
            for i in range(entry_count):
                metadata = ArchivalMetadata(
                    original_path=f"/logs/file_{i:04d}.log",
                    archive_path=f"/archive/file_{i:04d}.log.gz",
                    original_size=1000,
                    compressed_size=300,
                    checksum=f"checksum_{i}",
                    compression_ratio=0.7,
                    archived_at=now,
                    retention_policy="system_operational",
                    storage_tier=StorageTier.WARM
                )
                
                manager.archival_metadata[f"/logs/file_{i:04d}.log"] = metadata
            
            # Test metadata operations
            start_time = datetime.now()
            stats = manager.get_retention_stats()
            end_time = datetime.now()
            
            stats_time = (end_time - start_time).total_seconds()
            
            # Should complete quickly even with many entries
            assert stats_time < 1.0
            assert stats["total_archived_files"] == entry_count