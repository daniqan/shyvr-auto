"""
Test cases for Unified Safety State Manager - Phase 3.2.4

This module contains comprehensive TDD tests for the unified safety state management system
that persists safety states across modes, system restarts, and provides rollback capabilities.

Following strict TDD methodology - these tests MUST FAIL initially and drive implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

# TDD - Import should fail initially as module doesn't exist yet
try:
    from src.safety.unified_safety_state_manager import (
        UnifiedSafetyStateManager,
        SafetyState,
        SafetyStateSnapshot,
        SafetyStateConfig,
        SafetyStateTransaction,
        SafetyStateRollback,
        SafetyStatePersistence,
        SafetyStateValidator,
        SafetyStateError,
        SafetyStateConsistencyError,
        SafetyStateRollbackError
    )
    MODULE_EXISTS = True
except ImportError:
    MODULE_EXISTS = False
    # Create placeholder classes for type hints during TDD phase
    class UnifiedSafetyStateManager: pass
    class SafetyState: pass
    class SafetyStateSnapshot: pass
    class SafetyStateConfig: pass
    class SafetyStateTransaction: pass
    class SafetyStateRollback: pass
    class SafetyStatePersistence: pass
    class SafetyStateValidator: pass
    class SafetyStateError(Exception): pass
    class SafetyStateConsistencyError(Exception): pass
    class SafetyStateRollbackError(Exception): pass
from src.modes.base import ModeType, ModeStatus
from src.portfolio.base import Portfolio


class TestUnifiedSafetyStateManager:
    """Test cases for UnifiedSafetyStateManager following TDD methodology."""
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = MagicMock(spec=Portfolio)
        portfolio.total_value = Decimal("100000.00")
        portfolio.balance = Decimal("10000.00")
        portfolio.unrealized_pnl = Decimal("-500.00")
        portfolio.positions = {}
        return portfolio
    
    @pytest.fixture
    def safety_state_config(self):
        """Create safety state configuration for testing."""
        return SafetyStateConfig(
            persistence_enabled=True,
            auto_backup_interval=300,  # 5 minutes
            max_rollback_states=10,
            consistency_check_interval=60,
            enable_audit_trail=True,
            state_compression=True,
            max_transaction_size=1000
        )
    
    @pytest.fixture
    async def safety_state_manager(self, safety_state_config, mock_portfolio):
        """Create UnifiedSafetyStateManager instance for testing."""
        manager = UnifiedSafetyStateManager(
            config=safety_state_config,
            portfolio=mock_portfolio
        )
        await manager.initialize()
        return manager

    # Test: Unified Safety State Creation and Management
    
    async def test_create_unified_safety_state_fails_initially(self, safety_state_config, mock_portfolio):
        """Test that creating unified safety state fails initially (TDD - failing test)."""
        if MODULE_EXISTS:
            pytest.skip("Module already exists - TDD phase complete")
        
        # This should fail as the module doesn't exist yet
        assert not MODULE_EXISTS, "UnifiedSafetyStateManager module should not exist during TDD phase"
    
    async def test_safety_state_initialization_comprehensive(self, safety_state_manager):
        """Test comprehensive safety state initialization."""
        if not MODULE_EXISTS:
            pytest.skip("Module doesn't exist yet - implement first")
            
        # Should initialize with default safety state
        assert safety_state_manager.is_initialized
        assert safety_state_manager.current_state is not None
        assert safety_state_manager.current_state.state_id is not None
        assert safety_state_manager.current_state.emergency_stop_active == False
        assert safety_state_manager.current_state.risk_level >= 0.0
        assert safety_state_manager.current_state.active_modes == set()
        
        # Should have persistence layer initialized
        assert safety_state_manager.persistence is not None
        assert safety_state_manager.persistence.is_connected()
        
        # Should have validator initialized
        assert safety_state_manager.validator is not None
        assert safety_state_manager.validator.is_ready()

    # Test: Safety State Persistence Across System Restarts
    
    async def test_safety_state_persistence_save_and_load(self, safety_state_manager):
        """Test safety state persistence across system restarts."""
        # Create a complex safety state
        original_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            emergency_stop_active=True,
            emergency_stop_reason="test_reason",
            risk_level=0.75,
            active_modes={ModeType.LIVE_TRADING, ModeType.ANALYSIS},
            mode_risk_levels={
                ModeType.LIVE_TRADING: 0.8,
                ModeType.ANALYSIS: 0.3
            },
            position_limits={
                "max_position_size": Decimal("5000.00"),
                "max_total_exposure": Decimal("80000.00")
            },
            safety_thresholds={
                "max_drawdown": 0.15,
                "daily_loss_limit": 0.05
            },
            active_alerts=["high_risk_alert", "position_limit_warning"],
            system_health_score=0.85,
            metadata={"last_update_source": "emergency_trigger"}
        )
        
        # Save state
        save_result = await safety_state_manager.save_state(original_state)
        assert save_result.success
        assert save_result.persistence_id is not None
        
        # Simulate system restart - create new manager
        new_manager = UnifiedSafetyStateManager(
            config=safety_state_manager.config,
            portfolio=safety_state_manager.portfolio
        )
        await new_manager.initialize()
        
        # Load state should restore the exact same state
        loaded_state = await new_manager.load_latest_state()
        assert loaded_state is not None
        assert loaded_state.state_id == original_state.state_id
        assert loaded_state.emergency_stop_active == original_state.emergency_stop_active
        assert loaded_state.emergency_stop_reason == original_state.emergency_stop_reason
        assert loaded_state.risk_level == original_state.risk_level
        assert loaded_state.active_modes == original_state.active_modes
        assert loaded_state.position_limits == original_state.position_limits
        assert loaded_state.safety_thresholds == original_state.safety_thresholds
        assert loaded_state.active_alerts == original_state.active_alerts
        assert loaded_state.system_health_score == original_state.system_health_score

    async def test_safety_state_persistence_multiple_versions(self, safety_state_manager):
        """Test persistence of multiple safety state versions."""
        states = []
        
        # Create and save multiple state versions
        for i in range(5):
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                risk_level=0.1 * (i + 1),
                system_health_score=0.9 - (0.1 * i),
                metadata={"version": i}
            )
            
            save_result = await safety_state_manager.save_state(state)
            assert save_result.success
            states.append(state)
        
        # Should be able to load any version
        for i, original_state in enumerate(states):
            loaded_state = await safety_state_manager.load_state_by_id(original_state.state_id)
            assert loaded_state is not None
            assert loaded_state.state_id == original_state.state_id
            assert loaded_state.metadata["version"] == i
        
        # Latest state should be the last one
        latest_state = await safety_state_manager.load_latest_state()
        assert latest_state.state_id == states[-1].state_id
        assert latest_state.metadata["version"] == 4

    # Test: Safety State Consistency and Validation
    
    async def test_safety_state_consistency_validation(self, safety_state_manager):
        """Test safety state consistency validation."""
        # Create inconsistent state (should fail validation)
        inconsistent_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            risk_level=-0.5,  # Invalid: negative risk level
            active_modes={ModeType.LIVE_TRADING},
            mode_risk_levels={},  # Inconsistent: active mode without risk level
            position_limits={
                "max_position_size": Decimal("-1000.00")  # Invalid: negative limit
            }
        )
        
        # Validation should fail
        validation_result = await safety_state_manager.validate_state(inconsistent_state)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        assert "negative_risk_level" in [error.error_type for error in validation_result.errors]
        assert "missing_mode_risk_levels" in [error.error_type for error in validation_result.errors]
        assert "negative_position_limit" in [error.error_type for error in validation_result.errors]
        
        # Save should fail for invalid state
        with pytest.raises(SafetyStateConsistencyError):
            await safety_state_manager.save_state(inconsistent_state)

    async def test_safety_state_cross_mode_consistency(self, safety_state_manager):
        """Test safety state consistency across multiple modes."""
        # Create state with multiple modes
        multi_mode_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            active_modes={ModeType.LIVE_TRADING, ModeType.SIMULATION, ModeType.ANALYSIS},
            mode_risk_levels={
                ModeType.LIVE_TRADING: 0.8,
                ModeType.SIMULATION: 0.4,
                ModeType.ANALYSIS: 0.2
            },
            mode_positions={
                ModeType.LIVE_TRADING: {"ETH": Decimal("5000.00")},
                ModeType.SIMULATION: {"BTC": Decimal("3000.00")},
                ModeType.ANALYSIS: {}
            }
        )
        
        # Should validate successfully
        validation_result = await safety_state_manager.validate_state(multi_mode_state)
        assert validation_result.is_valid
        
        # Should maintain consistency when mode is removed
        updated_state = await safety_state_manager.remove_mode_from_state(
            multi_mode_state, ModeType.SIMULATION
        )
        
        assert ModeType.SIMULATION not in updated_state.active_modes
        assert ModeType.SIMULATION not in updated_state.mode_risk_levels
        assert ModeType.SIMULATION not in updated_state.mode_positions
        
        # Remaining modes should be intact
        assert ModeType.LIVE_TRADING in updated_state.active_modes
        assert ModeType.ANALYSIS in updated_state.active_modes

    # Test: Safety State Rollback Capabilities
    
    async def test_safety_state_rollback_basic(self, safety_state_manager):
        """Test basic safety state rollback functionality."""
        # Create initial state
        initial_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            risk_level=0.3,
            emergency_stop_active=False,
            metadata={"version": "initial"}
        )
        
        await safety_state_manager.save_state(initial_state)
        
        # Create updated state
        updated_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            risk_level=0.8,
            emergency_stop_active=True,
            emergency_stop_reason="high_risk",
            metadata={"version": "updated"}
        )
        
        await safety_state_manager.save_state(updated_state)
        
        # Current state should be updated
        current_state = await safety_state_manager.get_current_state()
        assert current_state.state_id == updated_state.state_id
        assert current_state.emergency_stop_active == True
        
        # Rollback to initial state
        rollback_result = await safety_state_manager.rollback_to_state(initial_state.state_id)
        assert rollback_result.success
        assert rollback_result.rolled_back_to_state_id == initial_state.state_id
        
        # Current state should now be initial state
        current_state = await safety_state_manager.get_current_state()
        assert current_state.state_id == initial_state.state_id
        assert current_state.emergency_stop_active == False
        assert current_state.metadata["version"] == "initial"

    async def test_safety_state_rollback_transaction_support(self, safety_state_manager):
        """Test rollback with transaction support."""
        # Start transaction
        transaction = await safety_state_manager.begin_transaction()
        assert transaction.transaction_id is not None
        assert transaction.is_active()
        
        # Make multiple state changes within transaction
        states_in_transaction = []
        for i in range(3):
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow(),
                risk_level=0.2 + (0.1 * i),
                metadata={"transaction_step": i}
            )
            
            await safety_state_manager.save_state_in_transaction(transaction, state)
            states_in_transaction.append(state)
        
        # Commit transaction
        commit_result = await safety_state_manager.commit_transaction(transaction)
        assert commit_result.success
        
        # All states should be available
        for state in states_in_transaction:
            loaded_state = await safety_state_manager.load_state_by_id(state.state_id)
            assert loaded_state is not None
        
        # Rollback entire transaction
        rollback_result = await safety_state_manager.rollback_transaction(transaction.transaction_id)
        assert rollback_result.success
        
        # States from transaction should be removed
        for state in states_in_transaction:
            loaded_state = await safety_state_manager.load_state_by_id(state.state_id)
            assert loaded_state is None

    async def test_safety_state_rollback_with_validation(self, safety_state_manager):
        """Test rollback with pre-rollback validation."""
        # Create states with dependencies
        dependent_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            active_modes={ModeType.LIVE_TRADING},
            mode_positions={
                ModeType.LIVE_TRADING: {"BTC": Decimal("10000.00")}
            },
            metadata={"has_active_positions": True}
        )
        
        await safety_state_manager.save_state(dependent_state)
        
        # Create clean state to rollback to
        clean_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow() - timedelta(hours=1),
            active_modes=set(),
            mode_positions={},
            metadata={"has_active_positions": False}
        )
        
        await safety_state_manager.save_state(clean_state)
        
        # Rollback should require validation
        rollback_validation = await safety_state_manager.validate_rollback(
            from_state_id=dependent_state.state_id,
            to_state_id=clean_state.state_id
        )
        
        # Should warn about active positions
        assert not rollback_validation.is_safe
        assert len(rollback_validation.warnings) > 0
        assert "active_positions_will_be_lost" in [w.warning_type for w in rollback_validation.warnings]
        
        # Forced rollback should work but create audit entry
        rollback_result = await safety_state_manager.rollback_to_state(
            clean_state.state_id, 
            force=True,
            reason="test_rollback"
        )
        
        assert rollback_result.success
        assert rollback_result.was_forced == True
        assert rollback_result.audit_entry_id is not None

    # Test: Safety State Snapshots and Backup
    
    async def test_safety_state_snapshot_creation(self, safety_state_manager):
        """Test safety state snapshot creation and management."""
        # Create complex state
        complex_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            emergency_stop_active=True,
            risk_level=0.85,
            active_modes={ModeType.LIVE_TRADING, ModeType.ANALYSIS},
            mode_risk_levels={
                ModeType.LIVE_TRADING: 0.9,
                ModeType.ANALYSIS: 0.4
            },
            position_limits={"max_exposure": Decimal("50000.00")},
            active_alerts=["critical_risk", "emergency_stop"],
            system_health_score=0.6
        )
        
        await safety_state_manager.save_state(complex_state)
        
        # Create snapshot
        snapshot = await safety_state_manager.create_snapshot("pre_rollback_snapshot")
        assert snapshot.snapshot_id is not None
        assert snapshot.name == "pre_rollback_snapshot"
        assert snapshot.state_count > 0
        assert snapshot.creation_timestamp is not None
        assert snapshot.compressed_size > 0
        
        # Snapshot should include current state
        snapshot_states = await safety_state_manager.get_snapshot_states(snapshot.snapshot_id)
        assert len(snapshot_states) > 0
        
        current_state_in_snapshot = next(
            (s for s in snapshot_states if s.state_id == complex_state.state_id), 
            None
        )
        assert current_state_in_snapshot is not None
        assert current_state_in_snapshot.emergency_stop_active == True

    async def test_safety_state_snapshot_restore(self, safety_state_manager):
        """Test safety state snapshot restoration."""
        # Create initial states
        initial_states = []
        for i in range(3):
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                risk_level=0.2 + (0.1 * i),
                metadata={"initial_batch": True, "index": i}
            )
            await safety_state_manager.save_state(state)
            initial_states.append(state)
        
        # Create snapshot
        snapshot = await safety_state_manager.create_snapshot("backup_point")
        
        # Add more states after snapshot
        for i in range(2):
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow() + timedelta(seconds=10 + i),
                risk_level=0.8 + (0.1 * i),
                metadata={"after_snapshot": True, "index": i}
            )
            await safety_state_manager.save_state(state)
        
        # Restore from snapshot
        restore_result = await safety_state_manager.restore_from_snapshot(
            snapshot.snapshot_id,
            cleanup_newer_states=True
        )
        
        assert restore_result.success
        assert restore_result.restored_state_count == len(initial_states)
        
        # States after snapshot should be removed
        all_states = await safety_state_manager.list_all_states()
        after_snapshot_states = [
            s for s in all_states 
            if s.metadata.get("after_snapshot", False)
        ]
        assert len(after_snapshot_states) == 0
        
        # Initial states should still exist
        for initial_state in initial_states:
            loaded_state = await safety_state_manager.load_state_by_id(initial_state.state_id)
            assert loaded_state is not None
            assert loaded_state.metadata["initial_batch"] == True

    # Test: Safety State Event Handling and Triggers
    
    async def test_safety_state_event_triggers(self, safety_state_manager):
        """Test safety state event triggers and handlers."""
        triggered_events = []
        
        # Register event handlers
        async def on_state_change(old_state: SafetyState, new_state: SafetyState):
            triggered_events.append(("state_change", old_state.state_id, new_state.state_id))
        
        async def on_emergency_stop(state: SafetyState, reason: str):
            triggered_events.append(("emergency_stop", state.state_id, reason))
        
        async def on_rollback(from_state_id: UUID, to_state_id: UUID):
            triggered_events.append(("rollback", from_state_id, to_state_id))
        
        safety_state_manager.register_event_handler("state_change", on_state_change)
        safety_state_manager.register_event_handler("emergency_stop", on_emergency_stop)
        safety_state_manager.register_event_handler("rollback", on_rollback)
        
        # Create initial state
        initial_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            emergency_stop_active=False
        )
        await safety_state_manager.save_state(initial_state)
        
        # Trigger emergency stop
        emergency_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            emergency_stop_active=True,
            emergency_stop_reason="test_trigger"
        )
        await safety_state_manager.save_state(emergency_state)
        
        # Trigger rollback
        await safety_state_manager.rollback_to_state(initial_state.state_id)
        
        # Verify events were triggered
        assert len(triggered_events) >= 3
        
        # Should have state_change events
        state_change_events = [e for e in triggered_events if e[0] == "state_change"]
        assert len(state_change_events) >= 2
        
        # Should have emergency_stop event
        emergency_events = [e for e in triggered_events if e[0] == "emergency_stop"]
        assert len(emergency_events) == 1
        assert emergency_events[0][2] == "test_trigger"
        
        # Should have rollback event
        rollback_events = [e for e in triggered_events if e[0] == "rollback"]
        assert len(rollback_events) == 1

    # Test: Safety State Performance and Scalability
    
    async def test_safety_state_performance_large_dataset(self, safety_state_manager):
        """Test safety state performance with large datasets."""
        # Create many states to test performance
        states = []
        batch_size = 100
        
        start_time = datetime.utcnow()
        
        # Create and save states in batches
        for batch in range(3):  # 300 states total
            batch_states = []
            for i in range(batch_size):
                state = SafetyState(
                    state_id=uuid4(),
                    timestamp=datetime.utcnow() + timedelta(seconds=batch * batch_size + i),
                    risk_level=0.1 + (0.001 * (batch * batch_size + i)),
                    metadata={"batch": batch, "index": i}
                )
                batch_states.append(state)
            
            # Batch save should be faster than individual saves
            batch_save_result = await safety_state_manager.save_states_batch(batch_states)
            assert batch_save_result.success
            assert batch_save_result.saved_count == batch_size
            
            states.extend(batch_states)
        
        save_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Performance requirements
        assert save_duration < 10.0  # Should save 300 states in under 10 seconds
        assert len(states) == 300
        
        # Test query performance
        query_start = datetime.utcnow()
        
        # Query by time range
        time_range_states = await safety_state_manager.query_states_by_time_range(
            start_time=start_time,
            end_time=datetime.utcnow()
        )
        
        query_duration = (datetime.utcnow() - query_start).total_seconds()
        
        assert query_duration < 2.0  # Should query in under 2 seconds
        assert len(time_range_states) == 300

    async def test_safety_state_concurrent_access(self, safety_state_manager):
        """Test safety state manager under concurrent access."""
        concurrent_results = []
        
        async def concurrent_state_operation(operation_id: int):
            try:
                # Each coroutine creates and saves a state
                state = SafetyState(
                    state_id=uuid4(),
                    timestamp=datetime.utcnow(),
                    risk_level=0.1 * operation_id,
                    metadata={"operation_id": operation_id}
                )
                
                save_result = await safety_state_manager.save_state(state)
                
                # Verify state can be loaded
                loaded_state = await safety_state_manager.load_state_by_id(state.state_id)
                
                concurrent_results.append({
                    "operation_id": operation_id,
                    "save_success": save_result.success,
                    "load_success": loaded_state is not None,
                    "state_id": state.state_id
                })
                
            except Exception as e:
                concurrent_results.append({
                    "operation_id": operation_id,
                    "error": str(e)
                })
        
        # Run concurrent operations
        concurrent_operations = [
            concurrent_state_operation(i) for i in range(10)
        ]
        
        await asyncio.gather(*concurrent_operations)
        
        # All operations should succeed
        assert len(concurrent_results) == 10
        
        successful_operations = [r for r in concurrent_results if r.get("save_success", False)]
        assert len(successful_operations) == 10
        
        # No data corruption should occur
        for result in successful_operations:
            assert result["load_success"] == True
            assert "error" not in result

    # Test: Safety State Integration with External Systems
    
    async def test_safety_state_gcp_integration(self, safety_state_manager):
        """Test safety state integration with GCP services."""
        # Mock GCP integration
        with patch('src.safety.unified_safety_state_manager.GCPStateStorage') as mock_gcp:
            mock_storage = MagicMock()
            mock_gcp.return_value = mock_storage
            
            # Configure GCP storage
            gcp_config = {
                "project_id": "test-project",
                "bucket_name": "safety-states",
                "encryption_key": "test-key"
            }
            
            await safety_state_manager.configure_gcp_storage(gcp_config)
            
            # Create state with GCP backup
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow(),
                risk_level=0.7,
                metadata={"backup_to_gcp": True}
            )
            
            save_result = await safety_state_manager.save_state(state, backup_to_gcp=True)
            
            assert save_result.success
            assert save_result.gcp_backup_id is not None
            
            # Verify GCP storage was called
            mock_storage.upload_state.assert_called_once()

    async def test_safety_state_monitoring_integration(self, safety_state_manager):
        """Test safety state integration with monitoring systems."""
        monitoring_events = []
        
        # Mock monitoring integration
        with patch('src.safety.unified_safety_state_manager.MonitoringClient') as mock_monitoring:
            mock_client = MagicMock()
            mock_monitoring.return_value = mock_client
            
            # Configure monitoring
            await safety_state_manager.configure_monitoring({
                "enable_metrics": True,
                "metric_prefix": "safety_state",
                "alert_thresholds": {
                    "high_risk_level": 0.8,
                    "emergency_stop": True
                }
            })
            
            # Create high-risk state
            high_risk_state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow(),
                risk_level=0.9,  # Above threshold
                emergency_stop_active=True
            )
            
            await safety_state_manager.save_state(high_risk_state)
            
            # Verify monitoring metrics were sent
            mock_client.send_metric.assert_called()
            mock_client.send_alert.assert_called()

    # Test: Error Handling and Recovery
    
    async def test_safety_state_error_handling(self, safety_state_manager):
        """Test safety state error handling and recovery."""
        # Test handling of persistence failures
        with patch.object(safety_state_manager.persistence, 'save_state', side_effect=Exception("DB Error")):
            state = SafetyState(
                state_id=uuid4(),
                timestamp=datetime.utcnow(),
                risk_level=0.5
            )
            
            with pytest.raises(SafetyStateError) as exc_info:
                await safety_state_manager.save_state(state)
            
            assert "persistence_error" in str(exc_info.value)
        
        # Test handling of validation failures
        invalid_state = SafetyState(
            state_id=uuid4(),
            timestamp=datetime.utcnow(),
            risk_level=1.5  # Invalid
        )
        
        with pytest.raises(SafetyStateConsistencyError):
            await safety_state_manager.save_state(invalid_state)
        
        # Test recovery from corrupted state
        corrupted_state_id = uuid4()
        
        with patch.object(safety_state_manager.persistence, 'load_state', 
                         return_value={"invalid": "data"}):
            loaded_state = await safety_state_manager.load_state_by_id(
                corrupted_state_id, 
                handle_corruption=True
            )
            
            # Should return None for corrupted state
            assert loaded_state is None
            
            # Should log corruption event
            corruption_events = await safety_state_manager.get_corruption_events()
            assert len(corruption_events) > 0
            assert corruption_events[-1]["state_id"] == str(corrupted_state_id)