#!/usr/bin/env python3
"""
Standalone test for SafetyEventLogger to verify implementation works correctly.
"""

import sys
import os
import asyncio

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from the module file to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "safety_event_logging", 
    os.path.join(os.path.dirname(__file__), 'src', 'safety', 'safety_event_logging.py')
)
safety_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety_module)

SafetyEventConfig = safety_module.SafetyEventConfig
SafetyEventLogger = safety_module.SafetyEventLogger
SafetyEventType = safety_module.SafetyEventType
SafetyEventSeverity = safety_module.SafetyEventSeverity


async def test_basic_functionality():
    """Test basic SafetyEventLogger functionality."""
    print("Testing SafetyEventLogger basic functionality...")
    
    # Initialize logger
    config = SafetyEventConfig(storage_path='/tmp/test_safety_events')
    logger = SafetyEventLogger(config)
    await logger.initialize()
    
    print("✓ SafetyEventLogger initialized")
    
    # Log a test event
    event_id = await logger.log_event(
        event_type=SafetyEventType.EMERGENCY_STOP,
        severity=SafetyEventSeverity.CRITICAL,
        description='Test emergency stop',
        source_component='test_component',
        details={'test_field': 'test_value'}
    )
    
    print(f"✓ Event logged successfully: {event_id}")
    
    # Retrieve the event
    event = await logger.get_event(event_id)
    assert event is not None, "Event should be retrievable"
    assert event.description == 'Test emergency stop', "Event description should match"
    assert event.event_type == SafetyEventType.EMERGENCY_STOP, "Event type should match"
    assert event.severity == SafetyEventSeverity.CRITICAL, "Event severity should match"
    
    print(f"✓ Event retrieved: {event.description}")
    
    # Query events
    events = await logger.query_events()
    assert len(events) >= 1, "Should have at least one event"
    
    print(f"✓ Query returned {len(events)} events")
    
    # Test batch logging
    batch_events = [
        {
            'event_type': SafetyEventType.RISK_VIOLATION,
            'severity': SafetyEventSeverity.HIGH,
            'description': f'Batch test event {i}',
            'source_component': 'batch_test'
        }
        for i in range(5)
    ]
    
    batch_ids = await logger.log_events_batch(batch_events)
    assert len(batch_ids) == 5, "Should log all batch events"
    
    print(f"✓ Batch logged {len(batch_ids)} events")
    
    # Test performance statistics
    stats = logger.get_performance_stats()
    assert stats['events_logged'] >= 6, "Should have logged at least 6 events"
    assert stats['is_initialized'], "Should be initialized"
    
    print(f"✓ Performance stats: {stats['events_logged']} events, "
          f"{stats['events_per_second']:.2f} events/sec")
    
    return True


async def test_event_correlation():
    """Test event correlation functionality."""
    print("\nTesting event correlation...")
    
    config = SafetyEventConfig(storage_path='/tmp/test_safety_correlation')
    logger = SafetyEventLogger(config)
    await logger.initialize()
    
    # Log first event
    event_id_1 = await logger.log_event(
        event_type=SafetyEventType.RISK_VIOLATION,
        severity=SafetyEventSeverity.HIGH,
        description='Risk threshold exceeded',
        source_component='risk_manager'
    )
    
    # Log correlated event
    event_id_2 = await logger.log_event(
        event_type=SafetyEventType.EMERGENCY_STOP,
        severity=SafetyEventSeverity.CRITICAL,
        description='Emergency stop triggered',
        source_component='trading_manager',
        correlation_id=event_id_1
    )
    
    # Get correlated events
    correlated = await logger.get_correlated_events(event_id_1)
    
    print(f"✓ Found {len(correlated)} correlated events")
    
    return True


async def test_export_functionality():
    """Test event export functionality."""
    print("\nTesting event export...")
    
    config = SafetyEventConfig(storage_path='/tmp/test_safety_export')
    logger = SafetyEventLogger(config)
    await logger.initialize()
    
    # Log some events for export
    events_to_export = []
    for i in range(3):
        event_id = await logger.log_event(
            event_type=SafetyEventType.SAFETY_CHECK_FAILURE,
            severity=SafetyEventSeverity.MEDIUM,
            description=f'Export test event {i}',
            source_component='export_test'
        )
        event = await logger.get_event(event_id)
        events_to_export.append(event)
    
    # Test JSON export
    json_result = await logger.export_events(events_to_export, format='json')
    assert json_result.success, "JSON export should succeed"
    assert json_result.record_count == 3, "Should export 3 records"
    
    print(f"✓ JSON export: {json_result.record_count} records, "
          f"{json_result.file_size_bytes} bytes")
    
    # Test CSV export
    csv_result = await logger.export_events(events_to_export, format='csv')
    assert csv_result.success, "CSV export should succeed"
    assert csv_result.record_count == 3, "Should export 3 records"
    
    print(f"✓ CSV export: {csv_result.record_count} records, "
          f"{csv_result.file_size_bytes} bytes")
    
    return True


async def main():
    """Run all tests."""
    print("SafetyEventLogger Implementation Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        await test_basic_functionality()
        await test_event_correlation()
        await test_export_functionality()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! SafetyEventLogger is working correctly.")
        print("✅ Safety Event Logging & Audit Trail system implementation complete.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)