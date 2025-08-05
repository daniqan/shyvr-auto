"""
Tests for Production Safety Checklists System - Phase 3.2.4

This module tests comprehensive production safety checklists including:
- Pre-deployment validation checklists
- Runtime safety verification
- Configuration validation
- Dependency health checks
- Automated checklist execution with detailed reports

Following TDD methodology - tests drive implementation.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4, UUID
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from src.safety.production_safety_checklists import (
    ProductionSafetyChecklistSystem,
    SafetyChecklist,
    ChecklistItem,
    CheckResult,
    ChecklistExecution,
    ChecklistConfig,
    ChecklistType,
    CheckSeverity,
    CheckStatus,
    PreDeploymentValidator,
    RuntimeSafetyValidator,
    ConfigurationValidator,
    DependencyHealthValidator,
    ChecklistReport,
    ChecklistReportSummary,
    ChecklistValidationError
)


class TestProductionSafetyChecklistSystem:
    """Test suite for production safety checklist system."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio for testing."""
        portfolio = Mock()
        portfolio.get_total_balance.return_value = Decimal('100000.00')
        portfolio.get_positions.return_value = {}
        portfolio.is_healthy.return_value = True
        return portfolio

    @pytest.fixture
    def checklist_config(self):
        """Create test checklist configuration."""
        return ChecklistConfig(
            enable_pre_deployment_checks=True,
            enable_runtime_checks=True,
            enable_config_validation=True,
            enable_dependency_checks=True,
            max_execution_time_seconds=300,
            retry_failed_checks=True,
            max_check_retries=2,
            parallel_execution=True,
            generate_reports=True,
            report_retention_days=30,
            critical_check_timeout=60,
            warning_threshold_score=0.8,
            failure_threshold_score=0.6
        )

    @pytest.fixture
    def checklist_system(self, checklist_config, mock_portfolio):
        """Create production safety checklist system for testing."""
        return ProductionSafetyChecklistSystem(
            config=checklist_config,
            portfolio=mock_portfolio
        )

    @pytest.mark.asyncio
    async def test_system_initialization(self, checklist_system):
        """Test system initialization."""
        await checklist_system.initialize()
        
        assert checklist_system.is_initialized
        assert len(checklist_system.validators) > 0
        assert checklist_system.pre_deployment_validator is not None
        assert checklist_system.runtime_validator is not None
        assert checklist_system.config_validator is not None
        assert checklist_system.dependency_validator is not None

    @pytest.mark.asyncio
    async def test_create_pre_deployment_checklist(self, checklist_system):
        """Test creation of pre-deployment checklist."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_pre_deployment_checklist()
        
        assert checklist.checklist_type == ChecklistType.PRE_DEPLOYMENT
        assert len(checklist.items) > 0
        assert any(item.check_name == 'database_connectivity' for item in checklist.items)
        assert any(item.check_name == 'configuration_validity' for item in checklist.items)
        assert any(item.check_name == 'dependency_availability' for item in checklist.items)
        assert any(item.check_name == 'security_compliance' for item in checklist.items)

    @pytest.mark.asyncio
    async def test_create_runtime_safety_checklist(self, checklist_system):
        """Test creation of runtime safety checklist."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_runtime_safety_checklist()
        
        assert checklist.checklist_type == ChecklistType.RUNTIME_SAFETY
        assert len(checklist.items) > 0
        assert any(item.check_name == 'emergency_stop_ready' for item in checklist.items)
        assert any(item.check_name == 'risk_limits_configured' for item in checklist.items)
        assert any(item.check_name == 'circuit_breakers_active' for item in checklist.items)
        assert any(item.check_name == 'position_monitoring_active' for item in checklist.items)

    @pytest.mark.asyncio
    async def test_execute_checklist_success(self, checklist_system):
        """Test successful checklist execution."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_pre_deployment_checklist()
        
        # Mock all validators to return success
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="mock_check",
                    status=CheckStatus.PASSED,
                    score=1.0,
                    message="Mock check passed",
                    severity=CheckSeverity.CRITICAL,
                    execution_time_ms=100,
                    details={}
                )
            ])
        
        execution = await checklist_system.execute_checklist(checklist)
        
        assert execution.execution_id is not None
        assert execution.checklist_id == checklist.checklist_id
        assert execution.status == CheckStatus.PASSED
        assert execution.overall_score >= 0.8
        assert execution.completed_at is not None
        assert len(execution.check_results) > 0

    @pytest.mark.asyncio
    async def test_execute_checklist_with_failures(self, checklist_system):
        """Test checklist execution with some failures."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_runtime_safety_checklist()
        
        # Mock validators with mixed results
        success_result = CheckResult(
            check_name="success_check",
            status=CheckStatus.PASSED,
            score=1.0,
            message="Check passed",
            severity=CheckSeverity.MEDIUM,
            execution_time_ms=50
        )
        
        failure_result = CheckResult(
            check_name="failure_check",
            status=CheckStatus.FAILED,
            score=0.0,
            message="Check failed",
            severity=CheckSeverity.CRITICAL,
            execution_time_ms=75,
            error="Mock failure"
        )
        
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[success_result, failure_result])
        
        execution = await checklist_system.execute_checklist(checklist)
        
        assert execution.status == CheckStatus.FAILED
        assert execution.overall_score < 1.0
        assert len(execution.failed_checks) > 0
        assert any(result.status == CheckStatus.FAILED for result in execution.check_results)

    @pytest.mark.asyncio
    async def test_execute_checklist_with_timeout(self, checklist_system):
        """Test checklist execution with timeout."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_pre_deployment_checklist()
        
        # Mock validator to simulate timeout
        async def slow_check():
            await asyncio.sleep(2)  # Longer than timeout
            return []
        
        for validator in checklist_system.validators.values():
            validator.execute_checks = slow_check
        
        # Set short timeout for test
        checklist_system.config.max_execution_time_seconds = 1
        
        execution = await checklist_system.execute_checklist(checklist)
        
        assert execution.status == CheckStatus.FAILED
        assert "timeout" in execution.error_message.lower()

    @pytest.mark.asyncio
    async def test_retry_failed_checks(self, checklist_system):
        """Test retry mechanism for failed checks."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_runtime_safety_checklist()
        
        # Mock validator that fails first time, succeeds on retry
        call_count = 0
        
        async def flaky_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [CheckResult(
                    check_name="flaky_check",
                    status=CheckStatus.FAILED,
                    score=0.0,
                    message="Temporary failure",
                    severity=CheckSeverity.HIGH,
                    execution_time_ms=100,
                    error="Flaky error"
                )]
            else:
                return [CheckResult(
                    check_name="flaky_check", 
                    status=CheckStatus.PASSED,
                    score=1.0,
                    message="Check passed on retry",
                    severity=CheckSeverity.HIGH,
                    execution_time_ms=50
                )]
        
        for validator in checklist_system.validators.values():
            validator.execute_checks = flaky_check
        
        execution = await checklist_system.execute_checklist(checklist)
        
        assert call_count >= 2  # Original + retry
        assert execution.retry_count > 0

    @pytest.mark.asyncio
    async def test_generate_checklist_report(self, checklist_system):
        """Test checklist report generation."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_pre_deployment_checklist()
        
        # Mock successful execution
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="test_check",
                    status=CheckStatus.PASSED,
                    score=0.9,
                    message="Test passed",
                    severity=CheckSeverity.MEDIUM,
                    execution_time_ms=100
                )
            ])
        
        execution = await checklist_system.execute_checklist(checklist)
        report = await checklist_system.generate_report(execution)
        
        assert report.execution_id == execution.execution_id
        assert report.checklist_type == checklist.checklist_type
        assert report.summary.total_checks > 0
        assert report.summary.passed_checks > 0
        assert report.summary.overall_score > 0
        assert len(report.check_details) > 0
        assert report.generated_at is not None

    @pytest.mark.asyncio
    async def test_validate_production_readiness(self, checklist_system):
        """Test production readiness validation."""
        await checklist_system.initialize()
        
        # Mock all validators to return positive results
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="readiness_check",
                    status=CheckStatus.PASSED,
                    score=0.95,
                    message="System ready",
                    severity=CheckSeverity.CRITICAL,
                    execution_time_ms=50
                )
            ])
        
        readiness_result = await checklist_system.validate_production_readiness()
        
        assert readiness_result['ready'] is True
        assert readiness_result['overall_score'] >= 0.8
        assert len(readiness_result['executed_checklists']) > 0
        assert 'pre_deployment' in readiness_result['executed_checklists']
        assert 'runtime_safety' in readiness_result['executed_checklists']

    @pytest.mark.asyncio
    async def test_get_checklist_history(self, checklist_system):
        """Test retrieval of checklist execution history."""
        await checklist_system.initialize()
        
        # Execute multiple checklists
        checklist1 = await checklist_system.create_pre_deployment_checklist()
        checklist2 = await checklist_system.create_runtime_safety_checklist()
        
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="history_check",
                    status=CheckStatus.PASSED, 
                    score=1.0,
                    message="Test check",
                    severity=CheckSeverity.LOW,
                    execution_time_ms=25
                )
            ])
        
        execution1 = await checklist_system.execute_checklist(checklist1)
        execution2 = await checklist_system.execute_checklist(checklist2)
        
        history = await checklist_system.get_execution_history(limit=10)
        
        assert len(history) >= 2
        assert any(h['checklist_type'] == ChecklistType.PRE_DEPLOYMENT.value for h in history)
        assert any(h['checklist_type'] == ChecklistType.RUNTIME_SAFETY.value for h in history)

    @pytest.mark.asyncio
    async def test_checklist_persistence(self, checklist_system):
        """Test checklist execution persistence."""
        await checklist_system.initialize()
        
        checklist = await checklist_system.create_pre_deployment_checklist()
        
        for validator in checklist_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="persistence_check",
                    status=CheckStatus.PASSED,
                    score=0.8,
                    message="Persistence test",
                    severity=CheckSeverity.MEDIUM,
                    execution_time_ms=80
                )
            ])
        
        execution = await checklist_system.execute_checklist(checklist)
        
        # Verify execution was saved
        saved_execution = await checklist_system.get_execution_by_id(execution.execution_id)
        
        assert saved_execution is not None
        assert saved_execution.execution_id == execution.execution_id
        assert saved_execution.checklist_id == execution.checklist_id
        assert saved_execution.status == execution.status

    @pytest.mark.asyncio
    async def test_checklist_metrics_collection(self, checklist_system):
        """Test collection of checklist execution metrics."""
        await checklist_system.initialize()
        
        # Execute several checklists to generate metrics
        for _ in range(3):
            checklist = await checklist_system.create_runtime_safety_checklist()
            
            for validator in checklist_system.validators.values():
                validator.execute_checks = AsyncMock(return_value=[
                    CheckResult(
                        check_name="metrics_check",
                        status=CheckStatus.PASSED,
                        score=0.85,
                        message="Metrics test",
                        severity=CheckSeverity.HIGH,
                        execution_time_ms=120
                    )
                ])
            
            await checklist_system.execute_checklist(checklist)
        
        metrics = await checklist_system.get_system_metrics()
        
        assert metrics['total_executions'] >= 3
        assert metrics['average_execution_time'] > 0
        assert metrics['success_rate'] > 0
        assert 'checklist_type_distribution' in metrics
        assert 'recent_trend' in metrics


class TestPreDeploymentValidator:
    """Test suite for pre-deployment validator."""

    @pytest.fixture
    def validator(self):
        """Create pre-deployment validator for testing."""
        return PreDeploymentValidator()

    @pytest.mark.asyncio
    async def test_database_connectivity_check(self, validator):
        """Test database connectivity check."""
        checks = await validator.execute_checks()
        
        db_check = next((c for c in checks if c.check_name == 'database_connectivity'), None)
        assert db_check is not None
        assert db_check.severity == CheckSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_configuration_validity_check(self, validator):
        """Test configuration validity check."""
        checks = await validator.execute_checks()
        
        config_check = next((c for c in checks if c.check_name == 'configuration_validity'), None)
        assert config_check is not None
        assert config_check.severity == CheckSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_security_compliance_check(self, validator):
        """Test security compliance check."""
        checks = await validator.execute_checks()
        
        security_check = next((c for c in checks if c.check_name == 'security_compliance'), None)
        assert security_check is not None
        assert security_check.severity == CheckSeverity.HIGH


class TestRuntimeSafetyValidator:
    """Test suite for runtime safety validator."""

    @pytest.fixture
    def validator(self):
        """Create runtime safety validator for testing."""
        return RuntimeSafetyValidator()

    @pytest.mark.asyncio
    async def test_emergency_stop_ready_check(self, validator):
        """Test emergency stop readiness check."""
        checks = await validator.execute_checks()
        
        emergency_check = next((c for c in checks if c.check_name == 'emergency_stop_ready'), None)
        assert emergency_check is not None
        assert emergency_check.severity == CheckSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_risk_limits_configured_check(self, validator):
        """Test risk limits configuration check."""
        checks = await validator.execute_checks()
        
        risk_check = next((c for c in checks if c.check_name == 'risk_limits_configured'), None)
        assert risk_check is not None
        assert risk_check.severity == CheckSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_circuit_breakers_active_check(self, validator):
        """Test circuit breakers active check."""
        checks = await validator.execute_checks()
        
        circuit_check = next((c for c in checks if c.check_name == 'circuit_breakers_active'), None)
        assert circuit_check is not None
        assert circuit_check.severity == CheckSeverity.HIGH


class TestConfigurationValidator:
    """Test suite for configuration validator."""

    @pytest.fixture
    def validator(self):
        """Create configuration validator for testing."""
        return ConfigurationValidator()

    @pytest.mark.asyncio
    async def test_environment_variables_check(self, validator):
        """Test environment variables check."""
        checks = await validator.execute_checks()
        
        env_check = next((c for c in checks if c.check_name == 'environment_variables'), None)
        assert env_check is not None

    @pytest.mark.asyncio
    async def test_api_keys_configured_check(self, validator):
        """Test API keys configuration check."""
        checks = await validator.execute_checks()
        
        api_check = next((c for c in checks if c.check_name == 'api_keys_configured'), None)
        assert api_check is not None
        assert api_check.severity == CheckSeverity.CRITICAL


class TestDependencyHealthValidator:
    """Test suite for dependency health validator."""

    @pytest.fixture
    def validator(self):
        """Create dependency health validator for testing."""
        return DependencyHealthValidator()

    @pytest.mark.asyncio
    async def test_external_services_check(self, validator):
        """Test external services health check."""
        checks = await validator.execute_checks()
        
        services_check = next((c for c in checks if c.check_name == 'external_services'), None)
        assert services_check is not None
        assert services_check.severity == CheckSeverity.HIGH

    @pytest.mark.asyncio
    async def test_system_resources_check(self, validator):
        """Test system resources check."""
        checks = await validator.execute_checks()
        
        resources_check = next((c for c in checks if c.check_name == 'system_resources'), None)
        assert resources_check is not None
        assert resources_check.severity == CheckSeverity.MEDIUM


class TestChecklistIntegration:
    """Integration tests for the complete checklist system."""

    @pytest.fixture
    def mock_integrations(self):
        """Mock external integrations."""
        return {
            'safety_state_manager': Mock(),
            'circuit_breaker': Mock(),
            'emergency_controller': Mock(),
            'monitoring_system': Mock()
        }

    @pytest.fixture
    def integrated_system(self, checklist_config, mock_portfolio, mock_integrations):
        """Create integrated checklist system with mocked dependencies."""
        system = ProductionSafetyChecklistSystem(
            config=checklist_config,
            portfolio=mock_portfolio,
            integrations=mock_integrations
        )
        return system

    @pytest.mark.asyncio
    async def test_full_production_readiness_workflow(self, integrated_system):
        """Test complete production readiness validation workflow."""
        await integrated_system.initialize()
        
        # Mock positive results from all integrations
        for validator in integrated_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=[
                CheckResult(
                    check_name="integration_check",
                    status=CheckStatus.PASSED,
                    score=0.9,
                    message="Integration healthy",
                    severity=CheckSeverity.HIGH,
                    execution_time_ms=100
                )
            ])
        
        # Execute full readiness validation
        readiness = await integrated_system.validate_production_readiness()
        
        assert readiness['ready'] is True
        assert readiness['overall_score'] >= 0.8
        
        # Verify all checklist types were executed
        executed_types = readiness['executed_checklists'].keys()
        assert 'pre_deployment' in executed_types
        assert 'runtime_safety' in executed_types
        assert 'configuration' in executed_types
        assert 'dependency_health' in executed_types

    @pytest.mark.asyncio
    async def test_checklist_failure_escalation(self, integrated_system):
        """Test failure escalation and notification workflow."""
        await integrated_system.initialize()
        
        # Mock critical failure in one validator
        critical_failure = CheckResult(
            check_name="critical_security_check",
            status=CheckStatus.FAILED,
            score=0.0,
            message="Critical security vulnerability detected",
            severity=CheckSeverity.CRITICAL,
            execution_time_ms=50,
            error="Security compliance failure"
        )
        
        integrated_system.validators['pre_deployment'].execute_checks = AsyncMock(
            return_value=[critical_failure]
        )
        
        # Other validators return success
        for name, validator in integrated_system.validators.items():
            if name != 'pre_deployment':
                validator.execute_checks = AsyncMock(return_value=[
                    CheckResult(
                        check_name="ok_check",
                        status=CheckStatus.PASSED,
                        score=1.0,
                        message="Check passed",
                        severity=CheckSeverity.MEDIUM,
                        execution_time_ms=25
                    )
                ])
        
        readiness = await integrated_system.validate_production_readiness()
        
        assert readiness['ready'] is False
        assert len(readiness['critical_failures']) > 0
        assert readiness['requires_immediate_attention'] is True

    @pytest.mark.asyncio
    async def test_checklist_report_generation_with_recommendations(self, integrated_system):
        """Test comprehensive report generation with recommendations."""
        await integrated_system.initialize()
        
        # Mix of results to generate meaningful recommendations
        mixed_results = [
            CheckResult(
                check_name="database_performance",
                status=CheckStatus.WARNING,
                score=0.7,
                message="Database performance degraded",
                severity=CheckSeverity.HIGH,
                execution_time_ms=200,
                recommendations=["Optimize database queries", "Add connection pooling"]
            ),
            CheckResult(
                check_name="memory_usage",
                status=CheckStatus.PASSED,
                score=0.85,
                message="Memory usage within limits",
                severity=CheckSeverity.MEDIUM,
                execution_time_ms=50
            )
        ]
        
        for validator in integrated_system.validators.values():
            validator.execute_checks = AsyncMock(return_value=mixed_results)
        
        checklist = await integrated_system.create_pre_deployment_checklist()
        execution = await integrated_system.execute_checklist(checklist)
        report = await integrated_system.generate_report(execution)
        
        assert len(report.recommendations) > 0
        assert any("database" in rec.lower() for rec in report.recommendations)
        assert report.summary.warning_checks > 0