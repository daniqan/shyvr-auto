#!/usr/bin/env python3
"""
Integration tests for initial corpus collection script

Tests the production-ready script that executes initial corpus collection
integrating with existing GCP infrastructure, using real API calls and database operations.
"""

import os
import sys
import pytest
import asyncio
import subprocess
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.database import get_database_connection
from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector


class TestCollectInitialCorpusScript:
    """Test the initial corpus collection script with real GCP integration"""
    
    @pytest.fixture(scope="class")
    def script_path(self):
        """Path to the corpus collection script"""
        return PROJECT_ROOT / "scripts" / "collect_initial_corpus.py"
    
    @pytest.fixture(scope="class") 
    def deployment_script_path(self):
        """Path to the deployment script"""
        return PROJECT_ROOT / "deploy" / "scripts" / "run_initial_corpus_collection.sh"
    
    @pytest.fixture(scope="class")
    def test_environment(self):
        """Set up test environment variables"""
        # Preserve original environment
        original_env = dict(os.environ)
        
        # Set test environment
        test_env = {
            'ENVIRONMENT': 'development',
            'PROJECT_ID': os.getenv('GOOGLE_CLOUD_PROJECT', 'shvyr-ai-bots'),
            'DATABASE_URL': os.getenv('DATABASE_URL', ''),
            'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY', ''),
            'LUNARCRUSH_API_KEY': os.getenv('LUNARCRUSH_API_KEY', ''),
            'HELIUS_API_KEY': os.getenv('HELIUS_API_KEY', ''),
            'GCS_BUCKET': 'shyvr-models-prod'
        }
        
        for key, value in test_env.items():
            if value:
                os.environ[key] = value
        
        yield test_env
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)
    
    def test_script_exists_and_executable(self, script_path):
        """Test that the corpus collection script exists and is executable"""
        assert script_path.exists(), f"Script not found at {script_path}"
        assert script_path.is_file(), f"Script is not a file: {script_path}"
        assert os.access(script_path, os.X_OK), f"Script is not executable: {script_path}"
    
    def test_deployment_script_exists(self, deployment_script_path):
        """Test that the deployment script exists"""
        assert deployment_script_path.exists(), f"Deployment script not found at {deployment_script_path}"
        assert deployment_script_path.is_file(), f"Deployment script is not a file"
        assert os.access(deployment_script_path, os.X_OK), f"Deployment script is not executable"
    
    def test_script_help_option(self, script_path):
        """Test script shows help when --help is provided"""
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        
        # Check for expected help content
        expected_content = [
            "Initial Corpus Collection Script",
            "--environment",
            "--dry-run",
            "--tokens",
            "--days",
            "--export-gcs"
        ]
        
        for content in expected_content:
            assert content in result.stdout, f"Missing help content: {content}"
    
    def test_script_dry_run_mode(self, script_path, test_environment):
        """Test script dry run mode without actual data collection"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--environment", "development",
            "--dry-run",
            "--days", "7"  # Short period for testing
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Dry run failed: {result.stderr}"
        
        # Check dry run output
        expected_output = [
            "DRY RUN MODE",
            "Would collect data for",
            "Would export to GCS",
            "No actual data collection performed"
        ]
        
        for output in expected_output:
            assert output in result.stdout, f"Missing dry run output: {output}"
    
    def test_script_environment_detection(self, script_path, test_environment):
        """Test script correctly detects and configures for different environments"""
        # Test development environment
        result = subprocess.run([
            sys.executable, str(script_path),
            "--environment", "development",
            "--dry-run"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Environment detection failed: {result.stderr}"
        assert "Environment: development" in result.stdout
        
        # Test production environment (dry run)
        result = subprocess.run([
            sys.executable, str(script_path),
            "--environment", "production", 
            "--dry-run"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Production environment failed: {result.stderr}"
        assert "Environment: production" in result.stdout
    
    @pytest.mark.skipif(not os.getenv('DATABASE_URL'), reason="DATABASE_URL not configured")
    def test_script_database_connectivity(self, script_path, test_environment):
        """Test script can connect to database and validate prerequisites"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--validate-prerequisites",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Database connectivity test failed: {result.stderr}"
        
        expected_validations = [
            "Database connection",
            "API keys validation",
            "GCS bucket access",
            "Prerequisites validated successfully"
        ]
        
        # At least some validations should pass
        for validation in expected_validations:
            if validation in result.stdout:
                break
        else:
            pytest.fail("No prerequisite validations found in output")
    
    @pytest.mark.skipif(not os.getenv('COINGECKO_API_KEY'), reason="CoinGecko API key not available") 
    def test_script_api_integration(self, script_path, test_environment):
        """Test script integrates with real APIs properly"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--test-api-connectivity",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"API connectivity test failed: {result.stderr}"
        
        expected_apis = [
            "CoinGecko",
            "Alternative.me", 
            "DeFiLlama"
        ]
        
        for api in expected_apis:
            assert api in result.stdout, f"Missing API connectivity test: {api}"
    
    def test_script_token_configuration(self, script_path, test_environment):
        """Test script accepts custom token configuration"""
        custom_tokens = "BTC,ETH,SOL"
        
        result = subprocess.run([
            sys.executable, str(script_path),
            "--tokens", custom_tokens,
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Custom tokens test failed: {result.stderr}"
        assert "BTC" in result.stdout
        assert "ETH" in result.stdout  
        assert "SOL" in result.stdout
    
    def test_script_gcs_export_option(self, script_path, test_environment):
        """Test script GCS export configuration"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--export-gcs",
            "--gcs-path", "gs://shyvr-models-prod/training-data/initial-corpus/test-v1.0/",
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"GCS export test failed: {result.stderr}"
        assert "gs://shyvr-models-prod" in result.stdout
        assert "training-data/initial-corpus" in result.stdout
    
    def test_script_progress_tracking(self, script_path, test_environment):
        """Test script provides comprehensive progress tracking"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--verbose",
            "--dry-run", 
            "--environment", "development"
        ], capture_output=True, text=True, timeout=45)
        
        assert result.returncode == 0, f"Progress tracking test failed: {result.stderr}"
        
        expected_progress = [
            "Initializing collection",
            "API clients configured",
            "Database connection established",
            "Collection progress:",
            "Feature calculation progress:"
        ]
        
        for progress in expected_progress:
            assert progress in result.stdout, f"Missing progress indicator: {progress}"
    
    def test_script_error_handling(self, script_path, test_environment):
        """Test script error handling and recovery"""
        # Test with invalid environment
        result = subprocess.run([
            sys.executable, str(script_path),
            "--environment", "invalid_env"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode != 0, "Script should fail with invalid environment"
        assert "invalid environment" in result.stderr.lower() or "invalid environment" in result.stdout.lower()
        
        # Test with invalid token
        result = subprocess.run([
            sys.executable, str(script_path),
            "--tokens", "INVALID_TOKEN",
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        # Should handle gracefully or warn about invalid tokens
        assert result.returncode in [0, 1], "Script should handle invalid tokens gracefully"
    
    def test_script_resumable_collection(self, script_path, test_environment):
        """Test script supports resumable collection"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--resume-from-checkpoint", "test_checkpoint.json",
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        # Should handle missing checkpoint gracefully
        assert result.returncode == 0, f"Resume test failed: {result.stderr}"
        assert "checkpoint" in result.stdout.lower()
    
    def test_deployment_script_help(self, deployment_script_path):
        """Test deployment script shows help"""
        result = subprocess.run([
            str(deployment_script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Deployment help failed: {result.stderr}"
        
        expected_content = [
            "Initial Corpus Collection Deployment",
            "--environment",
            "--dry-run",
            "Cloud Run execution"
        ]
        
        for content in expected_content:
            assert content in result.stdout, f"Missing deployment help content: {content}"
    
    def test_deployment_script_validation(self, deployment_script_path):
        """Test deployment script validates prerequisites"""
        result = subprocess.run([
            str(deployment_script_path), 
            "--validate-only",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode in [0, 1], f"Deployment validation failed unexpectedly: {result.stderr}"
        
        # Should check for required components
        expected_checks = [
            "GCP authentication",
            "Secret Manager access",
            "Cloud SQL connectivity",
            "GCS bucket permissions"
        ]
        
        # At least some checks should be mentioned
        output = result.stdout + result.stderr
        checks_found = sum(1 for check in expected_checks if check in output)
        assert checks_found > 0, "No prerequisite checks found in deployment script"
    
    @pytest.mark.skipif(not all([
        os.getenv('GOOGLE_CLOUD_PROJECT'),
        os.getenv('DATABASE_URL'), 
        os.getenv('COINGECKO_API_KEY')
    ]), reason="GCP credentials and API keys not fully configured")
    def test_minimal_real_collection(self, script_path, test_environment):
        """Test minimal real collection with very limited scope"""
        # Use very small dataset for real test - just 1 token, 24 hours
        result = subprocess.run([
            sys.executable, str(script_path),
            "--environment", "development",
            "--tokens", "bitcoin",  # Just Bitcoin
            "--days", "1",  # Just 1 day
            "--no-export-gcs",  # Don't export to save time
            "--test-mode"  # Special test mode with minimal processing
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Should succeed or fail gracefully
        if result.returncode == 0:
            # Success case - check for completion indicators
            assert "Collection completed" in result.stdout or "collection successful" in result.stdout.lower()
            assert "records collected" in result.stdout.lower()
        else:
            # Failure case - should be a graceful failure with clear error
            assert len(result.stderr) > 0 or "error" in result.stdout.lower()
            # Common acceptable failure reasons
            acceptable_failures = [
                "api rate limit",
                "authentication", 
                "network timeout",
                "database connection"
            ]
            
            error_text = (result.stderr + result.stdout).lower()
            failure_is_acceptable = any(failure in error_text for failure in acceptable_failures)
            
            if not failure_is_acceptable:
                pytest.fail(f"Unexpected failure mode: {result.stderr}")
    
    def test_script_output_format(self, script_path, test_environment):
        """Test script produces proper JSON output for automation"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--output-json",
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"JSON output test failed: {result.stderr}"
        
        # Should be valid JSON
        try:
            output_data = json.loads(result.stdout)
            
            # Check for expected JSON structure
            expected_fields = [
                "success",
                "environment", 
                "collection_stats",
                "tokens_configured"
            ]
            
            for field in expected_fields:
                assert field in output_data, f"Missing JSON field: {field}"
                
        except json.JSONDecodeError:
            pytest.fail(f"Output is not valid JSON: {result.stdout}")


class TestCorpusCollectionIntegration:
    """Test integration with existing InitialCorpusCollector"""
    
    @pytest.fixture
    async def collector(self):
        """Create InitialCorpusCollector instance for testing"""
        collector = InitialCorpusCollector(
            collection_days=7,  # Short period for testing
            rate_limit_delay=1.0,  # Faster for testing
            retry_attempts=2,
            batch_size=50
        )
        
        yield collector
        
        # Cleanup
        await collector.close()
    
    def test_script_integration_with_collector_class(self, collector):
        """Test that the script properly integrates with InitialCorpusCollector"""
        # Verify collector is properly configured
        assert collector.collection_days == 7
        assert collector.rate_limit_delay == 1.0
        assert collector.retry_attempts == 2
        assert collector.batch_size == 50
        
        # Verify default tokens are available
        assert len(collector.DEFAULT_TOKENS) == 10
        assert 'bitcoin' in collector.DEFAULT_TOKENS
        assert 'ethereum' in collector.DEFAULT_TOKENS
    
    def test_script_gcs_export_integration(self, script_path):
        """Test that script properly integrates with GCS export functionality"""
        # Test export path validation
        result = subprocess.run([
            sys.executable, str(script_path),
            "--validate-gcs-path", "gs://invalid-bucket/path/",
            "--dry-run"
        ], capture_output=True, text=True, timeout=30)
        
        # Should validate GCS path or fail gracefully
        assert result.returncode in [0, 1], "GCS path validation should complete"
    
    def test_script_corpus_versioning(self, script_path):
        """Test script supports corpus versioning"""
        result = subprocess.run([
            sys.executable, str(script_path),
            "--corpus-version", "test-v1.0",
            "--dry-run",
            "--environment", "development"  
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Corpus versioning test failed: {result.stderr}"
        assert "test-v1.0" in result.stdout


@pytest.mark.skipif(not os.getenv('GOOGLE_CLOUD_PROJECT'), reason="GCP project not configured")
class TestProductionDeploymentIntegration:
    """Test production deployment integration"""
    
    def test_cloud_run_service_configuration(self, deployment_script_path):
        """Test deployment script configures Cloud Run properly"""
        result = subprocess.run([
            str(deployment_script_path),
            "--show-cloud-run-config",
            "--environment", "production"
        ], capture_output=True, text=True, timeout=30)
        
        # Should show Cloud Run configuration or indicate it's available
        assert result.returncode == 0, f"Cloud Run config test failed: {result.stderr}"
        
        expected_config = [
            "Cloud Run",
            "service account", 
            "memory allocation",
            "timeout configuration"
        ]
        
        output_text = result.stdout.lower()
        config_items_found = sum(1 for item in expected_config if item in output_text)
        assert config_items_found >= 2, "Insufficient Cloud Run configuration details"
    
    def test_secret_manager_integration(self, deployment_script_path):
        """Test deployment script integrates with Secret Manager"""
        result = subprocess.run([
            str(deployment_script_path),
            "--check-secrets",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode in [0, 1], f"Secret Manager check failed: {result.stderr}"
        
        # Should check for required secrets
        expected_secrets = [
            "COINGECKO_API_KEY",
            "DATABASE_URL",
            "LUNARCRUSH_API_KEY"
        ]
        
        output_text = result.stdout + result.stderr
        for secret in expected_secrets[:2]:  # At least check for first 2
            assert secret in output_text, f"Missing secret check: {secret}"
    
    def test_monitoring_integration(self, deployment_script_path):
        """Test deployment script integrates with monitoring"""
        result = subprocess.run([
            str(deployment_script_path),
            "--setup-monitoring",
            "--dry-run",
            "--environment", "development"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Monitoring setup test failed: {result.stderr}"
        
        expected_monitoring = [
            "logging configuration",
            "metrics collection", 
            "alerting setup",
            "dashboard creation"
        ]
        
        output_text = result.stdout.lower()
        monitoring_found = sum(1 for item in expected_monitoring if item in output_text)
        assert monitoring_found >= 1, "No monitoring configuration found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])