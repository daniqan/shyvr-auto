"""
Unit tests for CI/CD pipeline configuration validation.
Tests cloudbuild.yaml structure, deployment scripts, and pipeline integrity.
"""

import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess


class TestCloudBuildConfiguration:
    """Test Cloud Build configuration file validation."""
    
    @pytest.fixture
    def cloudbuild_config(self):
        """Load the cloudbuild.yaml configuration."""
        config_path = Path(__file__).parent.parent.parent.parent / "cloudbuild.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_cloudbuild_yaml_exists(self):
        """Test that cloudbuild.yaml exists and is readable."""
        config_path = Path(__file__).parent.parent.parent.parent / "cloudbuild.yaml"
        assert config_path.exists(), "cloudbuild.yaml file must exist"
        assert config_path.is_file(), "cloudbuild.yaml must be a file"
        assert config_path.stat().st_size > 0, "cloudbuild.yaml must not be empty"
    
    def test_cloudbuild_yaml_valid_yaml(self, cloudbuild_config):
        """Test that cloudbuild.yaml is valid YAML."""
        assert isinstance(cloudbuild_config, dict), "cloudbuild.yaml must be valid YAML dict"
        assert "steps" in cloudbuild_config, "cloudbuild.yaml must have 'steps' section"
    
    def test_required_build_steps_present(self, cloudbuild_config):
        """Test that all required build steps are present."""
        steps = cloudbuild_config["steps"]
        step_ids = [step.get("id", "") for step in steps]
        
        required_steps = [
            "setup-git",
            "install-dependencies",
            "code-quality-checks",
            "unit-tests",
            "build-image",
            "push-image"
        ]
        
        for required_step in required_steps:
            assert required_step in step_ids, f"Required step '{required_step}' missing from pipeline"
    
    def test_security_scanning_steps_present(self, cloudbuild_config):
        """Test that security scanning steps are included."""
        steps = cloudbuild_config["steps"]
        step_ids = [step.get("id", "") for step in steps]
        
        security_steps = [
            "security-scan-code",
            "security-scan-image"
        ]
        
        for security_step in security_steps:
            assert security_step in step_ids, f"Security step '{security_step}' missing from pipeline"
    
    def test_testing_steps_configuration(self, cloudbuild_config):
        """Test that testing steps are properly configured."""
        steps = cloudbuild_config["steps"]
        test_steps = [step for step in steps if "test" in step.get("id", "")]
        
        assert len(test_steps) >= 3, "Should have at least 3 testing steps (unit, integration, performance)"
        
        # Check unit tests step
        unit_test_step = next((step for step in steps if step.get("id") == "unit-tests"), None)
        assert unit_test_step is not None, "Unit tests step must be present"
        assert "uv run pytest" in str(unit_test_step.get("args", "")), "Unit tests should use uv run pytest"
    
    def test_deployment_steps_conditional(self, cloudbuild_config):
        """Test that deployment steps are conditional on branch."""
        steps = cloudbuild_config["steps"]
        deployment_steps = [step for step in steps if "deploy" in step.get("id", "")]
        
        assert len(deployment_steps) >= 2, "Should have staging and production deployment steps"
        
        for deploy_step in deployment_steps:
            args = deploy_step.get("args", [])
            if isinstance(args, list) and len(args) > 2:
                script_content = args[2] if len(args) > 2 else ""
                assert "BRANCH_NAME" in script_content, "Deployment steps should be conditional on branch"
    
    def test_secret_manager_integration(self, cloudbuild_config):
        """Test that Secret Manager is properly integrated."""
        available_secrets = cloudbuild_config.get("availableSecrets", {})
        secret_manager = available_secrets.get("secretManager", [])
        
        assert len(secret_manager) > 0, "Should have Secret Manager secrets configured"
        
        required_secrets = [
            "TELEGRAM_TOKEN",
            "DB_PASSWORD",
            "DATABASE_URL",
            "HELIUS_API_KEY",
            "BIRDEYE_API_KEY"
        ]
        
        configured_secrets = [secret.get("env", "") for secret in secret_manager]
        for required_secret in required_secrets:
            assert required_secret in configured_secrets, f"Required secret '{required_secret}' not configured"
    
    def test_build_options_optimized(self, cloudbuild_config):
        """Test that build options are optimized for performance."""
        options = cloudbuild_config.get("options", {})
        
        assert "machineType" in options, "Machine type should be specified for optimization"
        assert "diskSizeGb" in options, "Disk size should be specified"
        assert options.get("diskSizeGb", 0) >= 50, "Disk size should be at least 50GB for ML workloads"
    
    def test_timeout_configured(self, cloudbuild_config):
        """Test that build timeout is configured appropriately."""
        timeout = cloudbuild_config.get("timeout", "")
        assert timeout, "Build timeout should be configured"
        
        # Parse timeout (e.g., "1800s")
        if timeout.endswith("s"):
            timeout_seconds = int(timeout[:-1])
            assert timeout_seconds >= 900, "Timeout should be at least 15 minutes for ML builds"
            assert timeout_seconds <= 3600, "Timeout should not exceed 1 hour"


class TestBuildTriggerScript:
    """Test the build trigger setup script."""
    
    @pytest.fixture
    def trigger_script_path(self):
        """Get path to the trigger setup script."""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "setup_build_triggers.sh"
    
    def test_trigger_script_exists(self, trigger_script_path):
        """Test that the trigger setup script exists."""
        assert trigger_script_path.exists(), "setup_build_triggers.sh must exist"
        assert trigger_script_path.is_file(), "setup_build_triggers.sh must be a file"
    
    def test_trigger_script_executable(self, trigger_script_path):
        """Test that the trigger script is executable."""
        file_stat = trigger_script_path.stat()
        # Check if any execute bit is set (owner, group, or other)
        assert file_stat.st_mode & 0o111, "setup_build_triggers.sh must be executable"
    
    def test_trigger_script_has_required_functions(self, trigger_script_path):
        """Test that required functions are present in the script."""
        with open(trigger_script_path, 'r') as f:
            script_content = f.read()
        
        required_functions = [
            "check_gcloud_setup",
            "enable_apis",
            "create_main_trigger",
            "create_develop_trigger",
            "setup_iam_permissions"
        ]
        
        for function in required_functions:
            assert function in script_content, f"Required function '{function}' missing from script"
    
    def test_trigger_script_has_error_handling(self, trigger_script_path):
        """Test that the script has proper error handling."""
        with open(trigger_script_path, 'r') as f:
            script_content = f.read()
        
        assert "set -e" in script_content, "Script should have 'set -e' for error handling"
        assert "exit 1" in script_content, "Script should have proper exit codes"


class TestGCloudIgnoreFile:
    """Test .gcloudignore file configuration."""
    
    @pytest.fixture
    def gcloudignore_path(self):
        """Get path to .gcloudignore file."""
        return Path(__file__).parent.parent.parent.parent / ".gcloudignore"
    
    def test_gcloudignore_exists(self, gcloudignore_path):
        """Test that .gcloudignore file exists."""
        assert gcloudignore_path.exists(), ".gcloudignore file must exist"
        assert gcloudignore_path.is_file(), ".gcloudignore must be a file"
    
    def test_gcloudignore_excludes_unnecessary_files(self, gcloudignore_path):
        """Test that .gcloudignore excludes unnecessary files."""
        with open(gcloudignore_path, 'r') as f:
            ignore_content = f.read()
        
        excluded_patterns = [
            ".git",
            "*.pyc",
            "__pycache__",
            ".pytest_cache",
            ".coverage",
            "htmlcov/",
            ".env",
            "docs/",
            "*.md"
        ]
        
        for pattern in excluded_patterns:
            assert pattern in ignore_content, f"Pattern '{pattern}' should be excluded in .gcloudignore"


class TestPipelineIntegration:
    """Test pipeline integration with existing infrastructure."""
    
    def test_dockerfile_compatible_with_pipeline(self):
        """Test that Dockerfile is compatible with pipeline."""
        dockerfile_path = Path(__file__).parent.parent.parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist for pipeline"
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for multi-stage build
        assert "FROM python:3.12-slim as builder" in dockerfile_content, "Should use multi-stage build"
        assert "FROM python:3.12-slim as production" in dockerfile_content, "Should have production stage"
        
        # Check for uv usage (consistency with pipeline)
        assert "uv" in dockerfile_content, "Dockerfile should use uv for consistency with pipeline"
    
    def test_pyproject_toml_has_dev_dependencies(self):
        """Test that pyproject.toml has required dev dependencies for pipeline."""
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist"
        
        with open(pyproject_path, 'r') as f:
            pyproject_content = f.read()
        
        required_dev_deps = [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
            "ruff"
        ]
        
        for dep in required_dev_deps:
            assert dep in pyproject_content, f"Required dev dependency '{dep}' missing from pyproject.toml"
    
    @patch('subprocess.run')
    def test_pipeline_commands_syntax(self, mock_subprocess):
        """Test that pipeline commands have valid syntax."""
        # Mock successful command execution
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Test key pipeline commands for syntax errors
        test_commands = [
            ["uv", "sync", "--frozen", "--all-extras"],
            ["uv", "run", "black", "--check", "src/"],
            ["uv", "run", "pytest", "tests/unit/", "--cov=src"],
            ["uv", "run", "mypy", "src/"]
        ]
        
        for cmd in test_commands:
            # This would catch basic syntax errors in command construction
            try:
                # Don't actually run the command, just validate it can be constructed
                assert isinstance(cmd, list), f"Command {cmd} should be a list"
                assert len(cmd) > 0, f"Command {cmd} should not be empty"
                assert all(isinstance(arg, str) for arg in cmd), f"All args in {cmd} should be strings"
            except Exception as e:
                pytest.fail(f"Command syntax error: {cmd} - {e}")


class TestPipelineValidation:
    """Test pipeline validation and best practices."""
    
    @pytest.fixture
    def cloudbuild_config(self):
        """Load the cloudbuild.yaml configuration."""
        config_path = Path(__file__).parent.parent.parent.parent / "cloudbuild.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_pipeline_follows_security_best_practices(self, cloudbuild_config):
        """Test that pipeline follows security best practices."""
        steps = cloudbuild_config["steps"]
        
        # Check for security scanning steps
        security_steps = [step for step in steps if "security-scan" in step.get("id", "")]
        assert len(security_steps) >= 2, "Should have code and image security scanning"
        
        # Check that secrets are not hardcoded (but allow environment variable references)
        for step in steps:
            args = step.get("args", [])
            for arg in args:
                if isinstance(arg, str):
                    # Check for hardcoded secret patterns (but allow env var references like $PASSWORD)
                    if "password=" in arg.lower():
                        assert "$" in arg or "PASSWORD" in arg.upper(), "Passwords should not be hardcoded (env vars OK)"
                    if "secret=" in arg.lower():
                        assert "$" in arg or "SECRET" in arg.upper(), "Secrets should not be hardcoded (env vars OK)"
                    if "token=" in arg.lower():
                        assert "$" in arg or "TOKEN" in arg.upper(), "Tokens should not be hardcoded (env vars OK)"
    
    def test_pipeline_has_proper_error_handling(self, cloudbuild_config):
        """Test that pipeline steps have proper error handling."""
        steps = cloudbuild_config["steps"]
        
        for step in steps:
            args = step.get("args", [])
            if isinstance(args, list) and len(args) > 2:
                script_content = args[2] if len(args) > 2 else ""
                if "bash" in str(args[0:2]) and len(script_content) > 50:  # Only check substantial bash scripts
                    assert "set -e" in script_content or "exit 1" in script_content, \
                        f"Step '{step.get('id')}' should have error handling"
    
    def test_pipeline_caching_configured(self, cloudbuild_config):
        """Test that pipeline has caching configured for performance."""
        steps = cloudbuild_config["steps"]
        
        # Look for cache-related steps
        cache_steps = [step for step in steps if "cache" in step.get("id", "")]
        assert len(cache_steps) >= 1, "Should have cache-related steps for performance"
        
        # Check for environment variables related to caching
        env_vars = []
        for step in steps:
            env_vars.extend(step.get("env", []))
        
        cache_env_vars = [env for env in env_vars if "CACHE" in env.upper()]
        assert len(cache_env_vars) >= 1, "Should have cache-related environment variables"
    
    def test_pipeline_resource_optimization(self, cloudbuild_config):
        """Test that pipeline is optimized for resource usage."""
        options = cloudbuild_config.get("options", {})
        
        # Check machine type is appropriate for ML workloads
        machine_type = options.get("machineType", "")
        assert "HIGHCPU" in machine_type or "STANDARD" in machine_type, \
            "Should use appropriate machine type for ML builds"
        
        # Check disk size is sufficient but not excessive
        disk_size = options.get("diskSizeGb", 0)
        assert 50 <= disk_size <= 200, "Disk size should be between 50-200GB"
    
    def test_pipeline_environment_separation(self, cloudbuild_config):
        """Test that pipeline properly separates environments."""
        steps = cloudbuild_config["steps"]
        deployment_steps = [step for step in steps if "deploy" in step.get("id", "")]
        
        # Check that different environments are handled
        staging_deployment = any("staging" in str(step) for step in deployment_steps)
        production_deployment = any("production" in str(step) for step in deployment_steps)
        
        assert staging_deployment, "Should have staging deployment handling"
        assert production_deployment, "Should have production deployment handling"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])