"""
Test suite to detect hardcoded credentials in configuration files.
This test follows TDD methodology to ensure no hardcoded credentials exist in production.
"""

import os
import re
import pytest
from pathlib import Path
from typing import List, Dict, Tuple


class TestHardcodedCredentials:
    """Test class to scan for hardcoded credentials in configuration files."""
    
    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @pytest.fixture
    def credential_patterns(self) -> Dict[str, str]:
        """Define patterns for detecting hardcoded credentials."""
        return {
            'admin_password': r'(?i)(admin123|admin/admin|password.*admin)',
            'default_password': r'(?i)(dev_password|default_password|change_in_prod)',
            'hardcoded_key': r'(?i)(dev-key|test-key|demo-key)',
            'basic_auth': r'(?i)(admin:admin|user:password)',
            'grafana_default': r'(?i)(GRAFANA_PASSWORD.*admin123)',
            'pgadmin_default': r'(?i)(PGADMIN_DEFAULT_PASSWORD.*admin123)',
        }
    
    @pytest.fixture
    def config_files(self, project_root: Path) -> List[Path]:
        """Get list of configuration files to scan."""
        config_extensions = {'.yml', '.yaml', '.env', '.conf', '.ini'}
        config_files = []
        
        # Scan specific directories for config files
        config_dirs = [
            project_root / 'config',
            project_root / 'docker',
            project_root / 'monitoring',
            project_root / '.github',
        ]
        
        for config_dir in config_dirs:
            if config_dir.exists():
                for file_path in config_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix in config_extensions:
                        config_files.append(file_path)
        
        # Also check specific files
        specific_files = [
            project_root / 'env.template',
            project_root / '.env.example',
        ]
        
        for file_path in specific_files:
            if file_path.exists():
                config_files.append(file_path)
        
        return config_files
    
    def test_no_hardcoded_credentials_in_production_files(
        self, 
        config_files: List[Path], 
        credential_patterns: Dict[str, str]
    ):
        """Test that no hardcoded credentials exist in production configuration files."""
        violations = []
        
        for file_path in config_files:
            # Skip local development files and template files - allow defaults for development
            if any(pattern in str(file_path).lower() for pattern in ['local', 'dev', 'test', 'template', 'example']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    line_number = 0
                    
                    for line in content.split('\n'):
                        line_number += 1
                        line_clean = line.strip()
                        
                        # Skip comments and empty lines
                        if not line_clean or line_clean.startswith('#'):
                            continue
                        
                        # Check for hardcoded credentials that don't use environment variables
                        if 'admin123' in line and '${' not in line:
                            violations.append({
                                'file': str(file_path),
                                'line': line_number,
                                'pattern': 'hardcoded_admin_password',
                                'content': line.strip()
                            })
                        
                        # Check for other unsafe patterns in production files
                        for pattern_name, pattern in credential_patterns.items():
                            if re.search(pattern, line) and '${' not in line:
                                violations.append({
                                    'file': str(file_path),
                                    'line': line_number,
                                    'pattern': pattern_name,
                                    'content': line.strip()
                                })
            
            except (UnicodeDecodeError, FileNotFoundError):
                # Skip binary files or files that can't be read
                continue
        
        # Report violations
        if violations:
            violation_report = "\n".join([
                f"File: {v['file']}:{v['line']} - Pattern: {v['pattern']} - Content: {v['content']}"
                for v in violations
            ])
            pytest.fail(
                f"Found {len(violations)} hardcoded credential violations in production files:\n{violation_report}"
            )
    
    def test_docker_compose_production_uses_env_vars(self, project_root: Path):
        """Test that docker-compose.production.yml uses environment variables for credentials."""
        production_compose = project_root / 'docker' / 'docker-compose.production.yml'
        
        if not production_compose.exists():
            pytest.skip("Production docker-compose file not found")
        
        with open(production_compose, 'r') as f:
            content = f.read()
        
        # Check that sensitive values use environment variables
        sensitive_patterns = [
            r'GF_SECURITY_ADMIN_PASSWORD=\$\{GRAFANA_PASSWORD\}',  # Should use ${GRAFANA_PASSWORD}
            r'POSTGRES_PASSWORD=\$\{POSTGRES_PASSWORD\}',  # Should use ${POSTGRES_PASSWORD}
            r'SECRET_KEY=\$\{SECRET_KEY\}',  # Should use ${SECRET_KEY}
        ]
        
        for pattern in sensitive_patterns:
            if not re.search(pattern, content):
                pytest.fail(
                    f"Production docker-compose should use environment variables for passwords. "
                    f"Pattern not found: {pattern}"
                )
    
    def test_no_localhost_in_production_configs(self, project_root: Path):
        """Test that production configuration files don't contain localhost references."""
        production_files = [
            project_root / 'docker' / 'docker-compose.production.yml',
            project_root / 'monitoring' / 'grafana' / 'prometheus.yml',
            project_root / 'config' / 'config.yaml',
        ]
        
        violations = []
        
        for file_path in production_files:
            if not file_path.exists():
                continue
            
            with open(file_path, 'r') as f:
                content = f.read()
                line_number = 0
                
                for line in content.split('\n'):
                    line_number += 1
                    line_clean = line.strip()
                    
                    # Skip comments
                    if line_clean.startswith('#'):
                        continue
                    
                    # Check for localhost references (excluding health checks)
                    if ('localhost' in line_clean.lower() and 
                        'healthcheck' not in line_clean.lower() and
                        'health' not in line_clean.lower()):
                        violations.append({
                            'file': str(file_path),
                            'line': line_number,
                            'content': line.strip()
                        })
        
        if violations:
            violation_report = "\n".join([
                f"File: {v['file']}:{v['line']} - Content: {v['content']}"
                for v in violations
            ])
            pytest.fail(
                f"Found {len(violations)} localhost references in production configs:\n{violation_report}"
            )
    
    def test_environment_variables_required_for_secrets(self, project_root: Path):
        """Test that all secret values must be provided via environment variables."""
        env_template = project_root / 'env.template'
        
        # Required environment variables for production
        required_secrets = [
            'SECRET_KEY',
            'GRAFANA_PASSWORD',
            'POSTGRES_PASSWORD',
            'DATABASE_URL',
        ]
        
        if env_template.exists():
            with open(env_template, 'r') as f:
                env_content = f.read()
            
            missing_secrets = []
            for secret in required_secrets:
                if secret not in env_content:
                    missing_secrets.append(secret)
            
            if missing_secrets:
                pytest.fail(
                    f"env.template missing required secrets: {', '.join(missing_secrets)}"
                )
        else:
            # If no env.template exists, create it with required variables
            pytest.fail("env.template file is required but not found")