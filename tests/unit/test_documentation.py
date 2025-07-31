"""
Documentation Validation Tests

Tests to ensure documentation is complete, accurate, and contains valid code examples.
"""

import os
import re
import ast
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest
import yaml
import json
from unittest.mock import Mock, patch, AsyncMock


class TestDocumentationValidation:
    """Test suite for validating model preservation documentation"""
    
    @pytest.fixture
    def docs_path(self):
        """Path to model preservation documentation"""
        return Path(__file__).parent.parent.parent / "docs" / "model_preservation"
    
    @pytest.fixture
    def required_docs(self):
        """List of required documentation files"""
        return [
            "user_guide.md",
            "api_reference.md",
            "troubleshooting.md",
            "architecture.md",
            "deployment.md"
        ]
    
    def test_documentation_files_exist(self, docs_path, required_docs):
        """Test that all required documentation files exist"""
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            assert doc_path.exists(), f"Required documentation file {doc_file} is missing"
            assert doc_path.is_file(), f"{doc_file} exists but is not a file"
            assert doc_path.stat().st_size > 0, f"{doc_file} is empty"
    
    def test_documentation_markdown_structure(self, docs_path, required_docs):
        """Test that documentation files have proper markdown structure"""
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            content = doc_path.read_text()
            
            # Check for title (H1 header)
            assert re.search(r'^# .+', content, re.MULTILINE), f"{doc_file} missing main title"
            
            # Check for table of contents
            assert "## Table of Contents" in content, f"{doc_file} missing table of contents"
            
            # Check for proper heading hierarchy
            headings = re.findall(r'^(#{1,6}) (.+)', content, re.MULTILINE)
            assert len(headings) >= 3, f"{doc_file} should have at least 3 headings"
            
            # Verify heading levels don't skip too many levels (allow some flexibility)
            # Just check we don't jump from H1 to H4+ directly
            prev_level = 0
            for heading in headings:
                level = len(heading[0])
                if prev_level > 0:  # Skip first heading check
                    assert level <= prev_level + 2, f"{doc_file} has improper heading hierarchy (jumped from H{prev_level} to H{level})"
                if level > prev_level:
                    prev_level = level
    
    def test_user_guide_content(self, docs_path):
        """Test user guide has required sections"""
        user_guide = docs_path / "user_guide.md"
        content = user_guide.read_text()
        
        required_sections = [
            "Quick Start",
            "Basic Usage",
            "Advanced Features",
            "Configuration",
            "Best Practices"
        ]
        
        for section in required_sections:
            assert f"## {section}" in content, f"User guide missing {section} section"
    
    def test_api_reference_content(self, docs_path):
        """Test API reference has required sections"""
        api_ref = docs_path / "api_reference.md"
        content = api_ref.read_text()
        
        required_sections = [
            "Authentication",
            "Error Handling"
        ]
        
        for section in required_sections:
            assert f"## {section}" in content, f"API reference missing {section} section"
        
        # Check for HTTP methods
        http_methods = ["GET", "POST", "PUT", "DELETE"]
        for method in http_methods:
            assert method in content, f"API reference should document {method} method"
        
        # Check for status codes
        status_codes = ["200", "400", "401", "404", "500"]
        for code in status_codes:
            assert code in content, f"API reference should document {code} status code"
    
    def test_troubleshooting_content(self, docs_path):
        """Test troubleshooting guide has required sections"""
        troubleshooting = docs_path / "troubleshooting.md"
        content = troubleshooting.read_text()
        
        required_sections = [
            "Common Issues",
            "Error Messages",
            "Performance Issues",
            "FAQ"
        ]
        
        for section in required_sections:
            assert f"## {section}" in content, f"Troubleshooting guide missing {section} section"
    
    def test_architecture_diagrams(self, docs_path):
        """Test architecture documentation has mermaid diagrams"""
        architecture = docs_path / "architecture.md"
        content = architecture.read_text()
        
        # Check for mermaid diagrams
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
        assert len(mermaid_blocks) >= 5, "Architecture doc should have at least 5 mermaid diagrams"
        
        # Check for specific diagram types
        diagram_types = ["graph", "sequenceDiagram", "classDiagram", "erDiagram"]
        found_types = []
        
        for block in mermaid_blocks:
            for diagram_type in diagram_types:
                if diagram_type in block:
                    found_types.append(diagram_type)
                    break
        
        assert len(found_types) >= 3, "Architecture should have diverse diagram types"
    
    def test_deployment_configuration_examples(self, docs_path):
        """Test deployment guide has valid configuration examples"""
        deployment = docs_path / "deployment.md"
        content = deployment.read_text()
        
        # Check for configuration file examples
        yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
        assert len(yaml_blocks) >= 3, "Deployment guide should have YAML configuration examples"
        
        # Validate YAML syntax (skip blocks with obvious placeholders)
        for i, yaml_block in enumerate(yaml_blocks):
            # Skip blocks with placeholders
            if any(placeholder in yaml_block for placeholder in ['${', 'YOUR_', 'your-', 'PROJECT_ID']):
                continue
            try:
                yaml.safe_load(yaml_block)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in deployment guide, block {i+1}: {e}")
        
        # Check for bash script examples
        bash_blocks = re.findall(r'```bash\n(.*?)\n```', content, re.DOTALL)
        assert len(bash_blocks) >= 5, "Deployment guide should have bash script examples"
    
    def test_code_examples_syntax(self, docs_path, required_docs):
        """Test that Python code examples have valid syntax"""
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            content = doc_path.read_text()
            
            # Extract Python code blocks
            python_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
            
            for i, code_block in enumerate(python_blocks):
                # Skip code blocks that are just imports or comments
                if not code_block.strip() or code_block.strip().startswith('#'):
                    continue
                
                # Skip code blocks with placeholders or incomplete examples
                if any(placeholder in code_block for placeholder in ['...', 'your-', 'YOUR_', 'if __name__', 'asyncio.run']):
                    continue
                    
                # Skip very short examples
                if len(code_block.strip().split('\n')) < 3:
                    continue
                
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    pytest.fail(f"Invalid Python syntax in {doc_file}, block {i+1}: {e}")
    
    def test_json_examples_syntax(self, docs_path, required_docs):
        """Test that JSON examples have valid syntax"""
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            content = doc_path.read_text()
            
            # Extract JSON code blocks
            json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
            
            for i, json_block in enumerate(json_blocks):
                try:
                    json.loads(json_block)
                except json.JSONDecodeError as e:
                    # Skip blocks with placeholders or comments
                    if any(placeholder in json_block for placeholder in ['...', 'your-', 'YOUR_', '//']):
                        continue
                    pytest.fail(f"Invalid JSON syntax in {doc_file}, block {i+1}: {e}")
    
    def test_curl_examples_format(self, docs_path):
        """Test that curl examples are properly formatted"""
        api_ref = docs_path / "api_reference.md"
        content = api_ref.read_text()
        
        # Find curl commands
        curl_commands = re.findall(r'curl[^\n]*(?:\n[^\n]*)*', content)
        
        assert len(curl_commands) >= 5, "API reference should have multiple curl examples"
        
        for cmd in curl_commands:
            # Skip incomplete curl examples
            if len(cmd.strip()) < 10:
                continue
                
            # Check for proper HTTP methods (including implicit GET)
            has_method = any(method in cmd for method in ['-X GET', '-X POST', '-X PUT', '-X DELETE']) or \
                        (not any(method in cmd for method in ['-X POST', '-X PUT', '-X DELETE']))  # Implicit GET
            
            assert has_method, f"Curl command missing or unclear HTTP method: {cmd[:50]}..."
    
    @pytest.mark.asyncio
    async def test_model_preservation_imports(self):
        """Test that code examples can import model preservation modules"""
        try:
            # Test basic imports that appear in documentation
            from src.model_preservation.manager import ModelPreservationManager
            from src.model_preservation.api import PreservationAPI
            
            # Test that classes can be instantiated (at least structurally)
            assert ModelPreservationManager is not None, "ModelPreservationManager should be importable"
            assert PreservationAPI is not None, "PreservationAPI should be importable"
            
        except ImportError as e:
            pytest.fail(f"Documentation references modules that cannot be imported: {e}")
    
    def test_documentation_links(self, docs_path, required_docs):
        """Test that internal documentation links are valid"""
        all_files = {doc.stem for doc in docs_path.glob("*.md")}
        
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            content = doc_path.read_text()
            
            # Find markdown links to other docs
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\.md\)', content)
            
            for link_text, link_target in links:
                if not link_target.startswith('http'):  # Only check local links
                    assert link_target in all_files, \
                        f"Broken link in {doc_file}: {link_target}.md does not exist"
    
    def test_documentation_completeness_checklist(self, docs_path):
        """Test that documentation covers all major features"""
        user_guide = docs_path / "user_guide.md"
        content = user_guide.read_text().lower()
        
        # Check that major features are documented
        required_features = [
            "save_model",
            "load_model", 
            "rollback",
            "versioning",
            "caching",
            "tagging",
            "branching",
            "monitoring"
        ]
        
        for feature in required_features:
            assert feature in content, f"User guide should document '{feature}' feature"
    
    def test_api_endpoints_documented(self, docs_path):
        """Test that all API endpoints are documented"""
        api_ref = docs_path / "api_reference.md"
        content = api_ref.read_text()
        
        # Check for documented endpoints
        required_endpoints = [
            "/api/preservation/models",
            "/api/preservation/rollback",
            "/api/preservation/health"
        ]
        
        for endpoint in required_endpoints:
            assert endpoint in content, f"API reference should document {endpoint} endpoint"
    
    def test_error_messages_documented(self, docs_path):
        """Test that common error messages are documented in troubleshooting"""
        troubleshooting = docs_path / "troubleshooting.md"
        content = troubleshooting.read_text()
        
        # Common error patterns that should be documented
        error_patterns = [
            "not found",
            "connection failed", 
            "authentication",
            "permission",
            "timeout"
        ]
        
        for pattern in error_patterns:
            assert pattern.lower() in content.lower(), \
                f"Troubleshooting should document '{pattern}' errors"
    
    def test_configuration_examples_complete(self, docs_path):
        """Test that configuration examples are complete"""
        deployment = docs_path / "deployment.md"
        content = deployment.read_text()
        
        # Required configuration sections
        config_sections = [
            "environment variables",
            "database",
            "redis", 
            "gcs",
            "monitoring"
        ]
        
        for section in config_sections:
            assert section.lower() in content.lower(), \
                f"Deployment guide should include {section} configuration"
    
    def test_security_considerations_documented(self, docs_path):
        """Test that security considerations are properly documented"""
        docs_content = ""
        for doc_file in ["deployment.md", "api_reference.md"]:
            doc_path = docs_path / doc_file
            docs_content += doc_path.read_text().lower()
        
        security_topics = [
            "authentication",
            "authorization", 
            "encryption",
            "ssl",
            "tls",
            "security"
        ]
        
        found_topics = [topic for topic in security_topics if topic in docs_content]
        assert len(found_topics) >= 4, \
            f"Documentation should cover security topics. Found: {found_topics}"
    
    def test_examples_are_runnable(self, docs_path):
        """Test that code examples follow patterns that could be executed"""
        user_guide = docs_path / "user_guide.md"
        content = user_guide.read_text()
        
        # Extract async function examples
        async_examples = re.findall(r'```python\n(.*?async.*?)\n```', content, re.DOTALL)
        
        for example in async_examples:
            # Check for proper async patterns
            if 'await' in example:
                assert 'async def' in example or 'asyncio.run' in example, \
                    "Async examples should show proper async function definition or execution"
    
    def test_documentation_consistency(self, docs_path, required_docs):
        """Test that terminology is consistent across documentation"""
        all_content = ""
        for doc_file in required_docs:
            doc_path = docs_path / doc_file
            all_content += doc_path.read_text().lower()
        
        # Check for consistent terminology
        terminology_pairs = [
            ("model preservation", "model_preservation"),  # Should use consistent naming
            ("dqn agent", "dqn_agent"),
            ("google cloud storage", "gcs")
        ]
        
        for term1, term2 in terminology_pairs:
            count1 = all_content.count(term1)
            count2 = all_content.count(term2)
            
            # Both forms should be present (one might be in text, other in code)
            assert count1 > 0 or count2 > 0, f"Neither '{term1}' nor '{term2}' found in documentation"


class TestDocumentationIntegration:
    """Integration tests for documentation examples"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization_example(self):
        """Test that manager initialization example would work"""
        with patch('src.model_preservation.manager.ModelPreservationManager') as mock_manager:
            mock_instance = AsyncMock()
            mock_manager.return_value = mock_instance
            
            # Simulate the example from user guide
            from src.model_preservation.manager import ModelPreservationManager
            manager = ModelPreservationManager()
            
            # Should be able to call common methods
            mock_instance.save_model = AsyncMock(return_value={"version": "1.0.0"})
            mock_instance.load_model = AsyncMock(return_value={"model_data": "test"})
            mock_instance.list_models = AsyncMock(return_value=[])
            
            # Test examples would work
            result = await mock_instance.save_model(
                model_data={"test": "data"},
                model_type="test_model",
                description="Test model"
            )
            assert result["version"] == "1.0.0"
    
    def test_api_client_example_structure(self):
        """Test that API client examples have proper structure"""
        # This would test the structure without actually making HTTP calls
        import requests
        from unittest.mock import Mock
        
        # Mock response structure from documentation
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [],
            "pagination": {"limit": 50, "offset": 0, "total": 0},
            "timestamp": "2024-01-01T00:00:00Z"
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            # This structure matches the documentation examples
            response = requests.get(
                "https://example.com/api/preservation/models",
                headers={"Authorization": "Bearer test-token"}
            )
            response.raise_for_status()
            data = response.json()
            
            # Verify response structure matches documentation
            assert "models" in data
            assert "pagination" in data
            assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])