"""
Test suite for semantic versioning functionality in model preservation system.

This test suite follows TDD methodology and tests all aspects of semantic versioning:
- Version parsing and validation
- Version comparison and ordering
- Version increment operations
- Error handling for invalid versions
- Integration with ModelMetadata
"""

import pytest
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.model_preservation.versioning import SemanticVersion, VersionError


class TestSemanticVersionParsing:
    """Test semantic version string parsing."""
    
    def test_parse_basic_version(self):
        """Test parsing basic semantic version strings."""
        version = SemanticVersion("1.0.0")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease is None
        assert version.build is None
    
    def test_parse_version_with_v_prefix(self):
        """Test parsing versions with 'v' prefix."""
        version = SemanticVersion("v2.1.3")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 3
    
    def test_parse_version_with_prerelease(self):
        """Test parsing versions with prerelease identifiers."""
        version = SemanticVersion("1.0.0-alpha")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha"
        
        version = SemanticVersion("2.1.0-beta.1")
        assert version.prerelease == "beta.1"
        
        version = SemanticVersion("1.2.3-rc.1.2.3")
        assert version.prerelease == "rc.1.2.3"
    
    def test_parse_version_with_build_metadata(self):
        """Test parsing versions with build metadata."""
        version = SemanticVersion("1.0.0+20230101.sha.abc123")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.build == "20230101.sha.abc123"
    
    def test_parse_version_with_prerelease_and_build(self):
        """Test parsing versions with both prerelease and build metadata."""
        version = SemanticVersion("1.0.0-alpha.1+build.1")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha.1"
        assert version.build == "build.1"
    
    def test_parse_invalid_version_formats(self):
        """Test that invalid version formats raise VersionError."""
        invalid_versions = [
            "1.0",  # Missing patch
            "1",    # Missing minor and patch
            "1.0.0.0",  # Too many version parts
            "a.b.c",    # Non-numeric version parts
            "1.0.-1",   # Negative patch version
            "",         # Empty string
            "1.0.0-",   # Empty prerelease
            "1.0.0+",   # Empty build metadata
        ]
        
        for invalid in invalid_versions:
            with pytest.raises(VersionError, match=f"Invalid semantic version format: {invalid}"):
                SemanticVersion(invalid)
    
    def test_parse_edge_cases(self):
        """Test parsing edge cases."""
        # Large version numbers
        version = SemanticVersion("999.999.999")
        assert version.major == 999
        assert version.minor == 999
        assert version.patch == 999
        
        # Zero versions
        version = SemanticVersion("0.0.0")
        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0


class TestSemanticVersionComparison:
    """Test semantic version comparison operations."""
    
    def test_equality_comparison(self):
        """Test version equality comparison."""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.0.0")
        v3 = SemanticVersion("v1.0.0")  # With prefix should be equal
        v4 = SemanticVersion("1.0.1")
        
        assert v1 == v2
        assert v1 == v3
        assert v1 != v4
        assert hash(v1) == hash(v2)  # Hash consistency
    
    def test_less_than_comparison(self):
        """Test version less-than comparison."""
        # Major version comparison
        assert SemanticVersion("1.0.0") < SemanticVersion("2.0.0")
        assert not SemanticVersion("2.0.0") < SemanticVersion("1.0.0")
        
        # Minor version comparison
        assert SemanticVersion("1.0.0") < SemanticVersion("1.1.0")
        assert not SemanticVersion("1.1.0") < SemanticVersion("1.0.0")
        
        # Patch version comparison
        assert SemanticVersion("1.0.0") < SemanticVersion("1.0.1")
        assert not SemanticVersion("1.0.1") < SemanticVersion("1.0.0")
    
    def test_greater_than_comparison(self):
        """Test version greater-than comparison."""
        assert SemanticVersion("2.0.0") > SemanticVersion("1.0.0")
        assert SemanticVersion("1.1.0") > SemanticVersion("1.0.0")
        assert SemanticVersion("1.0.1") > SemanticVersion("1.0.0")
        
        assert not SemanticVersion("1.0.0") > SemanticVersion("2.0.0")
    
    def test_less_equal_greater_equal(self):
        """Test less-equal and greater-equal comparisons."""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.0.0")
        v3 = SemanticVersion("1.0.1")
        
        assert v1 <= v2
        assert v1 <= v3
        assert v2 >= v1
        assert v3 >= v1
    
    def test_prerelease_comparison(self):
        """Test comparison with prerelease versions."""
        # Prerelease versions have lower precedence than normal versions
        assert SemanticVersion("1.0.0-alpha") < SemanticVersion("1.0.0")
        assert SemanticVersion("1.0.0-beta") < SemanticVersion("1.0.0")
        
        # Prerelease comparison
        assert SemanticVersion("1.0.0-alpha") < SemanticVersion("1.0.0-beta")
        assert SemanticVersion("1.0.0-alpha.1") < SemanticVersion("1.0.0-alpha.2")
        assert SemanticVersion("1.0.0-alpha.beta") < SemanticVersion("1.0.0-beta")
        assert SemanticVersion("1.0.0-beta") < SemanticVersion("1.0.0-beta.2")
        assert SemanticVersion("1.0.0-beta.2") < SemanticVersion("1.0.0-beta.11")
        assert SemanticVersion("1.0.0-beta.11") < SemanticVersion("1.0.0-rc.1")
    
    def test_build_metadata_ignored_in_comparison(self):
        """Test that build metadata is ignored in version comparison."""
        v1 = SemanticVersion("1.0.0+build.1")
        v2 = SemanticVersion("1.0.0+build.2") 
        v3 = SemanticVersion("1.0.0")
        
        assert v1 == v2  # Build metadata ignored
        assert v1 == v3  # Build metadata ignored
        assert not v1 < v2
        assert not v1 > v2
    
    def test_version_sorting(self):
        """Test sorting a list of versions."""
        versions = [
            SemanticVersion("2.0.0"),
            SemanticVersion("1.0.0-alpha"),
            SemanticVersion("1.0.0"),
            SemanticVersion("1.0.1"),
            SemanticVersion("1.0.0-beta"),
            SemanticVersion("0.9.0"),
        ]
        
        expected = [
            SemanticVersion("0.9.0"),
            SemanticVersion("1.0.0-alpha"),
            SemanticVersion("1.0.0-beta"),
            SemanticVersion("1.0.0"),
            SemanticVersion("1.0.1"),
            SemanticVersion("2.0.0"),
        ]
        
        sorted_versions = sorted(versions)
        assert sorted_versions == expected


class TestSemanticVersionIncrement:
    """Test semantic version increment operations."""
    
    def test_bump_patch(self):
        """Test patch version increment."""
        version = SemanticVersion("1.0.0")
        bumped = version.bump_patch()
        
        assert bumped.major == 1
        assert bumped.minor == 0
        assert bumped.patch == 1
        assert bumped.prerelease is None
        assert bumped.build is None
        
        # Original version unchanged
        assert version.patch == 0
    
    def test_bump_minor(self):
        """Test minor version increment."""
        version = SemanticVersion("1.0.5")
        bumped = version.bump_minor()
        
        assert bumped.major == 1
        assert bumped.minor == 1
        assert bumped.patch == 0  # Reset to 0
        assert bumped.prerelease is None
        assert bumped.build is None
    
    def test_bump_major(self):
        """Test major version increment."""
        version = SemanticVersion("1.2.5")
        bumped = version.bump_major()
        
        assert bumped.major == 2
        assert bumped.minor == 0  # Reset to 0
        assert bumped.patch == 0  # Reset to 0
        assert bumped.prerelease is None
        assert bumped.build is None
    
    def test_bump_with_prerelease(self):
        """Test version increment removes prerelease and build metadata."""
        version = SemanticVersion("1.0.0-alpha+build.1")
        
        patch_bumped = version.bump_patch()
        assert str(patch_bumped) == "1.0.1"
        
        minor_bumped = version.bump_minor()
        assert str(minor_bumped) == "1.1.0"
        
        major_bumped = version.bump_major()
        assert str(major_bumped) == "2.0.0"
    
    def test_bump_edge_cases(self):
        """Test version increment edge cases."""
        # Maximum typical version numbers
        version = SemanticVersion("999.999.999")
        
        patch_bumped = version.bump_patch()
        assert patch_bumped.patch == 1000
        
        minor_bumped = version.bump_minor()
        assert minor_bumped.minor == 1000
        assert minor_bumped.patch == 0
        
        major_bumped = version.bump_major()
        assert major_bumped.major == 1000
        assert major_bumped.minor == 0
        assert major_bumped.patch == 0


class TestSemanticVersionUtilities:
    """Test semantic version utility methods."""
    
    def test_string_representation(self):
        """Test string representation of semantic versions."""
        assert str(SemanticVersion("1.0.0")) == "1.0.0"
        assert str(SemanticVersion("v2.1.3")) == "2.1.3"  # v prefix removed
        assert str(SemanticVersion("1.0.0-alpha")) == "1.0.0-alpha"
        assert str(SemanticVersion("1.0.0+build.1")) == "1.0.0+build.1"
        assert str(SemanticVersion("1.0.0-alpha+build.1")) == "1.0.0-alpha+build.1"
    
    def test_repr_representation(self):
        """Test repr representation of semantic versions."""
        version = SemanticVersion("1.0.0-alpha+build.1")
        expected = "SemanticVersion('1.0.0-alpha+build.1')"
        assert repr(version) == expected
    
    def test_is_prerelease(self):
        """Test prerelease version identification."""
        assert SemanticVersion("1.0.0-alpha").is_prerelease()
        assert SemanticVersion("1.0.0-beta.1").is_prerelease()
        assert SemanticVersion("1.0.0-rc.1+build.1").is_prerelease()
        
        assert not SemanticVersion("1.0.0").is_prerelease()
        assert not SemanticVersion("1.0.0+build.1").is_prerelease()
    
    def test_is_stable(self):
        """Test stable version identification."""
        assert SemanticVersion("1.0.0").is_stable()
        assert SemanticVersion("2.1.3").is_stable()
        assert SemanticVersion("1.0.0+build.1").is_stable()
        
        assert not SemanticVersion("1.0.0-alpha").is_stable()
        assert not SemanticVersion("0.9.0").is_stable()  # Major version 0 not stable
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        version = SemanticVersion("1.2.3-alpha.1+build.123")
        expected = {
            "major": 1,
            "minor": 2,
            "patch": 3,
            "prerelease": "alpha.1",
            "build": "build.123"
        }
        assert version.to_dict() == expected
        
        # Test without prerelease and build
        version = SemanticVersion("1.0.0")
        expected = {
            "major": 1,
            "minor": 0,
            "patch": 0,
            "prerelease": None,
            "build": None
        }
        assert version.to_dict() == expected
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "major": 1,
            "minor": 2,
            "patch": 3,
            "prerelease": "alpha.1",
            "build": "build.123"
        }
        version = SemanticVersion.from_dict(data)
        assert str(version) == "1.2.3-alpha.1+build.123"
        
        # Test minimal data
        data = {"major": 1, "minor": 0, "patch": 0}
        version = SemanticVersion.from_dict(data)
        assert str(version) == "1.0.0"


class TestSemanticVersionValidation:
    """Test semantic version validation utilities."""
    
    def test_is_valid_version_string(self):
        """Test validation of version strings."""
        from src.model_preservation.versioning import is_valid_semantic_version
        
        # Valid versions
        assert is_valid_semantic_version("1.0.0")
        assert is_valid_semantic_version("v2.1.3")
        assert is_valid_semantic_version("1.0.0-alpha")
        assert is_valid_semantic_version("1.0.0-alpha.1+build.1")
        assert is_valid_semantic_version("10.20.30")
        assert is_valid_semantic_version("0.0.1")
        
        # Invalid versions
        assert not is_valid_semantic_version("1.0")
        assert not is_valid_semantic_version("1")
        assert not is_valid_semantic_version("1.0.0.0")
        assert not is_valid_semantic_version("a.b.c")
        assert not is_valid_semantic_version("")
        assert not is_valid_semantic_version("1.0.-1")
    
    def test_validate_version_components(self):
        """Test validation of individual version components."""
        from src.model_preservation.versioning import validate_version_components
        
        # Valid components
        assert validate_version_components(1, 0, 0)
        assert validate_version_components(0, 0, 0)
        assert validate_version_components(999, 999, 999)
        
        # Invalid components
        assert not validate_version_components(-1, 0, 0)
        assert not validate_version_components(0, -1, 0)
        assert not validate_version_components(0, 0, -1)
    
    def test_normalize_version_string(self):
        """Test version string normalization."""
        from src.model_preservation.versioning import normalize_version_string
        
        assert normalize_version_string("v1.0.0") == "1.0.0"
        assert normalize_version_string("1.0.0") == "1.0.0"
        assert normalize_version_string("V2.1.3") == "2.1.3"
        assert normalize_version_string("1.0.0-alpha") == "1.0.0-alpha"


class TestSemanticVersionErrorHandling:
    """Test error handling in semantic versioning."""
    
    def test_version_error_inheritance(self):
        """Test that VersionError properly inherits from Exception."""
        try:
            raise VersionError("Test error")
        except Exception as e:
            assert isinstance(e, VersionError)
            assert str(e) == "Test error"
    
    def test_invalid_version_error_details(self):
        """Test that VersionError provides detailed error messages."""
        invalid_versions = [
            ("1.0", "Missing patch version"),
            ("a.b.c", "Non-numeric version components"),
            ("1.0.-1", "Negative version numbers not allowed"),
            ("", "Empty version string"),
        ]
        
        for version_str, expected_msg in invalid_versions:
            with pytest.raises(VersionError) as exc_info:
                SemanticVersion(version_str)
            
            assert "Invalid semantic version format" in str(exc_info.value)
            assert version_str in str(exc_info.value)
    
    def test_comparison_with_non_version_objects(self):
        """Test comparison with non-SemanticVersion objects."""
        version = SemanticVersion("1.0.0")
        
        # Comparison with strings should work
        assert version != "1.0.0"  # Different types
        
        # Comparison with other types should raise TypeError
        with pytest.raises(TypeError):
            version < 1
        
        with pytest.raises(TypeError):
            version > None
        
        with pytest.raises(TypeError):
            version <= []


class TestSemanticVersionIntegration:
    """Test integration of semantic versioning with model preservation system."""
    
    def test_version_in_model_metadata(self):
        """Test that SemanticVersion integrates properly with ModelMetadata."""
        # This test will be updated once we modify ModelMetadata
        # For now, we test the interface we expect to implement
        
        version_str = "1.0.0-alpha"
        version = SemanticVersion(version_str)
        
        # Test that version can be serialized/deserialized for storage
        version_dict = version.to_dict()
        reconstructed = SemanticVersion.from_dict(version_dict)
        assert version == reconstructed
        
        # Test string representation for database storage
        assert str(version) == "1.0.0-alpha"
    
    def test_version_comparison_for_manager(self):
        """Test version comparison functionality needed by PreservationManager."""
        versions = [
            SemanticVersion("1.0.0"),
            SemanticVersion("1.0.1"),
            SemanticVersion("1.1.0"),
            SemanticVersion("2.0.0-alpha"),
            SemanticVersion("2.0.0"),
        ]
        
        # Test finding latest version
        latest = max(versions)
        assert latest == SemanticVersion("2.0.0")
        
        # Test finding latest stable version
        stable_versions = [v for v in versions if v.is_stable()]
        latest_stable = max(stable_versions)
        assert latest_stable == SemanticVersion("2.0.0")
    
    def test_auto_increment_logic(self):
        """Test auto-increment logic for version generation."""
        # Test increment strategies that manager might use
        current = SemanticVersion("1.2.3")
        
        # Default increment (patch)
        next_patch = current.bump_patch()
        assert str(next_patch) == "1.2.4"
        
        # Feature increment (minor)
        next_minor = current.bump_minor()
        assert str(next_minor) == "1.3.0"
        
        # Breaking change increment (major)
        next_major = current.bump_major()
        assert str(next_major) == "2.0.0"


# Fixtures for testing
@pytest.fixture
def sample_versions():
    """Provide sample versions for testing."""
    return [
        SemanticVersion("0.1.0"),
        SemanticVersion("0.9.0"),
        SemanticVersion("1.0.0-alpha"),
        SemanticVersion("1.0.0"),
        SemanticVersion("1.0.1"),
        SemanticVersion("1.1.0"),
        SemanticVersion("2.0.0"),
    ]


@pytest.fixture
def version_with_metadata():
    """Provide a version with prerelease and build metadata."""
    return SemanticVersion("1.0.0-alpha.1+build.20230101.sha.abc123")


class TestSemanticVersionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_version_with_large_numbers(self):
        """Test versions with very large numbers."""
        version = SemanticVersion("999999.999999.999999")
        assert version.major == 999999
        assert version.minor == 999999
        assert version.patch == 999999
        
        bumped = version.bump_patch()
        assert bumped.patch == 1000000
    
    def test_complex_prerelease_identifiers(self):
        """Test complex prerelease identifier formats."""
        complex_versions = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-alpha.1.2.3",
            "1.0.0-alpha-beta",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
        ]
        
        for version_str in complex_versions:
            version = SemanticVersion(version_str)
            assert version.prerelease is not None
            assert str(version) == version_str
    
    def test_complex_build_metadata(self):
        """Test complex build metadata formats."""
        complex_builds = [
            "1.0.0+20130313144700",
            "1.0.0+exp.sha.5114f85",
            "1.0.0+21AF26D3-117B344092BD",
            "1.0.0+build.1.2.b8f12d7",
            "1.0.0+beta.2.build.1",
        ]
        
        for version_str in complex_builds:
            version = SemanticVersion(version_str)
            assert version.build is not None
            assert str(version) == version_str