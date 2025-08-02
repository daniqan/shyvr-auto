"""
TDD Tests for Version Range Parsing Implementation

Following TDD methodology, these tests define expected behavior for:
1. Version range parsing functionality
2. SemVer range specifications (>=, ~, ^, etc.)
3. Complex range combinations

Tests are designed to FAIL initially and then PASS after implementation.
"""

import pytest
from typing import Dict, Any, List, Union

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.model_preservation.versioning import (
    parse_version_range, 
    SemanticVersion,
    VersionError
)


class TestVersionRangeParsingBasic:
    """TDD tests for basic version range parsing functionality."""
    
    def test_parse_version_range_should_not_raise_nie(self):
        """
        TDD: Test that parse_version_range doesn't raise NotImplementedError.
        
        This test should FAIL initially since the function raises NotImplementedError.
        After implementation, it should PASS with actual parsing.
        """
        # This should NOT raise NotImplementedError after implementation
        result = parse_version_range(">=1.0.0")
        
        # Should return a meaningful result
        assert result is not None
        assert isinstance(result, dict)
        
    def test_parse_greater_than_equal_range(self):
        """
        TDD: Test parsing >=version range.
        
        Should FAIL initially and PASS after implementing >= parsing.
        """
        result = parse_version_range(">=1.2.3")
        
        assert isinstance(result, dict)
        assert 'operator' in result
        assert 'version' in result
        assert result['operator'] == '>='
        assert isinstance(result['version'], SemanticVersion)
        assert str(result['version']) == "1.2.3"
        
    def test_parse_greater_than_range(self):
        """
        TDD: Test parsing >version range.
        
        Should FAIL initially and PASS after implementing > parsing.
        """
        result = parse_version_range(">2.0.0")
        
        assert result['operator'] == '>'
        assert str(result['version']) == "2.0.0"
        
    def test_parse_less_than_equal_range(self):
        """
        TDD: Test parsing <=version range.
        
        Should FAIL initially and PASS after implementing <= parsing.
        """
        result = parse_version_range("<=3.1.4")
        
        assert result['operator'] == '<='
        assert str(result['version']) == "3.1.4"
        
    def test_parse_less_than_range(self):
        """
        TDD: Test parsing <version range.
        
        Should FAIL initially and PASS after implementing < parsing.
        """
        result = parse_version_range("<1.0.0")
        
        assert result['operator'] == '<'
        assert str(result['version']) == "1.0.0"
        
    def test_parse_exact_version_range(self):
        """
        TDD: Test parsing exact version (no operator).
        
        Should FAIL initially and PASS after implementing exact matching.
        """
        result = parse_version_range("1.5.2")
        
        assert result['operator'] == '='
        assert str(result['version']) == "1.5.2"


class TestVersionRangeParsingAdvanced:
    """TDD tests for advanced version range syntax."""
    
    def test_parse_tilde_range_patch_level(self):
        """
        TDD: Test parsing ~version range (patch-level changes).
        
        ~1.2.3 should match >=1.2.3 and <1.3.0
        Should FAIL initially and PASS after implementing ~ parsing.
        """
        result = parse_version_range("~1.2.3")
        
        assert result['operator'] == '~'
        assert str(result['version']) == "1.2.3"
        assert 'min_version' in result
        assert 'max_version' in result
        assert str(result['min_version']) == "1.2.3"
        assert str(result['max_version']) == "1.3.0"
        
    def test_parse_tilde_range_minor_level(self):
        """
        TDD: Test parsing ~version range for minor-level.
        
        ~1.2 should match >=1.2.0 and <1.3.0
        """
        result = parse_version_range("~1.2")
        
        assert result['operator'] == '~'
        assert str(result['min_version']) == "1.2.0"
        assert str(result['max_version']) == "1.3.0"
        
    def test_parse_caret_range_compatible(self):
        """
        TDD: Test parsing ^version range (compatible within major version).
        
        ^1.2.3 should match >=1.2.3 and <2.0.0
        Should FAIL initially and PASS after implementing ^ parsing.
        """
        result = parse_version_range("^1.2.3")
        
        assert result['operator'] == '^'
        assert str(result['version']) == "1.2.3"
        assert 'min_version' in result
        assert 'max_version' in result
        assert str(result['min_version']) == "1.2.3"
        assert str(result['max_version']) == "2.0.0"
        
    def test_parse_caret_range_zero_major(self):
        """
        TDD: Test parsing ^version range for 0.x versions.
        
        ^0.2.3 should match >=0.2.3 and <0.3.0
        """
        result = parse_version_range("^0.2.3")
        
        assert str(result['min_version']) == "0.2.3"
        assert str(result['max_version']) == "0.3.0"
        
    def test_parse_caret_range_zero_minor(self):
        """
        TDD: Test parsing ^version range for 0.0.x versions.
        
        ^0.0.3 should match >=0.0.3 and <0.0.4
        """
        result = parse_version_range("^0.0.3")
        
        assert str(result['min_version']) == "0.0.3"
        assert str(result['max_version']) == "0.0.4"


class TestVersionRangeParsingPrerelease:
    """TDD tests for version range parsing with prerelease versions."""
    
    def test_parse_range_with_prerelease(self):
        """
        TDD: Test parsing ranges with prerelease versions.
        
        Should FAIL initially and PASS after implementing prerelease support.
        """
        result = parse_version_range(">=1.0.0-alpha")
        
        assert result['operator'] == '>='
        assert str(result['version']) == "1.0.0-alpha"
        assert result['version'].is_prerelease()
        
    def test_parse_tilde_range_with_prerelease(self):
        """
        TDD: Test tilde range with prerelease versions.
        
        ~1.2.3-beta should handle prerelease semantics correctly.
        """
        result = parse_version_range("~1.2.3-beta.1")
        
        assert result['operator'] == '~'
        assert str(result['version']) == "1.2.3-beta.1"
        
    def test_parse_caret_range_with_prerelease(self):
        """
        TDD: Test caret range with prerelease versions.
        
        ^1.2.3-alpha should handle prerelease semantics correctly.
        """
        result = parse_version_range("^1.2.3-alpha.2")
        
        assert result['operator'] == '^'
        assert str(result['version']) == "1.2.3-alpha.2"


class TestVersionRangeParsingComplex:
    """TDD tests for complex version range expressions."""
    
    def test_parse_compound_range(self):
        """
        TDD: Test parsing compound ranges (AND operations).
        
        ">=1.0.0 <2.0.0" should parse as compound range.
        Should FAIL initially and PASS after implementing compound parsing.
        """
        result = parse_version_range(">=1.0.0 <2.0.0")
        
        assert result['type'] == 'compound'
        assert 'ranges' in result
        assert len(result['ranges']) == 2
        
        range1, range2 = result['ranges']
        assert range1['operator'] == '>='
        assert str(range1['version']) == "1.0.0"
        assert range2['operator'] == '<'
        assert str(range2['version']) == "2.0.0"
        
    def test_parse_or_range(self):
        """
        TDD: Test parsing OR ranges.
        
        "1.0.x || 2.0.x" should parse as OR range.
        Should FAIL initially and PASS after implementing OR parsing.
        """
        result = parse_version_range(">=1.0.0 <1.1.0 || >=2.0.0 <2.1.0")
        
        assert result['type'] == 'or'
        assert 'ranges' in result
        assert len(result['ranges']) == 2
        
    def test_parse_wildcard_range(self):
        """
        TDD: Test parsing wildcard ranges.
        
        "1.2.x" should match any patch version in 1.2 series.
        Should FAIL initially and PASS after implementing wildcard parsing.
        """
        result = parse_version_range("1.2.x")
        
        assert result['operator'] == 'wildcard'
        assert result['pattern'] == "1.2.x"
        assert str(result['min_version']) == "1.2.0"
        assert str(result['max_version']) == "1.3.0"
        
    def test_parse_wildcard_minor_range(self):
        """
        TDD: Test parsing minor wildcard ranges.
        
        "1.x" should match any version in 1.x series.
        """
        result = parse_version_range("1.x")
        
        assert result['operator'] == 'wildcard'
        assert result['pattern'] == "1.x"
        assert str(result['min_version']) == "1.0.0"
        assert str(result['max_version']) == "2.0.0"


class TestVersionRangeParsingValidation:
    """TDD tests for version range parsing validation."""
    
    def test_parse_invalid_range_raises_error(self):
        """
        TDD: Test that invalid range strings raise appropriate errors.
        
        Should FAIL initially and PASS after implementing validation.
        """
        invalid_ranges = [
            ">>1.0.0",     # Invalid operator
            ">=invalid",   # Invalid version
            "~",           # Missing version
            "^",           # Missing version
            "1.0.0 1.1.0", # Missing operator in compound
        ]
        
        for invalid_range in invalid_ranges:
            with pytest.raises(VersionError):
                parse_version_range(invalid_range)
                
    def test_parse_empty_range_raises_error(self):
        """
        TDD: Test that empty range string raises error.
        """
        with pytest.raises(VersionError):
            parse_version_range("")
            
        with pytest.raises(VersionError):
            parse_version_range("   ")
            
    def test_parse_none_range_raises_error(self):
        """
        TDD: Test that None range raises appropriate error.
        """
        with pytest.raises((VersionError, TypeError)):
            parse_version_range(None)


class TestVersionRangeMatchingFunctionality:
    """TDD tests for version range matching functionality."""
    
    def test_version_matches_parsed_range(self):
        """
        TDD: Test that parsed ranges can match versions correctly.
        
        Should FAIL initially and PASS after implementing matching logic.
        """
        range_spec = parse_version_range(">=1.0.0")
        
        # Should have a method to check if version matches range
        assert 'matches' in range_spec or hasattr(range_spec, 'matches')
        
        # Test versions
        test_versions = [
            ("1.0.0", True),
            ("1.0.1", True),
            ("1.1.0", True),
            ("2.0.0", True),
            ("0.9.9", False),
        ]
        
        for version_str, should_match in test_versions:
            version = SemanticVersion(version_str)
            if 'matches' in range_spec:
                matches = range_spec['matches'](version)
            else:
                matches = range_spec.matches(version)
            assert matches == should_match, f"Version {version_str} should {'match' if should_match else 'not match'} range >=1.0.0"
            
    def test_tilde_range_matching(self):
        """
        TDD: Test tilde range matching logic.
        
        ~1.2.3 should match 1.2.3, 1.2.4, 1.2.10 but not 1.3.0
        """
        range_spec = parse_version_range("~1.2.3")
        
        test_cases = [
            ("1.2.3", True),
            ("1.2.4", True),
            ("1.2.10", True),
            ("1.3.0", False),
            ("1.1.9", False),
            ("2.0.0", False),
        ]
        
        for version_str, should_match in test_cases:
            version = SemanticVersion(version_str)
            if 'matches' in range_spec:
                matches = range_spec['matches'](version)
            else:
                matches = range_spec.matches(version)
            assert matches == should_match
            
    def test_caret_range_matching(self):
        """
        TDD: Test caret range matching logic.
        
        ^1.2.3 should match 1.2.3, 1.2.4, 1.9.0 but not 2.0.0
        """
        range_spec = parse_version_range("^1.2.3")
        
        test_cases = [
            ("1.2.3", True),
            ("1.2.4", True),
            ("1.9.0", True),
            ("1.9.99", True),
            ("2.0.0", False),
            ("1.1.9", False),
            ("0.9.0", False),
        ]
        
        for version_str, should_match in test_cases:
            version = SemanticVersion(version_str)
            if 'matches' in range_spec:
                matches = range_spec['matches'](version)
            else:
                matches = range_spec.matches(version)
            assert matches == should_match


class TestVersionRangeUtilityFunctions:
    """TDD tests for utility functions related to version ranges."""
    
    def test_version_satisfies_range_function(self):
        """
        TDD: Test utility function to check if version satisfies range.
        
        Should FAIL initially and PASS after implementing version_satisfies_range.
        """
        from src.model_preservation.versioning import version_satisfies_range
        
        # Should have this utility function
        assert version_satisfies_range is not None
        
        # Test basic cases
        assert version_satisfies_range("1.0.0", ">=1.0.0") is True
        assert version_satisfies_range("0.9.9", ">=1.0.0") is False
        assert version_satisfies_range("1.2.5", "~1.2.3") is True
        assert version_satisfies_range("1.3.0", "~1.2.3") is False
        
    def test_find_versions_matching_range(self):
        """
        TDD: Test utility function to find versions matching a range.
        
        Should FAIL initially and PASS after implementing find_versions_matching_range.
        """
        from src.model_preservation.versioning import find_versions_matching_range
        
        available_versions = [
            "1.0.0", "1.0.1", "1.1.0", "1.2.0", "1.2.3", "1.2.5", "2.0.0", "2.1.0"
        ]
        
        # Test >= range
        matching = find_versions_matching_range(available_versions, ">=1.2.0")
        expected = ["1.2.0", "1.2.3", "1.2.5", "2.0.0", "2.1.0"]
        assert set(str(v) for v in matching) == set(expected)
        
        # Test ~ range
        matching = find_versions_matching_range(available_versions, "~1.2.0")
        expected = ["1.2.0", "1.2.3", "1.2.5"]
        assert set(str(v) for v in matching) == set(expected)
        
        # Test ^ range
        matching = find_versions_matching_range(available_versions, "^1.1.0")
        expected = ["1.1.0", "1.2.0", "1.2.3", "1.2.5"]
        assert set(str(v) for v in matching) == set(expected)


class TestVersionRangeRegressionCases:
    """TDD tests for edge cases and regression scenarios."""
    
    def test_parse_range_with_v_prefix(self):
        """
        TDD: Test parsing ranges with 'v' prefix in versions.
        
        Should handle ">=v1.0.0" correctly.
        """
        result = parse_version_range(">=v1.0.0")
        
        assert result['operator'] == '>='
        assert str(result['version']) == "1.0.0"  # v prefix should be normalized
        
    def test_parse_range_with_whitespace(self):
        """
        TDD: Test parsing ranges with extra whitespace.
        
        Should handle " >= 1.0.0 " correctly.
        """
        result = parse_version_range("  >=  1.0.0  ")
        
        assert result['operator'] == '>='
        assert str(result['version']) == "1.0.0"
        
    def test_parse_range_case_insensitive_operators(self):
        """
        TDD: Test case insensitive parsing if applicable.
        
        May not be needed, but good to define expected behavior.
        """
        # Most range operators are case-sensitive by convention
        # But we should define the expected behavior
        result = parse_version_range(">=1.0.0")
        assert result['operator'] == '>='
        
    def test_performance_with_complex_ranges(self):
        """
        TDD: Test performance with complex range expressions.
        
        Should handle complex ranges efficiently.
        """
        complex_range = ">=1.0.0 <2.0.0 || >=3.0.0 <4.0.0 || ^5.0.0"
        
        # Should parse without timeout
        import time
        start_time = time.time()
        result = parse_version_range(complex_range)
        parse_time = time.time() - start_time
        
        # Should parse quickly (less than 1 second)
        assert parse_time < 1.0
        assert result is not None