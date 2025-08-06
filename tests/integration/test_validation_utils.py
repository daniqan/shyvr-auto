"""
Test Validation Utilities for Production Scenario Tests

Utilities to validate that production scenario tests are properly structured,
follow TDD methodology, and meet quality standards for testing a transformer
trading system in production conditions.

Use `uv run` for all Python execution.
"""

import pytest
import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class TestValidationResult:
    """Result of test validation"""
    test_file: str
    test_name: str
    is_valid: bool
    violations: List[str]
    recommendations: List[str]
    tdd_compliant: bool
    production_ready: bool


@dataclass
class TestSuiteValidationSummary:
    """Summary of test suite validation"""
    total_tests: int
    valid_tests: int
    invalid_tests: int
    tdd_compliant_tests: int
    production_ready_tests: int
    common_violations: List[Tuple[str, int]]  # (violation, count)
    overall_score: float
    recommendations: List[str]


class TestValidationRules:
    """Rules for validating production scenario tests"""
    
    # Required patterns in TDD tests
    TDD_PATTERNS = [
        r'with pytest\.raises\(\(AttributeError, NotImplementedError, AssertionError\)\)',
        r'pytest\.fail\(.*will fail until.*implementation',
        r'# Act - This will fail as implementation doesn\'t exist'
    ]
    
    # Production scenario requirements
    PRODUCTION_REQUIREMENTS = [
        'real.*market.*data',
        'ensemble.*decision',
        'transformer.*model',
        'risk.*management',
        'safety.*system',
        'performance.*metric',
        'scenario.*validation'
    ]
    
    # Anti-patterns to avoid
    ANTI_PATTERNS = [
        r'from unittest\.mock import.*Mock(?!.*# TDD|.*# Following TDD)',  # Mocks without TDD comment
        r'\.assert_called_with\(',  # Mock assertions
        r'return_value\s*=',  # Mock return values in production code
        r'side_effect\s*=',  # Mock side effects in production code
    ]
    
    # Required test structure elements
    REQUIRED_ELEMENTS = [
        'arrange',
        'act', 
        'assert',
        'scenario',
        'validation'
    ]


class ProductionScenarioTestValidator:
    """Validator for production scenario tests"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.rules = TestValidationRules()
    
    def validate_test_file(self, test_file_path: Path) -> List[TestValidationResult]:
        """Validate all tests in a test file"""
        self.logger.info(f"Validating test file: {test_file_path}")
        
        results = []
        
        try:
            # Read and parse the test file
            with open(test_file_path, 'r') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Find all test methods
            test_methods = self._extract_test_methods(tree, content)
            
            for test_method in test_methods:
                result = self._validate_individual_test(
                    test_file_path.name, 
                    test_method,
                    content
                )
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Error validating test file: {e}")
            # Create error result
            results.append(TestValidationResult(
                test_file=test_file_path.name,
                test_name="file_parsing_error",
                is_valid=False,
                violations=[f"Could not parse test file: {str(e)}"],
                recommendations=["Fix syntax errors in test file"],
                tdd_compliant=False,
                production_ready=False
            ))
        
        return results
    
    def _extract_test_methods(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract test methods from AST"""
        test_methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Extract method details
                method_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'lineno': node.lineno,
                    'source': self._get_method_source(node, content),
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                }
                test_methods.append(method_info)
        
        return test_methods
    
    def _get_method_source(self, node: ast.FunctionDef, content: str) -> str:
        """Extract source code for a method"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        
        # Find the end of the method (next def or class, or end of file)
        end_line = len(lines)
        for i, line in enumerate(lines[start_line + 1:], start_line + 1):
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                end_line = i
                break
        
        return '\n'.join(lines[start_line:end_line])
    
    def _validate_individual_test(
        self, 
        test_file: str, 
        test_method: Dict[str, Any],
        full_content: str
    ) -> TestValidationResult:
        """Validate an individual test method"""
        violations = []
        recommendations = []
        
        test_name = test_method['name']
        test_source = test_method['source']
        docstring = test_method['docstring'] or ""
        
        # Check TDD compliance
        tdd_compliant = self._check_tdd_compliance(test_source, violations, recommendations)
        
        # Check production scenario requirements
        production_ready = self._check_production_requirements(
            test_source, docstring, violations, recommendations
        )
        
        # Check for anti-patterns
        self._check_anti_patterns(test_source, violations, recommendations)
        
        # Check test structure
        self._check_test_structure(test_source, violations, recommendations)
        
        # Check async/await patterns for integration tests
        self._check_async_patterns(test_method, violations, recommendations)
        
        # Check scenario coverage
        self._check_scenario_coverage(test_name, docstring, violations, recommendations)
        
        is_valid = len(violations) == 0
        
        return TestValidationResult(
            test_file=test_file,
            test_name=test_name,
            is_valid=is_valid,
            violations=violations,
            recommendations=recommendations,
            tdd_compliant=tdd_compliant,
            production_ready=production_ready
        )
    
    def _check_tdd_compliance(
        self, 
        test_source: str, 
        violations: List[str], 
        recommendations: List[str]
    ) -> bool:
        """Check if test follows TDD methodology"""
        tdd_score = 0
        
        # Check for TDD patterns
        for pattern in self.rules.TDD_PATTERNS:
            if re.search(pattern, test_source, re.IGNORECASE | re.MULTILINE):
                tdd_score += 1
        
        # Check for TDD comments
        if 'TDD' in test_source or 'Test-Driven Development' in test_source:
            tdd_score += 1
        
        # Check for proper failure expectation
        if 'will fail until' in test_source.lower():
            tdd_score += 1
        
        # Check for implementation comments
        if "doesn't exist" in test_source or "implementation exists" in test_source:
            tdd_score += 1
        
        tdd_compliant = tdd_score >= 3  # At least 3 TDD indicators
        
        if not tdd_compliant:
            violations.append("Test does not follow TDD methodology properly")
            recommendations.append("Add proper TDD patterns: pytest.raises, failing assertions, implementation comments")
        
        return tdd_compliant
    
    def _check_production_requirements(
        self, 
        test_source: str, 
        docstring: str, 
        violations: List[str], 
        recommendations: List[str]
    ) -> bool:
        """Check if test meets production scenario requirements"""
        production_score = 0
        combined_text = (test_source + " " + docstring).lower()
        
        # Check for production requirements
        for requirement in self.rules.PRODUCTION_REQUIREMENTS:
            if re.search(requirement, combined_text):
                production_score += 1
        
        # Additional production checks
        production_indicators = [
            'real.world', 'actual.market', 'historical.event', 'production.condition',
            'transformer.ensemble', 'risk.limit', 'safety.trigger', 'performance.validation'
        ]
        
        for indicator in production_indicators:
            if re.search(indicator, combined_text):
                production_score += 1
        
        production_ready = production_score >= 5  # At least 5 production indicators
        
        if not production_ready:
            violations.append("Test does not adequately cover production scenario requirements")
            recommendations.append("Add more production-specific validations: real market data, transformer models, risk management, safety systems")
        
        return production_ready
    
    def _check_anti_patterns(
        self, 
        test_source: str, 
        violations: List[str], 
        recommendations: List[str]
    ):
        """Check for anti-patterns that should be avoided"""
        for pattern in self.rules.ANTI_PATTERNS:
            matches = re.finditer(pattern, test_source, re.MULTILINE)
            for match in matches:
                # Check if it's in a TDD context (has TDD comment nearby)
                context_start = max(0, match.start() - 200)
                context_end = min(len(test_source), match.end() + 200)
                context = test_source[context_start:context_end]
                
                if 'TDD' not in context and 'Following TDD' not in context:
                    violations.append(f"Anti-pattern detected: {match.group()}")
                    recommendations.append("Remove mock usage in production code - use real components following TDD")
    
    def _check_test_structure(
        self, 
        test_source: str, 
        violations: List[str], 
        recommendations: List[str]
    ):
        """Check test follows proper structure (Arrange, Act, Assert)"""
        structure_score = 0
        
        # Check for Arrange, Act, Assert comments or structure
        if re.search(r'#.*arrange', test_source, re.IGNORECASE):
            structure_score += 1
        if re.search(r'#.*act', test_source, re.IGNORECASE):
            structure_score += 1
        if re.search(r'#.*assert', test_source, re.IGNORECASE):
            structure_score += 1
        
        # Check for logical structure even without comments
        if 'trading_system' in test_source.lower() or 'system_components' in test_source:
            structure_score += 1
        if 'with pytest.raises' in test_source or 'await' in test_source:
            structure_score += 1
        if 'assert' in test_source or 'pytest.fail' in test_source:
            structure_score += 1
        
        if structure_score < 3:
            violations.append("Test lacks clear Arrange/Act/Assert structure")
            recommendations.append("Structure test with clear Arrange, Act, Assert sections")
    
    def _check_async_patterns(
        self, 
        test_method: Dict[str, Any], 
        violations: List[str], 
        recommendations: List[str]
    ):
        """Check async/await patterns for integration tests"""
        decorators = test_method.get('decorators', [])
        source = test_method.get('source', '')
        
        # Integration tests should be async
        if 'pytest.mark.asyncio' not in decorators and '@pytest.mark.asyncio' not in source:
            if 'await' in source:
                violations.append("Test uses await but missing @pytest.mark.asyncio decorator")
                recommendations.append("Add @pytest.mark.asyncio decorator for async tests")
        
        # Check for proper async patterns in integration tests
        if 'integration' in test_method.get('name', '').lower():
            if 'await' not in source:
                violations.append("Integration test should use async/await patterns")
                recommendations.append("Use async/await for integration test operations")
    
    def _check_scenario_coverage(
        self, 
        test_name: str, 
        docstring: str, 
        violations: List[str], 
        recommendations: List[str]
    ):
        """Check if test properly covers scenario requirements"""
        scenario_indicators = [
            'bull_market', 'bear_market', 'flash_crash', 'low_liquidity',
            'news_events', 'correlation_breakdown', 'hft_competition',
            'network_issues', 'regulatory', 'sentiment'
        ]
        
        has_scenario = any(indicator in test_name.lower() for indicator in scenario_indicators)
        
        if not has_scenario and 'scenario' in test_name.lower():
            violations.append("Test name suggests scenario testing but lacks specific scenario type")
            recommendations.append("Specify the type of scenario being tested in test name")
        
        # Check docstring for scenario description
        if docstring and len(docstring) > 50:
            if 'validates:' not in docstring.lower() and 'tests:' not in docstring.lower():
                recommendations.append("Add 'Validates:' section to docstring listing specific validations")


class TestSuiteValidator:
    """Validator for entire test suites"""
    
    def __init__(self):
        self.test_validator = ProductionScenarioTestValidator()
        self.logger = structlog.get_logger()
    
    def validate_test_suite(self, test_directory: Path) -> TestSuiteValidationSummary:
        """Validate an entire test suite"""
        self.logger.info(f"Validating test suite in: {test_directory}")
        
        all_results = []
        
        # Find all test files
        test_files = list(test_directory.glob("test_*.py"))
        
        for test_file in test_files:
            file_results = self.test_validator.validate_test_file(test_file)
            all_results.extend(file_results)
        
        # Analyze results
        total_tests = len(all_results)
        valid_tests = sum(1 for r in all_results if r.is_valid)
        invalid_tests = total_tests - valid_tests
        tdd_compliant_tests = sum(1 for r in all_results if r.tdd_compliant)
        production_ready_tests = sum(1 for r in all_results if r.production_ready)
        
        # Count common violations
        violation_counts = {}
        for result in all_results:
            for violation in result.violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        common_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate overall score
        if total_tests > 0:
            overall_score = (
                (valid_tests / total_tests * 0.4) +
                (tdd_compliant_tests / total_tests * 0.3) +
                (production_ready_tests / total_tests * 0.3)
            ) * 100
        else:
            overall_score = 0
        
        # Generate recommendations
        recommendations = self._generate_suite_recommendations(
            all_results, common_violations, overall_score
        )
        
        return TestSuiteValidationSummary(
            total_tests=total_tests,
            valid_tests=valid_tests,
            invalid_tests=invalid_tests,
            tdd_compliant_tests=tdd_compliant_tests,
            production_ready_tests=production_ready_tests,
            common_violations=common_violations[:10],  # Top 10
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def _generate_suite_recommendations(
        self, 
        results: List[TestValidationResult],
        common_violations: List[Tuple[str, int]],
        overall_score: float
    ) -> List[str]:
        """Generate recommendations for the entire test suite"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("Overall test quality is below 70% - focus on improving test structure and compliance")
        
        # Address most common violations
        if common_violations:
            top_violation = common_violations[0]
            recommendations.append(f"Most common issue: '{top_violation[0]}' appears in {top_violation[1]} tests")
        
        # TDD compliance
        tdd_compliance_rate = sum(1 for r in results if r.tdd_compliant) / len(results) if results else 0
        if tdd_compliance_rate < 0.8:
            recommendations.append("Less than 80% of tests are TDD compliant - add proper TDD patterns and failure expectations")
        
        # Production readiness
        production_readiness_rate = sum(1 for r in results if r.production_ready) / len(results) if results else 0
        if production_readiness_rate < 0.7:
            recommendations.append("Less than 70% of tests are production ready - add more real-world scenario validation")
        
        return recommendations
    
    def print_validation_report(self, summary: TestSuiteValidationSummary):
        """Print validation report to console"""
        print("\n" + "="*80)
        print("PRODUCTION SCENARIO TEST VALIDATION REPORT")
        print("="*80)
        print(f"Total Tests: {summary.total_tests}")
        print(f"Valid Tests: {summary.valid_tests}")
        print(f"Invalid Tests: {summary.invalid_tests}")
        print(f"TDD Compliant: {summary.tdd_compliant_tests}")
        print(f"Production Ready: {summary.production_ready_tests}")
        print(f"Overall Score: {summary.overall_score:.1f}%")
        
        if summary.common_violations:
            print("\nMOST COMMON VIOLATIONS:")
            for violation, count in summary.common_violations[:5]:
                print(f"  {count:2d}x {violation}")
        
        if summary.recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"{i}. {rec}")
        
        print("="*80)


def validate_production_scenario_tests():
    """Main function to validate production scenario tests"""
    test_directory = Path(__file__).parent
    validator = TestSuiteValidator()
    
    summary = validator.validate_test_suite(test_directory)
    validator.print_validation_report(summary)
    
    return summary.overall_score >= 70  # Return success if score >= 70%


if __name__ == "__main__":
    success = validate_production_scenario_tests()
    exit(0 if success else 1)