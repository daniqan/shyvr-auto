#!/usr/bin/env python3
"""
Activity Logging Test Suite Runner
Comprehensive test execution for the activity logging system following TDD methodology

Usage:
    python tests/run_activity_logging_tests.py [options]
    
Options:
    --fast          Run only fast tests (skip performance tests)
    --integration   Run integration tests
    --unit          Run unit tests only
    --coverage      Run with coverage reporting
    --verbose       Verbose output
    --parallel      Run tests in parallel
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(args):
    """Run the activity logging test suite"""
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Test directory - adjust for new location in tests/utils/
    test_dir = Path(__file__).parent.parent / "unit" / "logging"
    pytest_cmd.append(str(test_dir))
    
    # Add verbose output if requested
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src.logging",
            "--cov=src.utils.database",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution if requested
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    # Test selection based on arguments
    if args.fast:
        # Skip performance tests for fast execution
        pytest_cmd.extend(["-m", "not performance"])
    elif args.integration:
        # Run only integration tests
        pytest_cmd.extend(["-m", "integration"])
    elif args.unit:
        # Run only unit tests
        pytest_cmd.extend(["-m", "unit or not integration"])
    
    # Additional pytest options
    pytest_cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(pytest_cmd)}")
    print("=" * 80)
    
    # Execute the tests
    try:
        result = subprocess.run(pytest_cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Activity Logging Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  Core Functionality:
    - ActivityLogger class functionality
    - Database connectivity and schema
    - Activity types and categories
    
  Integration:
    - Database integration (CRUD operations)
    - Dashboard integration (real-time feeds)
    
  Performance:
    - Batch processing performance
    - Query performance requirements
    - Memory and concurrency testing
    
  Reliability:
    - Edge cases and error handling
    - Data retention and cleanup
    - Security and audit features

Following TDD methodology, all tests are designed to FAIL initially
until the corresponding implementation is completed.
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--fast", 
        action="store_true",
        help="Run fast tests only (skip performance tests)"
    )
    test_group.add_argument(
        "--integration",
        action="store_true", 
        help="Run integration tests only"
    )
    test_group.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    
    # Execution options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Print test suite information
    print("Activity Logging Test Suite")
    print("=" * 80)
    print("Following TDD methodology - comprehensive failing tests")
    print()
    print("Test Coverage Areas:")
    print("  ✓ Database connectivity and schema setup")
    print("  ✓ ActivityLogger core functionality")
    print("  ✓ Activity types and categories (trading, system, user, ml/rl, security)")
    print("  ✓ Database integration (CRUD, querying, filtering)")
    print("  ✓ Dashboard integration (real-time feeds, activity retrieval)")
    print("  ✓ Performance requirements (batch processing, query performance)")
    print("  ✓ Edge cases and error handling")
    print("  ✓ Data retention and cleanup functionality")
    print("  ✓ Security and audit features")
    print()
    
    # Run the tests
    exit_code = run_tests(args)
    
    # Print summary
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
        print("Note: In TDD, passing tests indicate implementation is complete.")
    else:
        print("❌ Some tests failed.")
        print("Note: In TDD, failing tests are expected until implementation is complete.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())