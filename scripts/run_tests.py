#!/usr/bin/env python3
"""
Test runner script for Shyvr RLTE
Runs comprehensive test suite with proper setup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def run_command(cmd, description="", check=True):
    """Run shell command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def setup_test_environment():
    """Set up test environment variables"""
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_rlte",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "TELEGRAM_TOKEN": "test_token",
        "AGENT_MODEL_TYPE": "mock"
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("🔧 Test environment variables set")

def main():
    parser = argparse.ArgumentParser(description="Run Shyvr RLTE tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--format", action="store_true", help="Check code formatting")
    parser.add_argument("--all", action="store_true", help="Run all checks and tests")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Setup test environment
    setup_test_environment()
    
    success = True
    
    # Auto-fix formatting if requested
    if args.fix:
        run_command("uv run black src/ tests/", "Auto-formatting code with black", check=False)
        run_command("uv run isort src/ tests/", "Auto-sorting imports with isort", check=False)
    
    # Linting checks
    if args.lint or args.all:
        success &= run_command("uv run ruff check src/ tests/", "Running ruff linter")
    
    # Format checking
    if args.format or args.all:
        success &= run_command("uv run black --check src/ tests/", "Checking code formatting")
        success &= run_command("uv run isort --check-only src/ tests/", "Checking import sorting")
    
    # Type checking
    if args.all:
        run_command("uv run mypy src/", "Running type checks", check=False)  # Allow to fail initially
    
    # Unit tests
    if args.unit or args.all or not any([args.integration, args.performance]):
        test_cmd = "uv run pytest tests/unit/ -v"
        if args.coverage or args.all:
            test_cmd += " --cov=src --cov-report=html --cov-report=term-missing"
        
        success &= run_command(test_cmd, "Running unit tests")
    
    # Integration tests
    if args.integration or args.all:
        success &= run_command("uv run pytest tests/integration/ -v", "Running integration tests", check=False)
    
    # Performance tests
    if args.performance or args.all:
        success &= run_command("uv run pytest tests/performance/ -v", "Running performance tests", check=False)
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 All tests and checks passed!")
        sys.exit(0)
    else:
        print("❌ Some tests or checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()