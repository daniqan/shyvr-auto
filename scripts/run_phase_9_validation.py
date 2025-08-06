#!/usr/bin/env python3
"""
Phase 9 Validation Runner Script

This script runs the complete Phase 9 - Final Integration and Validation suite.
It executes all validation components and generates comprehensive reports.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from testing.phase_9_final_validation_report import run_complete_phase_9_validation, Phase9Status

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run complete Phase 9 validation."""
    logger.info("Starting Phase 9 - Final Integration and Validation")
    
    try:
        # Set up output directory
        output_dir = project_root / "reports" / "phase_9_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run complete validation
        report = await run_complete_phase_9_validation(str(output_dir))
        
        # Print results
        print("\n" + "="*80)
        print("PHASE 9 VALIDATION COMPLETED")
        print("="*80)
        print(f"Overall Status: {report.overall_status.value.upper()}")
        print(f"Validation Duration: {report.validation_duration:.2f}s")
        print()
        
        # Print readiness assessment
        readiness = report.deployment_readiness
        print("DEPLOYMENT READINESS:")
        print(f"  Status: {readiness.get('status', 'unknown').upper()}")
        print(f"  Score: {readiness.get('score', 0)}/3 ({readiness.get('percentage', 0):.0f}%)")
        print(f"  {readiness.get('message', 'No assessment available')}")
        print()
        
        # Print summary
        print("VALIDATION RESULTS:")
        summaries = [
            ("End-to-End", report.end_to_end_summary),
            ("Performance", report.performance_summary),
            ("Safety", report.safety_summary)
        ]
        
        for name, summary in summaries:
            status_symbol = {"success": "✅", "warning": "⚠️", "failure": "❌", "error": "❌"}.get(summary.status, "❓")
            print(f"  {status_symbol} {name}: {summary.tests_passed}/{summary.tests_total} tests passed")
        
        print()
        
        if report.critical_issues:
            print("CRITICAL ISSUES:")
            for issue in report.critical_issues:
                print(f"  {issue}")
            print()
        
        if report.key_achievements:
            print("KEY ACHIEVEMENTS:")
            for achievement in report.key_achievements:
                print(f"  {achievement}")
            print()
        
        # Determine exit status
        success = report.overall_status in [Phase9Status.SUCCESS, Phase9Status.WARNING]
        
        if success:
            print("🎉 Phase 9 validation completed successfully!")
            if report.overall_status == Phase9Status.WARNING:
                print("⚠️  Some warnings detected - review recommendations before production deployment")
        else:
            print("❌ Phase 9 validation failed!")
            print("🔧 Review critical issues and fix before proceeding")
        
        print("="*80)
        
        return success
    
    except Exception as e:
        logger.error(f"Phase 9 validation failed with error: {e}")
        print(f"\n❌ VALIDATION ERROR: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)