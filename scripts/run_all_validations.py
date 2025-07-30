#!/usr/bin/env python3
"""
Automated Validation Orchestration Script for Shyvr RLTE
Runs all validation scripts in the correct order and aggregates results
"""

import asyncio
import subprocess
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationStepResult:
    """Individual validation step result"""
    step_name: str
    script_name: str
    passed: bool
    exit_code: int
    execution_time_seconds: float
    stdout: str = ""
    stderr: str = ""
    report_file: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class ValidationOrchestrator:
    """Orchestrates all validation scripts and aggregates results"""
    
    def __init__(self, project_id: str = "shvyr-ai-bots", base_url: str = "http://localhost:8080"):
        self.project_id = project_id
        self.base_url = base_url
        self.results: List[ValidationStepResult] = []
        self.reports_dir = Path(__file__).parent.parent / "reports"
        self.scripts_dir = Path(__file__).parent
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        
        # Define validation steps in execution order
        self.validation_steps = [
            {
                "name": "Pre-deployment Validation",
                "script": "pre_deployment_validation.py",
                "description": "Comprehensive pre-deployment system validation",
                "critical": True,
                "timeout": 300  # 5 minutes
            },
            {
                "name": "Database Schema Validation", 
                "script": "validate_database_schema.py",
                "description": "Database connectivity and schema validation",
                "args": ["--project-id", self.project_id],
                "critical": True,
                "timeout": 180  # 3 minutes
            },
            {
                "name": "ML/RL Model Validation",
                "script": "validate_ml_rl_models.py", 
                "description": "ML/RL model loading and functionality validation",
                "critical": True,
                "timeout": 240  # 4 minutes
            },
            {
                "name": "Secret Validation",
                "script": "validate_secrets_comprehensive.py",
                "description": "Comprehensive secret availability and security validation",
                "args": ["--project-id", self.project_id],
                "critical": True,
                "timeout": 120  # 2 minutes
            },
            {
                "name": "API Endpoint Validation",
                "script": "validate_api_endpoints.py",
                "description": "API endpoint health and functionality validation",
                "args": ["--base-url", self.base_url],
                "critical": False,  # Non-critical if service not running
                "timeout": 180  # 3 minutes
            }
        ]
    
    def add_result(self, result: ValidationStepResult):
        """Add validation step result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        time_info = f" ({result.execution_time_seconds:.1f}s)"
        logger.info(f"{status} {result.step_name}{time_info}: {result.script_name}")
    
    async def run_validation_step(self, step_config: Dict[str, Any]) -> ValidationStepResult:
        """Run a single validation step"""
        step_name = step_config["name"]
        script_name = step_config["script"]
        args = step_config.get("args", [])
        timeout = step_config.get("timeout", 300)
        
        logger.info(f"🔍 Running {step_name}...")
        logger.info(f"📝 {step_config['description']}")
        
        # Prepare command
        script_path = self.scripts_dir / script_name
        
        # Generate report file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"{script_name.replace('.py', '')}_{timestamp}.json"
        
        # Add output file argument if not present
        cmd_args = args.copy()
        if "--output-file" not in cmd_args:
            cmd_args.extend(["--output-file", str(report_file)])
        
        command = [sys.executable, str(script_path)] + cmd_args
        
        start_time = time.time()
        try:
            # Run the validation script
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.scripts_dir.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                exit_code = process.returncode
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ {step_name} timed out after {timeout} seconds")
                process.kill()
                await process.wait()
                execution_time = time.time() - start_time
                
                return ValidationStepResult(
                    step_name=step_name,
                    script_name=script_name,
                    passed=False,
                    exit_code=-1,
                    execution_time_seconds=execution_time,
                    stdout="",
                    stderr=f"Validation timed out after {timeout} seconds",
                    report_file=str(report_file) if report_file.exists() else None
                )
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            # Determine if validation passed
            passed = exit_code == 0
            
            return ValidationStepResult(
                step_name=step_name,
                script_name=script_name,
                passed=passed,
                exit_code=exit_code,
                execution_time_seconds=execution_time,
                stdout=stdout_text,
                stderr=stderr_text,
                report_file=str(report_file) if report_file.exists() else None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"💥 {step_name} failed with exception: {str(e)}")
            
            return ValidationStepResult(
                step_name=step_name,
                script_name=script_name,
                passed=False,
                exit_code=-2,
                execution_time_seconds=execution_time,
                stdout="",
                stderr=f"Validation script failed with exception: {str(e)}",
                report_file=None
            )
    
    def load_individual_reports(self) -> Dict[str, Any]:
        """Load and aggregate individual validation reports"""
        aggregated_reports = {}
        
        for result in self.results:
            if result.report_file and Path(result.report_file).exists():
                try:
                    with open(result.report_file, 'r') as f:
                        report_data = json.load(f)
                        aggregated_reports[result.step_name] = report_data
                except Exception as e:
                    logger.warning(f"Failed to load report {result.report_file}: {str(e)}")
                    aggregated_reports[result.step_name] = {"error": f"Failed to load report: {str(e)}"}
            else:
                aggregated_reports[result.step_name] = {"error": "No report file generated"}
        
        return aggregated_reports
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Basic statistics
        total_steps = len(self.results)
        passed_steps = sum(1 for r in self.results if r.passed)
        failed_steps = total_steps - passed_steps
        success_rate = (passed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Critical step analysis
        critical_steps = [r for r in self.results if any(
            step["name"] == r.step_name and step.get("critical", False) 
            for step in self.validation_steps
        )]
        critical_passed = sum(1 for r in critical_steps if r.passed)
        critical_failed = len(critical_steps) - critical_passed
        
        # Execution time analysis
        total_execution_time = sum(r.execution_time_seconds for r in self.results)
        avg_execution_time = total_execution_time / total_steps if total_steps > 0 else 0
        
        # Load individual reports
        individual_reports = self.load_individual_reports()
        
        # Determine overall deployment readiness
        deployment_ready = (
            critical_failed == 0 and  # All critical steps must pass
            success_rate >= 80  # At least 80% overall success rate
        )
        
        summary = {
            "validation_orchestration_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": self.project_id,
            "base_url": self.base_url,
            "deployment_ready": deployment_ready,
            "execution_summary": {
                "total_steps": total_steps,
                "passed_steps": passed_steps,
                "failed_steps": failed_steps,
                "success_rate_percent": round(success_rate, 2),
                "critical_steps_total": len(critical_steps),
                "critical_steps_passed": critical_passed,
                "critical_steps_failed": critical_failed,
                "total_execution_time_seconds": round(total_execution_time, 2),
                "average_execution_time_seconds": round(avg_execution_time, 2)
            },
            "step_results": [asdict(result) for result in self.results],
            "individual_reports": individual_reports,
            "deployment_blockers": [
                asdict(r) for r in self.results 
                if not r.passed and any(
                    step["name"] == r.step_name and step.get("critical", False) 
                    for step in self.validation_steps
                )
            ],
            "recommendations": self.generate_recommendations()
        }
        
        return summary
    
    def generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations based on validation results"""
        recommendations = []
        
        # Check for critical failures
        critical_failures = [
            r for r in self.results 
            if not r.passed and any(
                step["name"] == r.step_name and step.get("critical", False) 
                for step in self.validation_steps
            )
        ]
        
        if critical_failures:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical_failures)} critical validation steps failed. "
                "Deployment should be blocked until these issues are resolved."
            )
            for failure in critical_failures:
                recommendations.append(f"  - Fix {failure.step_name}: {failure.script_name}")
        
        # Check for non-critical failures
        non_critical_failures = [
            r for r in self.results 
            if not r.passed and not any(
                step["name"] == r.step_name and step.get("critical", False) 
                for step in self.validation_steps
            )
        ]
        
        if non_critical_failures:
            recommendations.append(
                f"⚠️ {len(non_critical_failures)} non-critical validation steps failed. "
                "Consider fixing these before deployment for optimal system health."
            )
        
        # Performance recommendations
        slow_steps = [r for r in self.results if r.execution_time_seconds > 120]
        if slow_steps:
            recommendations.append(
                f"🐌 {len(slow_steps)} validation steps took longer than 2 minutes. "
                "Consider optimizing these components for better deployment speed."
            )
        
        # Success case
        if not critical_failures and not non_critical_failures:
            recommendations.append(
                "🎉 All validation steps passed! System is ready for deployment."
            )
            recommendations.append(
                "✅ Proceed with post-deployment smoke tests after deployment."
            )
        
        return recommendations
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation steps in sequence"""
        logger.info("🚀 Starting comprehensive validation orchestration")
        logger.info(f"📊 Running {len(self.validation_steps)} validation steps")
        
        start_time = time.time()
        
        # Run each validation step
        for i, step_config in enumerate(self.validation_steps, 1):
            logger.info(f"📋 Step {i}/{len(self.validation_steps)}: {step_config['name']}")
            
            step_result = await self.run_validation_step(step_config)
            self.add_result(step_result)
            
            # If this is a critical step and it failed, consider early termination
            if step_config.get("critical", False) and not step_result.passed:
                logger.error(f"💥 Critical validation step failed: {step_config['name']}")
                logger.error("🛑 Consider reviewing this failure before proceeding with remaining validations")
                # Continue with remaining validations for complete assessment
            
            # Small delay between steps
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive summary
        summary = self.generate_summary_report()
        summary["total_orchestration_time_seconds"] = round(total_time, 2)
        
        # Log summary
        passed_steps = sum(1 for r in self.results if r.passed)
        total_steps = len(self.results)
        success_rate = (passed_steps / total_steps * 100) if total_steps > 0 else 0
        
        logger.info(f"✅ Validation orchestration complete: {passed_steps}/{total_steps} steps passed ({success_rate:.1f}%)")
        logger.info(f"⏱️ Total execution time: {total_time:.1f} seconds")
        
        if summary["deployment_ready"]:
            logger.info("🎉 DEPLOYMENT READY: All critical validations passed")
        else:
            logger.error("🚨 DEPLOYMENT BLOCKED: Critical validations failed")
        
        # Print recommendations
        logger.info("📋 Recommendations:")
        for recommendation in summary["recommendations"]:
            logger.info(f"  {recommendation}")
        
        return summary

async def main():
    """Main orchestration runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all Shyvr RLTE validation scripts")
    parser.add_argument("--project-id", default="shvyr-ai-bots", help="GCP project ID")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Base URL for API validation")
    parser.add_argument("--output-file", help="Output file for orchestration results")
    parser.add_argument("--skip-api-validation", action="store_true", help="Skip API validation (if service not running)")
    
    args = parser.parse_args()
    
    orchestrator = ValidationOrchestrator(
        project_id=args.project_id,
        base_url=args.base_url
    )
    
    # Remove API validation step if requested
    if args.skip_api_validation:
        orchestrator.validation_steps = [
            step for step in orchestrator.validation_steps 
            if step["name"] != "API Endpoint Validation"
        ]
        logger.info("🔧 Skipping API endpoint validation as requested")
    
    try:
        results = await orchestrator.run_all_validations()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = orchestrator.reports_dir / f"validation_orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Orchestration report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["deployment_ready"] else 1)
        
    except Exception as e:
        logger.error(f"💥 Validation orchestration failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())