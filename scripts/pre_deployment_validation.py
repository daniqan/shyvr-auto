#!/usr/bin/env python3
"""
Pre-deployment validation script for Shyvr RLTE
Performs comprehensive validation of all system components before deployment
"""

import asyncio
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Validation result container"""
    component: str
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class PreDeploymentValidator:
    """Comprehensive pre-deployment validation system"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    def add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} {result.component}: {result.test_name} - {result.message}")
        
    async def validate_environment(self) -> List[ValidationResult]:
        """Validate environment configuration"""
        results = []
        
        # Check Python version
        import sys
        python_version = sys.version_info
        required_version = (3, 9)
        
        passed = python_version >= required_version
        results.append(ValidationResult(
            component="Environment",
            test_name="Python Version",
            passed=passed,
            message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro} {'meets' if passed else 'below'} requirement {required_version[0]}.{required_version[1]}+",
            details={"current": f"{python_version.major}.{python_version.minor}.{python_version.micro}", "required": f"{required_version[0]}.{required_version[1]}+"}
        ))
        
        # Check required directories exist
        required_dirs = ["src", "static", "config", "database", "deploy", "scripts"]
        for dir_name in required_dirs:
            dir_path = Path(__file__).parent.parent / dir_name
            passed = dir_path.exists() and dir_path.is_dir()
            results.append(ValidationResult(
                component="Environment",
                test_name=f"Directory {dir_name}",
                passed=passed,
                message=f"Directory {dir_name} {'exists' if passed else 'missing'}",
                details={"path": str(dir_path)}
            ))
            
        # Check critical files exist
        critical_files = [
            "main.py",
            "config/config.yaml", 
            "Dockerfile",
            "pyproject.toml"
        ]
        for file_path in critical_files:
            full_path = Path(__file__).parent.parent / file_path
            passed = full_path.exists() and full_path.is_file()
            results.append(ValidationResult(
                component="Environment",
                test_name=f"File {file_path}",
                passed=passed,
                message=f"File {file_path} {'exists' if passed else 'missing'}",
                details={"path": str(full_path)}
            ))
            
        return results
    
    async def validate_configuration(self) -> List[ValidationResult]:
        """Validate configuration loading and structure"""
        results = []
        
        try:
            from src.utils.config import init_config, get_config
            
            # Test configuration loading
            start_time = time.time()
            config = init_config()
            load_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component="Configuration",
                test_name="Config Loading",
                passed=True,
                message="Configuration loaded successfully",
                execution_time_ms=load_time,
                details={"environment": config.app.environment if hasattr(config, 'app') else "unknown"}
            ))
            
            # Validate critical configuration sections
            critical_sections = ["app", "database", "trading", "ml", "rl"]
            for section in critical_sections:
                has_section = hasattr(config, section)
                results.append(ValidationResult(
                    component="Configuration",
                    test_name=f"Section {section}",
                    passed=has_section,
                    message=f"Configuration section {section} {'present' if has_section else 'missing'}",
                    details={"section": section}
                ))
            
            # Validate trading mode safety
            if hasattr(config, 'trading') and hasattr(config.trading, 'modes'):
                live_mode_enabled = getattr(config.trading.modes, 'live', True)  # Default to True for safety check
                results.append(ValidationResult(
                    component="Configuration",
                    test_name="Live Trading Safety",
                    passed=not live_mode_enabled,  # Pass if live trading is disabled
                    message=f"Live trading mode is {'ENABLED (DANGEROUS)' if live_mode_enabled else 'safely disabled'}",
                    details={"live_trading": live_mode_enabled}
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="Configuration",
                test_name="Config Loading",
                passed=False,
                message=f"Configuration loading failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_dependencies(self) -> List[ValidationResult]:
        """Validate Python dependencies"""
        results = []
        
        critical_dependencies = [
            "fastapi", "uvicorn", "asyncpg", "pydantic", "structlog",
            "prometheus_client", "numpy", "pandas", "torch", "transformers",
            "google-cloud-secret-manager", "solana", "eth_account"
        ]
        
        for dep in critical_dependencies:
            try:
                start_time = time.time()
                __import__(dep.replace("-", "_"))
                import_time = (time.time() - start_time) * 1000
                
                results.append(ValidationResult(
                    component="Dependencies",
                    test_name=f"Import {dep}",
                    passed=True,
                    message=f"Successfully imported {dep}",
                    execution_time_ms=import_time
                ))
            except ImportError as e:
                results.append(ValidationResult(
                    component="Dependencies",
                    test_name=f"Import {dep}",
                    passed=False,
                    message=f"Failed to import {dep}: {str(e)}",
                    details={"dependency": dep, "error": str(e)}
                ))
                
        return results
    
    async def validate_database_readiness(self) -> List[ValidationResult]:
        """Validate database connectivity and schema"""
        results = []
        
        try:
            from src.utils.database import check_database_health
            
            start_time = time.time()
            health_status = await check_database_health()
            check_time = (time.time() - start_time) * 1000
            
            is_healthy = health_status.get("status") == "healthy"
            results.append(ValidationResult(
                component="Database",
                test_name="Connectivity",
                passed=is_healthy,
                message=f"Database health check {'passed' if is_healthy else 'failed'}",
                execution_time_ms=check_time,
                details=health_status
            ))
            
            # Check specific database components
            if "tables_exist" in health_status:
                tables_exist = health_status["tables_exist"]
                results.append(ValidationResult(
                    component="Database",
                    test_name="Schema Tables",
                    passed=tables_exist,
                    message=f"Database tables {'exist' if tables_exist else 'missing'}",
                    details={"tables_status": health_status.get("table_status", {})}
                ))
                
            # Check RL experience storage
            if "rl_experience_tables" in health_status:
                rl_tables = health_status["rl_experience_tables"]
                results.append(ValidationResult(
                    component="Database",
                    test_name="RL Experience Tables",
                    passed=rl_tables,
                    message=f"RL experience tables {'ready' if rl_tables else 'not ready'}",
                    details={"rl_tables": health_status.get("rl_table_details", {})}
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="Database",  
                test_name="Connectivity",
                passed=False,
                message=f"Database validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_secrets_availability(self) -> List[ValidationResult]:
        """Validate secret availability"""
        results = []
        
        try:
            # Import the existing secret validation
            sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))
            from validate_secrets import validate_secrets
            
            start_time = time.time()
            secret_results = validate_secrets("shvyr-ai-bots")  # Default project
            validation_time = (time.time() - start_time) * 1000
            
            # Analyze secret validation results
            required_secrets = ["TELEGRAM_TOKEN", "WEBHOOK_SECRET", "DB_PASSWORD", "DATABASE_URL"]
            required_available = sum(1 for secret in required_secrets if secret_results.get(secret, False))
            
            all_required_available = required_available == len(required_secrets)
            results.append(ValidationResult(
                component="Secrets",
                test_name="Required Secrets",
                passed=all_required_available,
                message=f"Required secrets: {required_available}/{len(required_secrets)} available",
                execution_time_ms=validation_time,
                details={"secrets": secret_results, "required_count": len(required_secrets), "available_count": required_available}
            ))
            
            # Check optional API keys
            optional_secrets = ["HELIUS_API_KEY", "BIRDEYE_API_KEY", "ETHERSCAN_API_KEY"]
            optional_available = sum(1 for secret in optional_secrets if secret_results.get(secret, False))
            
            results.append(ValidationResult(
                component="Secrets",
                test_name="Optional API Keys",
                passed=True,  # Optional secrets don't fail validation
                message=f"Optional API keys: {optional_available}/{len(optional_secrets)} available",
                details={"optional_secrets": {k: v for k, v in secret_results.items() if k in optional_secrets}}
            ))
            
            # Check trading secrets (should be missing for safety)
            trading_secrets = ["SOLANA_PRIVATE_KEY", "ETHEREUM_PRIVATE_KEY", "HYPERLIQUID_PRIVATE_KEY"]
            trading_available = sum(1 for secret in trading_secrets if secret_results.get(secret, False))
            
            # For pre-deployment, trading secrets should ideally be missing unless live trading is intended
            results.append(ValidationResult(
                component="Secrets",
                test_name="Trading Secrets Safety",
                passed=trading_available == 0,  # Pass if no trading secrets (safer)
                message=f"Trading secrets: {trading_available}/{len(trading_secrets)} configured {'(LIVE TRADING ENABLED)' if trading_available > 0 else '(safe)'}",
                details={"trading_secrets": {k: v for k, v in secret_results.items() if k in trading_secrets}}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component="Secrets",
                test_name="Secret Validation",
                passed=False,
                message=f"Secret validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_ml_models(self) -> List[ValidationResult]:
        """Validate ML model loading and readiness"""
        results = []
        
        try:
            from src.ml_analysis.model_manager import ModelManager
            from src.utils.config import get_config
            
            config = get_config()
            model_manager = ModelManager(config.dict() if hasattr(config, 'dict') else None)
            
            start_time = time.time()
            health_status = await model_manager.health_check()
            check_time = (time.time() - start_time) * 1000
            
            is_healthy = health_status.get("overall_healthy", False)
            results.append(ValidationResult(
                component="ML Models",
                test_name="Model Manager Health",
                passed=is_healthy,
                message=f"ML model manager {'healthy' if is_healthy else 'unhealthy'}",
                execution_time_ms=check_time,
                details=health_status
            ))
            
            # Check individual models
            models = health_status.get("models", {})
            for model_name, model_status in models.items():
                model_healthy = model_status.get("status") == "ready"
                results.append(ValidationResult(
                    component="ML Models",
                    test_name=f"Model {model_name}",
                    passed=model_healthy,
                    message=f"Model {model_name} is {'ready' if model_healthy else 'not ready'}",
                    details=model_status
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="ML Models",
                test_name="Model Loading",
                passed=False,
                message=f"ML model validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_rl_agent(self) -> List[ValidationResult]:
        """Validate RL agent initialization"""
        results = []
        
        try:
            from src.rl_agent.base import AgentConfig, ModelType
            from src.rl_agent.dqn_agent import DQNTradingAgent
            
            # Create test agent configuration
            agent_config = AgentConfig(
                model_type=ModelType.DQN,
                learning_rate=0.001,
                batch_size=32,
                replay_buffer_size=10000,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=1000,
                target_update_frequency=100,
                max_position_size=0.1,
                max_daily_loss=0.05
            )
            
            start_time = time.time()
            agent = DQNTradingAgent(agent_config)
            init_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component="RL Agent",
                test_name="Agent Initialization",
                passed=True,
                message="RL agent initialized successfully",
                execution_time_ms=init_time,
                details={"config": asdict(agent_config)}
            ))
            
            # Check agent health
            start_time = time.time()
            health_status = await agent.health_check()
            health_time = (time.time() - start_time) * 1000
            
            meets_targets = health_status.get("meets_targets", False)
            results.append(ValidationResult(
                component="RL Agent",
                test_name="Agent Health Check",
                passed=True,  # Don't fail on untrained agent
                message=f"Agent health check completed - {'meets performance targets' if meets_targets else 'requires training'}",
                execution_time_ms=health_time,
                details=health_status
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component="RL Agent",
                test_name="Agent Validation",
                passed=False,
                message=f"RL agent validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_api_endpoints(self) -> List[ValidationResult]:
        """Validate API endpoint structure (without server)"""
        results = []
        
        try:
            # Import FastAPI app
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from main import app
            
            # Check that app is properly configured
            results.append(ValidationResult(
                component="API",
                test_name="FastAPI App",
                passed=app is not None,
                message="FastAPI application configured",
                details={"title": app.title, "version": app.version}
            ))
            
            # Check routes are registered
            routes = [route.path for route in app.routes]
            critical_routes = ["/", "/health", "/metrics", "/api"]
            
            for route in critical_routes:
                route_exists = any(r.startswith(route) for r in routes)
                results.append(ValidationResult(
                    component="API", 
                    test_name=f"Route {route}",
                    passed=route_exists,
                    message=f"Route {route} {'registered' if route_exists else 'missing'}",
                    details={"route": route}
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component="API",
                test_name="API Structure",
                passed=False,
                message=f"API validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_monitoring_setup(self) -> List[ValidationResult]:
        """Validate monitoring and metrics setup"""
        results = []
        
        try:
            from src.monitoring.base import MetricsRegistry
            from src.monitoring.trading_metrics import TradingMetricsCollector
            from src.monitoring.safety_metrics import SafetyMetricsCollector
            
            # Test metrics registry
            registry = MetricsRegistry()
            results.append(ValidationResult(
                component="Monitoring",
                test_name="Metrics Registry",
                passed=True,
                message="Metrics registry initialized",
                details={"registry_type": type(registry).__name__}
            ))
            
            # Test metrics collectors
            trading_metrics = TradingMetricsCollector(registry)
            safety_metrics = SafetyMetricsCollector(registry)
            
            results.append(ValidationResult(
                component="Monitoring",
                test_name="Metrics Collectors",
                passed=True,
                message="Metrics collectors initialized",
                details={"collectors": ["TradingMetricsCollector", "SafetyMetricsCollector"]}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component="Monitoring",
                test_name="Monitoring Setup",
                passed=False,
                message=f"Monitoring validation failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all pre-deployment validations"""
        logger.info("🚀 Starting pre-deployment validation")
        
        # Run all validation categories
        validation_tasks = [
            ("Environment", self.validate_environment()),
            ("Configuration", self.validate_configuration()),
            ("Dependencies", self.validate_dependencies()),
            ("Database", self.validate_database_readiness()),
            ("Secrets", self.validate_secrets_availability()),
            ("ML Models", self.validate_ml_models()),
            ("RL Agent", self.validate_rl_agent()),
            ("API", self.validate_api_endpoints()),
            ("Monitoring", self.validate_monitoring_setup())
        ]
        
        all_results = []
        for category, task in validation_tasks:
            logger.info(f"🔍 Validating {category}...")
            try:
                category_results = await task
                all_results.extend(category_results)
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                error_result = ValidationResult(
                    component=category,
                    test_name="Category Validation",
                    passed=False,
                    message=f"Category validation failed: {str(e)}",
                    details={"error": str(e)}
                )
                all_results.append(error_result)
                self.add_result(error_result)
        
        # Generate summary
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate execution time
        total_time = time.time() - self.start_time
        
        # Categorize results
        results_by_component = {}
        for result in all_results:
            if result.component not in results_by_component:
                results_by_component[result.component] = []
            results_by_component[result.component].append(asdict(result))
        
        summary = {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time_seconds": total_time,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2)
            },
            "deployment_ready": failed_tests == 0,
            "results_by_component": results_by_component,
            "critical_issues": [
                asdict(r) for r in all_results 
                if not r.passed and r.component in ["Database", "Configuration", "Secrets"]
            ]
        }
        
        # Log summary
        logger.info(f"✅ Validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} tests failed - review required before deployment")
        else:
            logger.info("🎉 All validations passed - system ready for deployment")
            
        return summary

async def main():
    """Main validation runner"""
    validator = PreDeploymentValidator()
    
    try:
        results = await validator.run_all_validations()
        
        # Save results to file
        output_file = Path(__file__).parent.parent / "reports" / f"pre_deployment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"📄 Validation report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["deployment_ready"] else 1)
        
    except Exception as e:
        logger.error(f"💥 Validation failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())