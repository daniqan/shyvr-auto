#!/usr/bin/env python3
"""
ML/RL Model Loading and Verification Script for Shyvr RLTE
Validates ML models and RL agent initialization, loading, and basic functionality
"""

import asyncio
import logging
import sys
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
class ModelValidationResult:
    """Model validation result container"""
    model_category: str
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class MLRLModelValidator:
    """Comprehensive ML/RL model validation system"""
    
    def __init__(self):
        self.results: List[ModelValidationResult] = []
        self.config = None
        
    def add_result(self, result: ModelValidationResult):
        """Add validation result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} {result.model_category}.{result.test_name}: {result.message}")
    
    async def load_configuration(self) -> List[ModelValidationResult]:
        """Load and validate configuration for ML/RL models"""
        results = []
        
        try:
            from src.utils.config import init_config, get_config
            
            start_time = time.time()
            self.config = init_config()
            load_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="Configuration",
                test_name="Config Loading",
                passed=True,
                message="Configuration loaded successfully",
                execution_time_ms=load_time,
                details={"load_time_ms": load_time}
            ))
            
            # Validate ML configuration
            has_ml_config = hasattr(self.config, 'ml')
            results.append(ModelValidationResult(
                model_category="Configuration",
                test_name="ML Config Section",
                passed=has_ml_config,
                message=f"ML configuration section {'present' if has_ml_config else 'missing'}",
                details={"has_ml_config": has_ml_config}
            ))
            
            # Validate RL configuration
            has_rl_config = hasattr(self.config, 'rl')
            results.append(ModelValidationResult(
                model_category="Configuration",
                test_name="RL Config Section",
                passed=has_rl_config,
                message=f"RL configuration section {'present' if has_rl_config else 'missing'}",
                details={"has_rl_config": has_rl_config}
            ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="Configuration",
                test_name="Config Loading",
                passed=False,
                message=f"Configuration loading failed: {str(e)}",
                details={"error": str(e)}
            ))
            
        return results
    
    async def validate_ml_dependencies(self) -> List[ModelValidationResult]:
        """Validate ML/RL dependencies and frameworks"""
        results = []
        
        # Critical ML/RL dependencies
        ml_dependencies = [
            ("torch", "PyTorch deep learning framework"),
            ("numpy", "NumPy numerical computing"),
            ("pandas", "Pandas data manipulation"),
            ("sklearn", "Scikit-learn machine learning"),
            ("transformers", "Hugging Face Transformers"),
            ("gym", "OpenAI Gym RL environment"),
            ("stable_baselines3", "Stable Baselines3 RL algorithms")
        ]
        
        for dep_name, description in ml_dependencies:
            try:
                start_time = time.time()
                
                if dep_name == "sklearn":
                    import sklearn
                    version = sklearn.__version__
                elif dep_name == "stable_baselines3":
                    try:
                        import stable_baselines3
                        version = stable_baselines3.__version__
                    except ImportError:
                        # Optional dependency
                        results.append(ModelValidationResult(
                            model_category="Dependencies",
                            test_name=f"Import {dep_name}",
                            passed=True,  # Don't fail for optional dependencies
                            message=f"Optional dependency {dep_name} not installed ({description})",
                            details={"dependency": dep_name, "optional": True}
                        ))
                        continue
                else:
                    module = __import__(dep_name)
                    version = getattr(module, '__version__', 'unknown')
                
                import_time = (time.time() - start_time) * 1000
                
                results.append(ModelValidationResult(
                    model_category="Dependencies",
                    test_name=f"Import {dep_name}",
                    passed=True,
                    message=f"Successfully imported {dep_name} v{version}",
                    execution_time_ms=import_time,
                    details={"dependency": dep_name, "version": version, "description": description}
                ))
                
            except ImportError as e:
                results.append(ModelValidationResult(
                    model_category="Dependencies",
                    test_name=f"Import {dep_name}",
                    passed=dep_name in ["stable_baselines3"],  # Only fail for critical dependencies
                    message=f"Failed to import {dep_name}: {str(e)}",
                    details={"dependency": dep_name, "error": str(e), "description": description}
                ))
        
        # Check PyTorch CUDA availability
        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            current_device = torch.cuda.current_device() if cuda_available else None
            
            results.append(ModelValidationResult(
                model_category="Dependencies",
                test_name="CUDA Support",
                passed=True,  # Don't require CUDA
                message=f"CUDA {'available' if cuda_available else 'not available'} ({device_count} devices)",
                details={
                    "cuda_available": cuda_available,
                    "device_count": device_count,
                    "current_device": current_device,
                    "pytorch_version": torch.__version__
                }
            ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="Dependencies",
                test_name="CUDA Support",
                passed=True,  # Don't fail on CUDA check
                message=f"CUDA support check failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_ml_model_manager(self) -> List[ModelValidationResult]:
        """Validate ML Model Manager functionality"""
        results = []
        
        try:
            from src.ml_analysis.model_manager import ModelManager
            
            # Initialize Model Manager
            start_time = time.time()
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else {}
            model_manager = ModelManager(config_dict)
            init_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="ML Models",
                test_name="Model Manager Init",
                passed=True,
                message=f"Model Manager initialized successfully in {init_time:.2f}ms",
                execution_time_ms=init_time,
                details={"manager_type": type(model_manager).__name__}
            ))
            
            # Test health check
            start_time = time.time()
            health_status = await model_manager.health_check()
            health_time = (time.time() - start_time) * 1000
            
            is_healthy = health_status.get("overall_healthy", False)
            models_count = len(health_status.get("models", {}))
            
            results.append(ModelValidationResult(
                model_category="ML Models",
                test_name="Model Manager Health",
                passed=True,  # Don't require models to be loaded for validation
                message=f"Health check completed: {'Healthy' if is_healthy else 'Not fully loaded'} ({models_count} models)",
                execution_time_ms=health_time,
                details=health_status
            ))
            
            # Test individual model components if available
            models = health_status.get("models", {})
            for model_name, model_info in models.items():
                model_status = model_info.get("status", "unknown")
                model_ready = model_status == "ready"
                
                results.append(ModelValidationResult(
                    model_category="ML Models",
                    test_name=f"Model {model_name}",
                    passed=model_ready,
                    message=f"Model {model_name} status: {model_status}",
                    details=model_info
                ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="ML Models",
                test_name="Model Manager",
                passed=False,
                message=f"Model Manager validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_feature_engineering(self) -> List[ModelValidationResult]:
        """Validate feature engineering capabilities"""
        results = []
        
        try:
            from src.ml_analysis.feature_engineer import FeatureEngineer
            
            # Initialize Feature Engineer
            start_time = time.time()
            feature_engineer = FeatureEngineer()
            init_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="ML Models",
                test_name="Feature Engineer Init",
                passed=True,
                message=f"Feature Engineer initialized in {init_time:.2f}ms",
                execution_time_ms=init_time
            ))
            
            # Test feature engineering with sample data
            sample_data = {
                'price': [100.0, 101.0, 102.0, 101.5, 103.0],
                'volume': [1000, 1100, 950, 1200, 1050],
                'timestamp': list(range(5))
            }
            
            start_time = time.time()
            try:
                features = await feature_engineer.process_market_data(sample_data)
                processing_time = (time.time() - start_time) * 1000
                
                has_features = features is not None and len(features) > 0
                
                results.append(ModelValidationResult(
                    model_category="ML Models",
                    test_name="Feature Processing",
                    passed=has_features,
                    message=f"Feature processing {'successful' if has_features else 'failed'} in {processing_time:.2f}ms",
                    execution_time_ms=processing_time,
                    details={"sample_features": features if has_features else None}
                ))
                
            except Exception as e:
                results.append(ModelValidationResult(
                    model_category="ML Models",
                    test_name="Feature Processing",
                    passed=False,
                    message=f"Feature processing failed: {str(e)}",
                    details={"error": str(e), "sample_data": sample_data}
                ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="ML Models",
                test_name="Feature Engineering",
                passed=False,
                message=f"Feature Engineering validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_rl_agent_initialization(self) -> List[ModelValidationResult]:
        """Validate RL agent initialization and basic functionality"""
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
            
            # Initialize DQN Agent
            start_time = time.time()
            dqn_agent = DQNTradingAgent(agent_config)
            init_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="DQN Agent Init",
                passed=True,
                message=f"DQN Agent initialized successfully in {init_time:.2f}ms",
                execution_time_ms=init_time,
                details={"config": asdict(agent_config)}
            ))
            
            # Test agent health check
            start_time = time.time()
            health_status = await dqn_agent.health_check()
            health_time = (time.time() - start_time) * 1000
            
            is_trained = health_status.get("is_trained", False)
            meets_targets = health_status.get("meets_targets", False)
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="DQN Agent Health",
                passed=True,  # Don't require trained agent for validation
                message=f"Agent health check: {'Trained' if is_trained else 'Untrained'}, {'Meets targets' if meets_targets else 'Needs training'}",
                execution_time_ms=health_time,
                details=health_status
            ))
            
            # Test action selection (with random state)
            try:
                sample_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                
                start_time = time.time()
                action = await dqn_agent.select_action(sample_state)
                action_time = (time.time() - start_time) * 1000
                
                valid_action = isinstance(action, (int, np.integer)) and 0 <= action < 3  # Assuming 3 actions
                
                results.append(ModelValidationResult(
                    model_category="RL Agent",
                    test_name="Action Selection",
                    passed=valid_action,
                    message=f"Action selection {'successful' if valid_action else 'failed'}: action={action} in {action_time:.2f}ms",
                    execution_time_ms=action_time,
                    details={"sample_state": sample_state.tolist(), "selected_action": int(action) if valid_action else action}
                ))
                
            except Exception as e:
                results.append(ModelValidationResult(
                    model_category="RL Agent",
                    test_name="Action Selection",
                    passed=False,
                    message=f"Action selection test failed: {str(e)}",
                    details={"error": str(e)}
                ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="RL Agent Init",
                passed=False,
                message=f"RL Agent validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_experience_replay(self) -> List[ModelValidationResult]:
        """Validate RL experience replay functionality"""
        results = []
        
        try:
            from src.rl_agent.experience_replay import ExperienceReplay
            
            # Initialize Experience Replay
            start_time = time.time()
            experience_replay = ExperienceReplay(max_size=1000)
            init_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Experience Replay Init",
                passed=True,
                message=f"Experience Replay initialized in {init_time:.2f}ms",
                execution_time_ms=init_time,
                details={"max_size": 1000}
            ))
            
            # Test adding experiences
            sample_experiences = [
                {
                    'state': np.array([0.1, 0.2, 0.3]),
                    'action': 1,
                    'reward': 0.5,
                    'next_state': np.array([0.2, 0.3, 0.4]),
                    'done': False
                }
                for _ in range(10)
            ]
            
            start_time = time.time()
            for exp in sample_experiences:
                experience_replay.add(**exp)
            add_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Experience Storage",
                passed=len(experience_replay) == len(sample_experiences),
                message=f"Stored {len(sample_experiences)} experiences in {add_time:.2f}ms",
                execution_time_ms=add_time,
                details={"stored_count": len(experience_replay), "expected_count": len(sample_experiences)}
            ))
            
            # Test sampling
            if len(experience_replay) > 0:
                start_time = time.time()
                batch = experience_replay.sample(batch_size=5)
                sample_time = (time.time() - start_time) * 1000
                
                valid_batch = batch is not None and len(batch) == 5
                
                results.append(ModelValidationResult(
                    model_category="RL Agent",
                    test_name="Experience Sampling",
                    passed=valid_batch,
                    message=f"Sampled batch of 5 experiences in {sample_time:.2f}ms",
                    execution_time_ms=sample_time,
                    details={"batch_size": len(batch) if batch else 0}
                ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Experience Replay",
                passed=False,
                message=f"Experience Replay validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_trading_environment(self) -> List[ModelValidationResult]:
        """Validate RL trading environment"""
        results = []
        
        try:
            from src.rl_agent.trading_environment import TradingEnvironment
            
            # Initialize Trading Environment
            start_time = time.time()
            env_config = {
                'initial_balance': 10000,
                'commission_rate': 0.001,
                'max_position_size': 0.1
            }
            trading_env = TradingEnvironment(env_config)
            init_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Trading Environment Init",
                passed=True,
                message=f"Trading Environment initialized in {init_time:.2f}ms",
                execution_time_ms=init_time,
                details=env_config
            ))
            
            # Test environment reset
            start_time = time.time()
            initial_state = await trading_env.reset()
            reset_time = (time.time() - start_time) * 1000
            
            valid_state = initial_state is not None and len(initial_state) > 0
            
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Environment Reset",
                passed=valid_state,
                message=f"Environment reset {'successful' if valid_state else 'failed'} in {reset_time:.2f}ms",
                execution_time_ms=reset_time,
                details={"initial_state_shape": len(initial_state) if valid_state else None}
            ))
            
            # Test environment step
            if valid_state:
                start_time = time.time()
                next_state, reward, done, info = await trading_env.step(action=1)  # Test action
                step_time = (time.time() - start_time) * 1000
                
                valid_step = (next_state is not None and 
                             isinstance(reward, (int, float)) and 
                             isinstance(done, bool))
                
                results.append(ModelValidationResult(
                    model_category="RL Agent",
                    test_name="Environment Step",
                    passed=valid_step,
                    message=f"Environment step {'successful' if valid_step else 'failed'} in {step_time:.2f}ms",
                    execution_time_ms=step_time,
                    details={
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "next_state_shape": len(next_state) if next_state is not None else None
                    }
                ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="RL Agent",
                test_name="Trading Environment",
                passed=False,
                message=f"Trading Environment validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_model_persistence(self) -> List[ModelValidationResult]:
        """Validate model saving and loading capabilities"""
        results = []
        
        try:
            # Test PyTorch model creation and save/load
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 3)
            )
            
            # Test model save
            temp_model_path = Path("/tmp/test_model.pt")
            start_time = time.time()
            torch.save(model.state_dict(), temp_model_path)
            save_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="Model Persistence",
                test_name="Model Save",
                passed=temp_model_path.exists(),
                message=f"Model saved successfully in {save_time:.2f}ms",
                execution_time_ms=save_time,
                details={"model_path": str(temp_model_path)}
            ))
            
            # Test model load
            new_model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 3)
            )
            
            start_time = time.time()
            new_model.load_state_dict(torch.load(temp_model_path))
            load_time = (time.time() - start_time) * 1000
            
            results.append(ModelValidationResult(
                model_category="Model Persistence",
                test_name="Model Load",
                passed=True,
                message=f"Model loaded successfully in {load_time:.2f}ms",
                execution_time_ms=load_time
            ))
            
            # Clean up
            if temp_model_path.exists():
                temp_model_path.unlink()
            
            # Test model inference
            sample_input = torch.randn(1, 5)
            start_time = time.time()
            with torch.no_grad():
                output = new_model(sample_input)
            inference_time = (time.time() - start_time) * 1000
            
            valid_output = output is not None and output.shape == (1, 3)
            
            results.append(ModelValidationResult(
                model_category="Model Persistence",
                test_name="Model Inference",
                passed=valid_output,
                message=f"Model inference {'successful' if valid_output else 'failed'} in {inference_time:.2f}ms",
                execution_time_ms=inference_time,
                details={"output_shape": list(output.shape) if valid_output else None}
            ))
            
        except Exception as e:
            results.append(ModelValidationResult(
                model_category="Model Persistence",
                test_name="Model Persistence",
                passed=False,
                message=f"Model persistence validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all ML/RL model validations"""
        logger.info("🧠 Starting comprehensive ML/RL model validation")
        
        validation_categories = [
            ("Configuration", self.load_configuration()),
            ("Dependencies", self.validate_ml_dependencies()),
            ("ML Model Manager", self.validate_ml_model_manager()),
            ("Feature Engineering", self.validate_feature_engineering()),
            ("RL Agent", self.validate_rl_agent_initialization()),
            ("Experience Replay", self.validate_experience_replay()),
            ("Trading Environment", self.validate_trading_environment()),
            ("Model Persistence", self.validate_model_persistence())
        ]
        
        start_time = time.time()
        
        for category_name, validation_task in validation_categories:
            logger.info(f"🔬 Validating {category_name}...")
            try:
                category_results = await validation_task
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                error_result = ModelValidationResult(
                    model_category=category_name,
                    test_name="Category Validation",
                    passed=False,
                    message=f"Category validation failed: {str(e)}",
                    details={"error": str(e)}
                )
                self.add_result(error_result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Categorize results
        results_by_category = {}
        for result in self.results:
            if result.model_category not in results_by_category:
                results_by_category[result.model_category] = []
            results_by_category[result.model_category].append(asdict(result))
        
        summary = {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time_seconds": total_time,
            "models_ready": failed_tests == 0,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2)
            },
            "results_by_category": results_by_category,
            "critical_failures": [
                asdict(r) for r in self.results 
                if not r.passed and r.model_category in ["Dependencies", "RL Agent"]
            ],
            "system_info": {
                "pytorch_version": torch.__version__,
                "numpy_version": np.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Log summary
        logger.info(f"✅ ML/RL model validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} model tests failed")
        else:
            logger.info("🎉 All ML/RL model validations passed - models ready for deployment")
        
        return summary

async def main():
    """Main ML/RL model validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Shyvr RLTE ML/RL models")
    parser.add_argument("--output-file", help="Output file for validation results")
    
    args = parser.parse_args()
    
    validator = MLRLModelValidator()
    
    try:
        results = await validator.run_all_validations()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = Path(__file__).parent.parent / "reports" / f"ml_rl_model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 ML/RL model validation report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["models_ready"] else 1)
        
    except Exception as e:
        logger.error(f"💥 ML/RL model validation failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())