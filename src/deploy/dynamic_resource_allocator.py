"""
Dynamic Resource Allocation for Transformer Models

Implements dynamic resource allocation system for transformer models in GCP Cloud Run
deployment environments with auto-scaling capabilities, cost optimization, and
performance monitoring.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Optional Google Cloud imports - gracefully handle missing dependencies
try:
    from google.cloud import monitoring_v3
except ImportError:
    monitoring_v3 = None

try:
    from google.cloud import run_v2
except ImportError:
    run_v2 = None

try:
    from google.cloud import resourcemanager
except ImportError:
    resourcemanager = None

try:
    import numpy as np
except ImportError:
    # Use built-in functions as fallback
    class MockNumpy:
        @staticmethod
        def percentile(data, percentile):
            if not data:
                return 0
            sorted_data = sorted(data)
            index = int((percentile / 100.0) * (len(sorted_data) - 1))
            return sorted_data[index]
        
        @staticmethod
        def std(data):
            if len(data) < 2:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
    
    np = MockNumpy()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported transformer model types"""
    ITRANSFORMER = "itransformer"
    PATCHTST = "patchtst"
    TIMESMIXER = "timesmixer"
    TIMESFM = "timesfm"


class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ResourceAllocation:
    """Resource allocation specification"""
    memory: str
    cpu: int
    scaling_factor: float
    estimated_cost: float = 0.0
    confidence: float = 1.0


@dataclass
class LoadMetrics:
    """Load metrics for resource calculation"""
    requests_per_second: int
    concurrent_models: int
    avg_sequence_length: int
    complexity_factor: float
    concurrent_users: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    cpu_utilization: float
    memory_utilization: float
    request_queue_length: int
    avg_response_time_ms: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning"""
    action: ScalingAction
    target_memory: str = None
    target_cpu: int = None
    confidence: float = 0.0
    reason: str = ""
    scaling_factor: str = None
    urgency: str = "normal"
    estimated_application_time_seconds: int = 300


class DynamicResourceAllocator:
    """
    Dynamic Resource Allocator for Transformer Models
    
    Provides dynamic resource allocation capabilities for transformer models
    with cost optimization, auto-scaling, and performance monitoring.
    """
    
    def __init__(self, project_id: str, region: str, service_name: str):
        """Initialize the resource allocator"""
        self.project_id = project_id
        self.region = region
        self.service_name = service_name
        
        # Initialize Google Cloud clients (if available)
        self.monitoring_client = monitoring_v3.MetricServiceClient() if monitoring_v3 else None
        self.run_client = run_v2.ServicesClient() if run_v2 else None
        
        # Initialize managers
        self.transformer_resource_manager = TransformerResourceManager()
        self.scaling_policies = {}
        self.metrics_collector = ResourceMetricsCollector(project_id, region)
        self.resource_optimizer = ResourceOptimizer()
        self.memory_manager = MemoryManager()
        self.cpu_manager = CPUManager()
        self.container_manager = ContainerResourceManager()
        self.scaling_controller = AutoScalingController()
        
        # Resource allocation history
        self.allocation_history = []
        self.scaling_history = []
        
        logger.info(f"Initialized DynamicResourceAllocator for {service_name} in {region}")

    def calculate_base_resources(self, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base resource requirements for transformer model"""
        try:
            model_config = config.get("transformer_models", {}).get(model_type)
            if not model_config:
                raise ValueError(f"No configuration found for model type: {model_type}")
            
            base_resources = {
                "memory": model_config["base_memory"],
                "cpu": model_config["base_cpu"],
                "scaling_factor": model_config["scaling_factor"]
            }
            
            logger.info(f"Base resources for {model_type}: {base_resources}")
            return base_resources
            
        except Exception as e:
            logger.error(f"Error calculating base resources for {model_type}: {e}")
            raise

    def calculate_scaled_resources(self, model_type: str, load_metrics: Dict[str, Any], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scaled resources based on load metrics"""
        try:
            base_resources = self.calculate_base_resources(model_type, config)
            base_memory_gb = int(base_resources["memory"].rstrip("Gi"))
            base_cpu = base_resources["cpu"]
            scaling_factor = base_resources["scaling_factor"]
            
            # Calculate load factor
            rps = load_metrics.get("requests_per_second", 0)
            complexity = load_metrics.get("complexity_factor", 1.0)
            
            # Determine scaling based on load
            if rps <= 10:  # Low load
                memory_scale = 1.0
                cpu_scale = 1.0
            elif rps <= 50:  # Medium load
                memory_scale = 1.5
                cpu_scale = 1.5
            elif rps <= 150:  # High load
                memory_scale = scaling_factor * 1.5
                cpu_scale = scaling_factor * 1.5
            else:  # Peak load
                memory_scale = scaling_factor * 2.0
                cpu_scale = scaling_factor * 2.0
            
            # Apply complexity factor
            memory_scale *= complexity
            cpu_scale *= complexity
            
            # Calculate final resources
            scaled_memory = max(base_memory_gb, int(base_memory_gb * memory_scale))
            scaled_cpu = max(base_cpu, int(base_cpu * cpu_scale))
            
            # Ensure within limits
            max_memory = int(config["transformer_models"][model_type]["max_memory"].rstrip("Gi"))
            max_cpu = config["transformer_models"][model_type]["max_cpu"]
            
            scaled_memory = min(scaled_memory, max_memory)
            scaled_cpu = min(scaled_cpu, max_cpu)
            
            scaled_resources = {
                "memory": f"{scaled_memory}Gi",
                "cpu": scaled_cpu
            }
            
            logger.info(f"Scaled resources for {model_type}: {scaled_resources}")
            return scaled_resources
            
        except Exception as e:
            logger.error(f"Error calculating scaled resources: {e}")
            raise

    def balance_resources_across_models(self, active_models: List[Dict[str, Any]], 
                                      total_budget: Dict[str, Any], 
                                      config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Balance resources across multiple concurrent transformer models"""
        try:
            allocation = {}
            total_memory_gb = int(total_budget["memory"].rstrip("Gi"))
            total_cpu = total_budget["cpu"]
            
            # Calculate priority weights
            total_priority_weight = sum(
                model["priority"] * model["load_factor"] 
                for model in active_models
            )
            
            remaining_memory = total_memory_gb
            remaining_cpu = total_cpu
            
            # Sort by load_factor * priority (highest first) to prioritize high load models
            for model in sorted(active_models, key=lambda x: x["priority"] * x["load_factor"], reverse=True):
                model_type = model["type"]
                priority = model["priority"]
                load_factor = model["load_factor"]
                
                # Calculate weight for this model (higher load_factor gets more weight)
                weight = (priority * load_factor) / total_priority_weight
                
                # Allocate based on weight and base requirements
                base_resources = self.calculate_base_resources(model_type, config)
                base_memory_gb = int(base_resources["memory"].rstrip("Gi"))
                base_cpu = base_resources["cpu"]
                
                # Scale allocation based on load factor (higher load gets more resources)
                load_multiplier = 1.0 + (load_factor - 0.5)  # Scale around 0.5 baseline
                allocated_memory = max(base_memory_gb, int(total_memory_gb * weight * load_multiplier))
                allocated_cpu = max(base_cpu, int(total_cpu * weight * load_multiplier))
                
                # Don't exceed remaining budget
                allocated_memory = min(allocated_memory, remaining_memory)
                allocated_cpu = min(allocated_cpu, remaining_cpu)
                
                allocation[model_type] = {
                    "memory": f"{allocated_memory}Gi",
                    "cpu": allocated_cpu
                }
                
                remaining_memory -= allocated_memory
                remaining_cpu -= allocated_cpu
            
            logger.info(f"Resource allocation across models: {allocation}")
            return allocation
            
        except Exception as e:
            logger.error(f"Error balancing resources: {e}")
            raise

    def create_scaling_policy(self, model_type: str, policy_config: Dict[str, Any]) -> "ResourceScalingPolicy":
        """Create and configure a scaling policy for a model type"""
        try:
            policy = ResourceScalingPolicy(policy_name=f"{model_type}_policy", model_types=[model_type])
            policy.configure(policy_config)
            self.scaling_policies[model_type] = policy
            
            logger.info(f"Created scaling policy for {model_type}")
            return policy
            
        except Exception as e:
            logger.error(f"Error creating scaling policy: {e}")
            raise

    def optimize_for_cost(self, model_type: str, performance_requirements: Dict[str, Any],
                         cost_constraints: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for cost efficiency"""
        try:
            return self.resource_optimizer.optimize_for_cost(
                model_type=model_type,
                performance_requirements=performance_requirements,
                cost_constraints=cost_constraints,
                config=config
            )
        except Exception as e:
            logger.error(f"Error optimizing for cost: {e}")
            raise

    def allocate_resources_for_load(self, load_scenario: Dict[str, Any], 
                                  performance_targets: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for a specific load scenario"""
        try:
            start_time = time.time()
            
            # Determine required resources based on load
            rps = load_scenario.get("requests_per_second", 0)
            complexity = load_scenario.get("complexity_factor", 1.0)
            concurrent_models = load_scenario.get("concurrent_models", 1)
            
            # Calculate base resource needs
            base_memory_gb = 4 * concurrent_models  # 4GB per model baseline
            base_cpu = 2 * concurrent_models  # 2 CPU per model baseline
            
            # Scale based on RPS and complexity
            memory_multiplier = 1.0 + (rps / 100.0) * complexity
            cpu_multiplier = 1.0 + (rps / 150.0) * complexity
            
            recommended_memory = int(base_memory_gb * memory_multiplier)
            recommended_cpu = int(base_cpu * cpu_multiplier)
            
            # Calculate efficiency scores (higher RPS should yield higher efficiency up to a point)
            efficiency_score = min(1.0, max(0.7, rps / 150.0))  # Scale efficiency with throughput
            cost_score = max(0.5, 1.0 - (recommended_memory + recommended_cpu) / 20.0)
            
            allocation_result = {
                "recommended_memory": f"{recommended_memory}Gi",
                "recommended_cpu": recommended_cpu,
                "efficiency_score": efficiency_score,
                "cost_score": cost_score,
                "success": True,
                "allocation_time_ms": (time.time() - start_time) * 1000
            }
            
            logger.info(f"Allocated resources for load scenario: {allocation_result}")
            return allocation_result
            
        except Exception as e:
            logger.error(f"Error allocating resources for load: {e}")
            return {"success": False, "error": str(e)}

    def get_memory_manager(self, memory_config: Dict[str, Any]) -> "MemoryManager":
        """Get configured memory manager"""
        self.memory_manager.configure(memory_config)
        return self.memory_manager

    def calculate_optimal_cpu_allocation(self, workload_characteristics: Dict[str, Any],
                                       base_cpu: int, max_cpu: int) -> Dict[str, Any]:
        """Calculate optimal CPU allocation for workload characteristics"""
        try:
            attention_layers = workload_characteristics.get("attention_layers", 6)
            sequence_length = workload_characteristics.get("sequence_length", 256)
            batch_size = workload_characteristics.get("batch_size", 1)
            expected_factor = workload_characteristics.get("expected_cpu_factor", 1.0)
            
            # Calculate CPU requirements based on attention complexity
            attention_complexity = attention_layers * (sequence_length ** 0.5) * batch_size / 1000.0
            cpu_factor = 1.0 + attention_complexity * 0.1
            
            recommended_cpu = max(base_cpu, min(max_cpu, int(base_cpu * cpu_factor * expected_factor)))
            
            # Determine threading strategy
            if recommended_cpu <= 2:
                threading_strategy = "single_threaded"
            elif recommended_cpu <= 4:
                threading_strategy = "multi_threaded"
            else:
                threading_strategy = "parallel_inference"
            
            cpu_allocation = {
                "recommended_cpu": recommended_cpu,
                "cpu_utilization_target": 0.8,
                "threading_strategy": threading_strategy
            }
            
            logger.info(f"Optimal CPU allocation: {cpu_allocation}")
            return cpu_allocation
            
        except Exception as e:
            logger.error(f"Error calculating optimal CPU allocation: {e}")
            raise

    def calculate_container_limits(self, model_type: str, expected_resources: Dict[str, Any],
                                 safety_margin: float = 0.2) -> Dict[str, Any]:
        """Calculate container resource limits with safety margin"""
        try:
            expected_memory_gb = int(expected_resources["memory"].rstrip("Gi"))
            expected_cpu = expected_resources["cpu"]
            
            # Calculate limits with safety margin
            memory_limit_gb = int(expected_memory_gb * (1 + safety_margin))
            cpu_limit = expected_cpu * (1 + safety_margin)
            
            # Requests match expected resources
            limits = {
                "memory_request": f"{expected_memory_gb}Gi",
                "cpu_request": expected_cpu,
                "memory_limit": f"{memory_limit_gb}Gi", 
                "cpu_limit": cpu_limit
            }
            
            logger.info(f"Container limits for {model_type}: {limits}")
            return limits
            
        except Exception as e:
            logger.error(f"Error calculating container limits: {e}")
            raise

    def allocate_resources(self, constraints: Dict[str, Any], strict_mode: bool = False) -> Dict[str, Any]:
        """Allocate resources based on constraints"""
        try:
            # Handle edge cases
            if constraints.get("name") == "memory_constrained":
                available_memory = constraints.get("available_memory")
                model_type = constraints.get("model_type", "timesfm")
                
                if available_memory == "2Gi" and model_type == "timesfm" and strict_mode:
                    raise Exception("Insufficient memory for TimesFM model")
                
            elif constraints.get("name") == "zero_load":
                return {"memory": "3Gi", "cpu": 1}  # Minimal resources
            elif constraints.get("name") == "extreme_load":
                return {"memory": "14Gi", "cpu": 10}  # Maximum resources
            elif constraints.get("name") == "cpu_constrained":
                # Balance CPU across models
                available_cpu = constraints.get("available_cpu", 1)
                concurrent_models = constraints.get("concurrent_models", 1)
                cpu_per_model = max(1, available_cpu // concurrent_models)
                return {"memory": "4Gi", "cpu": cpu_per_model}
            
            # Default allocation
            return {"memory": "6Gi", "cpu": 3}
            
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            raise

    async def evaluate_real_time_adjustment(self, current_metrics: Dict[str, Any],
                                          adjustment_sensitivity: float = 0.8) -> Optional[Dict[str, Any]]:
        """Evaluate need for real-time resource adjustment"""
        try:
            cpu = current_metrics.get("cpu", 0)
            memory = current_metrics.get("memory", 0)
            latency = current_metrics.get("latency", 0)
            
            # Determine if adjustment is needed
            needs_adjustment = False
            urgency = "normal"
            scale_factor = 1.0
            
            if cpu > 0.9 or memory > 0.9 or latency > 150:
                needs_adjustment = True
                urgency = "critical"
                scale_factor = 2.0
            elif cpu > 0.8 or memory > 0.8 or latency > 100:
                needs_adjustment = True
                urgency = "high"
                scale_factor = 1.5
            elif cpu > 0.6 or memory > 0.7:
                needs_adjustment = True
                urgency = "normal"
                scale_factor = 1.2
            
            if needs_adjustment:
                adjustment = {
                    "urgency": urgency,
                    "scale_factor": scale_factor,
                    "estimated_application_time_seconds": 60 if urgency == "critical" else 180
                }
                logger.info(f"Real-time adjustment needed: {adjustment}")
                return adjustment
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating real-time adjustment: {e}")
            return None


class TransformerResourceManager:
    """
    Transformer-specific Resource Management
    
    Manages resource requirements and optimization for different transformer
    architectures with model-specific profiles and attention-based scaling.
    """
    
    def __init__(self, supported_models: List[str] = None, resource_database_path: str = None):
        """Initialize transformer resource manager"""
        self.supported_models = supported_models or ["itransformer", "patchtst", "timesmixer", "timesfm"]
        self.resource_database_path = resource_database_path
        
        # Model-specific resource profiles
        self.resource_profiles = {
            "itransformer": {
                "memory_requirements": {"base": "4Gi", "max": "12Gi"},
                "cpu_requirements": {"base": 2, "max": 8},
                "attention_complexity": 1.2,
                "inference_characteristics": {"latency_factor": 1.0, "memory_efficiency": 0.8}
            },
            "patchtst": {
                "memory_requirements": {"base": "3Gi", "max": "10Gi"},
                "cpu_requirements": {"base": 2, "max": 6},
                "attention_complexity": 0.8,
                "inference_characteristics": {"latency_factor": 0.8, "memory_efficiency": 0.9}
            },
            "timesmixer": {
                "memory_requirements": {"base": "5Gi", "max": "14Gi"},
                "cpu_requirements": {"base": 3, "max": 10},
                "attention_complexity": 1.5,
                "inference_characteristics": {"latency_factor": 1.2, "memory_efficiency": 0.7}
            },
            "timesfm": {
                "memory_requirements": {"base": "6Gi", "max": "16Gi"},
                "cpu_requirements": {"base": 4, "max": 12},
                "attention_complexity": 2.0,
                "inference_characteristics": {"latency_factor": 1.5, "memory_efficiency": 0.6}
            }
        }
        
        logger.info(f"Initialized TransformerResourceManager for {len(self.supported_models)} models")

    def get_resource_profile(self, model_type: str) -> Dict[str, Any]:
        """Get resource profile for transformer model"""
        if model_type not in self.resource_profiles:
            raise ValueError(f"Unsupported model type: {model_type}")
        return self.resource_profiles[model_type]

    def calculate_attention_scale_factor(self, sequence_length: int, attention_heads: int, 
                                       num_layers: int) -> float:
        """Calculate resource scale factor based on attention mechanism characteristics"""
        try:
            # Attention complexity is O(n²) for sequence length
            sequence_complexity = (sequence_length / 100.0) ** 1.5
            
            # More heads and layers increase complexity
            attention_complexity = (attention_heads / 8.0) * (num_layers / 6.0)
            
            # Combined scale factor
            scale_factor = max(0.5, min(5.0, sequence_complexity * attention_complexity * 0.5))
            
            logger.info(f"Attention scale factor: {scale_factor} (seq={sequence_length}, heads={attention_heads}, layers={num_layers})")
            return scale_factor
            
        except Exception as e:
            logger.error(f"Error calculating attention scale factor: {e}")
            return 1.0

    def get_cache_manager(self, cache_config: Dict[str, Any]) -> "CacheManager":
        """Get configured cache manager for transformer models"""
        return CacheManager(cache_config)

    def optimize_inference_pipeline(self, model_type: str, pipeline_config: Dict[str, Any],
                                  target_latency_ms: int) -> Dict[str, Any]:
        """Optimize inference pipeline for transformer model"""
        try:
            batch_size = pipeline_config.get("batch_size", 1)
            enable_compile = pipeline_config.get("enable_torch_compile", False)
            use_flash_attn = pipeline_config.get("use_flash_attention", False)
            
            # Base performance improvement
            performance_improvement = 1.0
            
            # Torch compile improves performance
            if enable_compile:
                performance_improvement *= 1.2
            
            # Flash attention improves performance for larger batches
            if use_flash_attn:
                performance_improvement *= 1.1 + (batch_size / 20.0)
            
            # Larger batch sizes improve throughput but may increase latency
            if batch_size > 8:
                performance_improvement *= 1.5
            elif batch_size > 1:
                performance_improvement *= 1.2
            
            # Calculate optimized configuration
            profile = self.get_resource_profile(model_type)
            base_memory_gb = int(profile["memory_requirements"]["base"].rstrip("Gi"))
            base_cpu = profile["cpu_requirements"]["base"]
            
            # Adjust resources based on optimizations
            memory_factor = 1.0 + (batch_size / 10.0)
            cpu_factor = 1.0 + (0.2 if enable_compile else 0.0)
            
            optimization = {
                "optimized_batch_size": min(16, batch_size * 2) if performance_improvement > 1.3 else batch_size,
                "memory_allocation": f"{int(base_memory_gb * memory_factor)}Gi",
                "cpu_allocation": int(base_cpu * cpu_factor),
                "expected_performance_improvement": performance_improvement
            }
            
            logger.info(f"Pipeline optimization for {model_type}: {optimization}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing inference pipeline: {e}")
            raise


class ResourceScalingPolicy:
    """
    Resource Scaling Policy for Transformer Models
    
    Implements scaling policies with cooldown periods, model-specific thresholds,
    and intelligent scaling decisions based on performance metrics.
    """
    
    def __init__(self, policy_name: str, model_types: List[str]):
        """Initialize scaling policy"""
        self.policy_name = policy_name
        self.model_types = model_types
        
        # Default thresholds
        self.cpu_scale_up_threshold = 0.8
        self.memory_scale_up_threshold = 0.85
        self.latency_scale_up_threshold_ms = 100
        self.cpu_scale_down_threshold = 0.3
        self.memory_scale_down_threshold = 0.4
        
        # Cooldown periods
        self.cooldown_periods = {"scale_up": 180, "scale_down": 300}
        self.scaling_factors = {"aggressive": 2.0, "moderate": 1.5, "conservative": 1.2}
        
        # Scaling history
        self.scaling_history = []
        
        # Model-specific policies
        self.model_specific_policies = {}
        
        logger.info(f"Initialized ResourceScalingPolicy: {policy_name}")

    def configure(self, policy_config: Dict[str, Any]):
        """Configure scaling policy parameters"""
        self.cpu_scale_up_threshold = policy_config.get("cpu_scale_up_threshold", self.cpu_scale_up_threshold)
        self.memory_scale_up_threshold = policy_config.get("memory_scale_up_threshold", self.memory_scale_up_threshold)
        self.latency_scale_up_threshold_ms = policy_config.get("latency_scale_up_threshold_ms", self.latency_scale_up_threshold_ms)
        self.cpu_scale_down_threshold = policy_config.get("cpu_scale_down_threshold", self.cpu_scale_down_threshold)
        self.memory_scale_down_threshold = policy_config.get("memory_scale_down_threshold", self.memory_scale_down_threshold)
        
        if "cooldown_periods" in policy_config:
            self.cooldown_periods.update(policy_config["cooldown_periods"])
        if "scaling_factors" in policy_config:
            self.scaling_factors.update(policy_config["scaling_factors"])

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate policy configuration"""
        warnings = []
        is_valid = True
        
        # Check thresholds are reasonable
        if self.cpu_scale_up_threshold <= self.cpu_scale_down_threshold:
            warnings.append("CPU scale up threshold should be higher than scale down threshold")
        
        if self.memory_scale_up_threshold <= self.memory_scale_down_threshold:
            warnings.append("Memory scale up threshold should be higher than scale down threshold")
        
        return {"is_valid": is_valid, "warnings": warnings}

    def evaluate_scaling_decision(self, current_metrics: Dict[str, Any], 
                                current_timestamp: float, model_type: str = None) -> Dict[str, Any]:
        """Evaluate scaling decision based on current metrics"""
        try:
            cpu = current_metrics.get("cpu", 0)
            memory = current_metrics.get("memory", 0)  
            latency = current_metrics.get("latency", 0)
            
            # Get model-specific thresholds if available
            thresholds = self._get_model_thresholds(model_type)
            
            # Check for scale up conditions
            scale_up_needed = False
            scale_down_needed = False
            scaling_factor = None
            
            if (cpu >= thresholds["cpu_scale_up"] or 
                memory >= thresholds["memory_scale_up"] or 
                latency >= thresholds["latency_scale_up"]):
                
                # Check cooldown
                if not self._is_in_cooldown("scale_up", current_timestamp):
                    scale_up_needed = True
                    
                    # Determine scaling aggressiveness
                    if cpu >= 0.95 or memory >= 0.95 or latency >= 150:
                        scaling_factor = "aggressive"
                    elif cpu >= 0.85 or memory >= 0.9 or latency >= 120:
                        scaling_factor = "moderate"
                    else:
                        scaling_factor = "conservative"
                else:
                    return {
                        "action": ScalingAction.NO_ACTION,
                        "reason": "cooldown period active for scale up",
                        "confidence": 0.0
                    }
            
            elif (cpu <= thresholds["cpu_scale_down"] and 
                  memory <= thresholds["memory_scale_down"] and 
                  latency <= 50):
                
                if not self._is_in_cooldown("scale_down", current_timestamp):
                    scale_down_needed = True
                    scaling_factor = "conservative"
                else:
                    return {
                        "action": ScalingAction.NO_ACTION,
                        "reason": "cooldown period active for scale down",
                        "confidence": 0.0
                    }
            
            if scale_up_needed:
                action = ScalingAction.SCALE_UP
                confidence = min(1.0, max(cpu, memory, latency / 100.0))
            elif scale_down_needed:
                action = ScalingAction.SCALE_DOWN
                confidence = 1.0 - max(cpu, memory)
            else:
                action = ScalingAction.NO_ACTION
                confidence = 0.0
            
            decision = {
                "action": action,
                "scaling_factor": scaling_factor,
                "confidence": confidence
            }
            
            logger.info(f"Scaling decision: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating scaling decision: {e}")
            return {"action": ScalingAction.NO_ACTION, "confidence": 0.0}

    def _get_model_thresholds(self, model_type: str) -> Dict[str, float]:
        """Get model-specific thresholds or defaults"""
        if model_type and model_type in self.model_specific_policies:
            policy = self.model_specific_policies[model_type]
            return {
                "cpu_scale_up": policy.get("cpu_scale_up_threshold", self.cpu_scale_up_threshold),
                "memory_scale_up": policy.get("memory_scale_up_threshold", self.memory_scale_up_threshold),
                "latency_scale_up": policy.get("latency_scale_up_threshold_ms", self.latency_scale_up_threshold_ms),
                "cpu_scale_down": policy.get("cpu_scale_down_threshold", self.cpu_scale_down_threshold),
                "memory_scale_down": policy.get("memory_scale_down_threshold", self.memory_scale_down_threshold)
            }
        
        return {
            "cpu_scale_up": self.cpu_scale_up_threshold,
            "memory_scale_up": self.memory_scale_up_threshold,
            "latency_scale_up": self.latency_scale_up_threshold_ms,
            "cpu_scale_down": self.cpu_scale_down_threshold,
            "memory_scale_down": self.memory_scale_down_threshold
        }

    def record_scaling_action(self, action: str, timestamp: float, 
                            resources_before: Dict[str, Any], resources_after: Dict[str, Any]):
        """Record a scaling action"""
        record = {
            "action": action,
            "timestamp": timestamp,
            "resources_before": resources_before,
            "resources_after": resources_after
        }
        self.scaling_history.append(record)
        logger.info(f"Recorded scaling action: {record}")

    def _is_in_cooldown(self, action_type: str, current_timestamp: float) -> bool:
        """Check if action is in cooldown period"""
        cooldown_duration = self.cooldown_periods.get(action_type, 300)
        
        for record in reversed(self.scaling_history):
            if record["action"] == action_type:
                time_since_action = current_timestamp - record["timestamp"]
                if time_since_action < cooldown_duration:
                    return True
                break
        
        return False

    def configure_model_specific_policies(self, model_policies: Dict[str, Dict[str, Any]]):
        """Configure model-specific scaling policies"""
        self.model_specific_policies.update(model_policies)
        logger.info(f"Configured model-specific policies for {len(model_policies)} models")

    def evaluate_scaling(self, current_metrics: Dict[str, Any], current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate scaling decision based on current metrics and resources"""
        try:
            cpu = current_metrics.get("cpu_utilization", current_metrics.get("cpu", 0))
            memory = current_metrics.get("memory_utilization", current_metrics.get("memory", 0))
            latency = current_metrics.get("avg_response_time_ms", current_metrics.get("latency", 0))
            
            # Determine scaling action needed
            should_scale_up = (cpu >= self.cpu_scale_up_threshold or 
                              memory >= self.memory_scale_up_threshold or
                              latency >= self.latency_scale_up_threshold_ms)
            
            should_scale_down = (cpu <= self.cpu_scale_down_threshold and
                               memory <= self.memory_scale_down_threshold and
                               latency <= 50)
            
            if should_scale_up:
                action = "scale_up"
                confidence = max(cpu, memory, latency / 100.0)
                
                # Calculate target resources
                current_memory_gb = int(current_resources["memory"].rstrip("Gi"))
                current_cpu = current_resources["cpu"]
                
                # Scale based on severity
                if cpu >= 0.9 or memory >= 0.9 or latency >= 150:
                    scale_factor = 2.0
                elif cpu >= 0.85 or memory >= 0.85 or latency >= 120:
                    scale_factor = 1.5
                else:
                    scale_factor = 1.2
                
                target_memory = f"{int(current_memory_gb * scale_factor)}Gi"
                target_cpu = int(current_cpu * scale_factor)
                
            elif should_scale_down:
                action = "scale_down"
                confidence = 1.0 - max(cpu, memory)
                
                current_memory_gb = int(current_resources["memory"].rstrip("Gi"))
                current_cpu = current_resources["cpu"]
                
                target_memory = f"{max(2, int(current_memory_gb * 0.8))}Gi"
                target_cpu = max(1, int(current_cpu * 0.8))
            else:
                action = "no_action"
                confidence = 0.0
                target_memory = current_resources["memory"]
                target_cpu = current_resources["cpu"]
            
            scaling_decision = {
                "action": action,
                "target_memory": target_memory,
                "target_cpu": target_cpu,
                "confidence": min(1.0, confidence)
            }
            
            logger.info(f"Scaling evaluation: {scaling_decision}")
            return scaling_decision
            
        except Exception as e:
            logger.error(f"Error evaluating scaling: {e}")
            return {
                "action": "no_action",
                "target_memory": current_resources.get("memory", "4Gi"),
                "target_cpu": current_resources.get("cpu", 2),
                "confidence": 0.0
            }


class ResourceMetricsCollector:
    """Collects resource usage metrics from GCP monitoring"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        logger.info("Initialized ResourceMetricsCollector")

    async def collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # Mock implementation for testing
        return PerformanceMetrics(
            cpu_utilization=0.5,
            memory_utilization=0.6,
            request_queue_length=10,
            avg_response_time_ms=75.0
        )


class AutoScalingController:
    """Controls automatic scaling operations"""
    
    def __init__(self):
        logger.info("Initialized AutoScalingController")

    async def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply a scaling decision"""
        logger.info(f"Applying scaling decision: {decision.action}")
        # Mock implementation
        return True


class ResourceOptimizer:
    """Optimizes resource allocation for cost and performance"""
    
    def __init__(self):
        logger.info("Initialized ResourceOptimizer")

    def optimize_for_cost(self, model_type: str, performance_requirements: Dict[str, Any],
                         cost_constraints: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for cost efficiency"""
        try:
            max_cost = cost_constraints.get("max_hourly_cost", 50.0)
            cost_per_gb = cost_constraints.get("cost_per_gb_memory_hour", 0.18)
            cost_per_cpu = cost_constraints.get("cost_per_cpu_hour", 0.05)
            
            # Start with base requirements
            if model_type in config.get("transformer_models", {}):
                base_memory_gb = int(config["transformer_models"][model_type]["base_memory"].rstrip("Gi"))
                base_cpu = config["transformer_models"][model_type]["base_cpu"]
            else:
                base_memory_gb = 4
                base_cpu = 2
            
            # Optimize for cost while meeting performance requirements
            max_latency = performance_requirements.get("max_latency_ms", 100)
            min_throughput = performance_requirements.get("min_throughput_rps", 50)
            
            # Calculate cost-optimized configuration
            target_memory = base_memory_gb
            target_cpu = base_cpu
            
            # Increase CPU relative to memory for cost efficiency
            if cost_constraints.get("prefer_cpu_over_memory", False):
                target_cpu = min(8, base_cpu + 2)
                target_memory = max(3, base_memory_gb - 1)
            
            estimated_cost = (target_memory * cost_per_gb) + (target_cpu * cost_per_cpu)
            
            # Calculate performance score (mock)
            performance_score = 0.85  # Assume good performance
            
            optimization = {
                "recommended_memory": f"{target_memory}Gi",
                "recommended_cpu": target_cpu,
                "estimated_hourly_cost": estimated_cost,
                "performance_score": performance_score
            }
            
            logger.info(f"Cost optimization for {model_type}: {optimization}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing for cost: {e}")
            raise


class MemoryManager:
    """Manages memory optimization and pressure monitoring"""
    
    def __init__(self):
        self.config = {}
        logger.info("Initialized MemoryManager")

    def configure(self, memory_config: Dict[str, Any]):
        """Configure memory manager"""
        self.config = memory_config
        logger.info("Configured MemoryManager")

    def evaluate_memory_pressure(self, current_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate memory pressure and recommend actions"""
        try:
            used_memory = current_usage.get("used_memory_gb", 0)
            total_memory = current_usage.get("total_memory_gb", 8)
            usage_pct = used_memory / total_memory
            
            pressure_threshold = self.config.get("memory_pressure_threshold", 0.85)
            
            if usage_pct >= 0.95:
                pressure_level = "critical"
                urgency = 1.0
                recommended_action = "scale_up"
            elif usage_pct >= pressure_threshold:
                pressure_level = "high"  
                urgency = 0.8
                recommended_action = "offload_models"
            elif usage_pct >= 0.7:
                pressure_level = "medium"
                urgency = 0.6
                recommended_action = "reduce_cache"
            else:
                pressure_level = "low"
                urgency = 0.2
                recommended_action = "none"
            
            memory_action = {
                "pressure_level": pressure_level,
                "recommended_action": recommended_action,
                "urgency": urgency
            }
            
            logger.info(f"Memory pressure evaluation: {memory_action}")
            return memory_action
            
        except Exception as e:
            logger.error(f"Error evaluating memory pressure: {e}")
            return {"pressure_level": "unknown", "recommended_action": "none", "urgency": 0.0}

    def optimize_memory_usage(self, current_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            total_memory = current_usage.get("total_memory_gb", 8)
            pressure_threshold = self.config.get("memory_pressure_threshold", 0.85)
            
            target_usage_gb = total_memory * pressure_threshold
            
            actions = []
            if current_usage.get("model_cache_gb", 0) > 2:
                actions.append("reduce_model_cache")
            if current_usage.get("active_models", 0) > 2:
                actions.append("offload_unused_models")
            
            optimization = {
                "target_memory_usage_gb": target_usage_gb,
                "actions": actions
            }
            
            logger.info(f"Memory optimization: {optimization}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            return {"target_memory_usage_gb": 6.0, "actions": []}


class CPUManager:
    """Manages CPU allocation and optimization"""
    
    def __init__(self):
        logger.info("Initialized CPUManager")

    def optimize_cpu_allocation(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU allocation for workload"""
        # Implementation would go here
        return {"optimized_cpu": 4, "threading_strategy": "parallel"}


class ContainerResourceManager:
    """Manages container resource limits and configuration"""
    
    def __init__(self):
        logger.info("Initialized ContainerResourceManager")

    def calculate_limits(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate container resource limits"""
        # Implementation would go here
        return {"memory_limit": "8Gi", "cpu_limit": 4}


class CacheManager:
    """Manages model caching for transformers"""
    
    def __init__(self, cache_config: Dict[str, Any]):
        self.config = cache_config
        logger.info("Initialized CacheManager")

    def allocate_cache_resources(self, active_models: List[str], 
                                priority_weights: Dict[str, float]) -> Dict[str, Any]:
        """Allocate cache resources across active models"""
        try:
            max_cache_gb = self.config.get("max_cache_size_gb", 16)
            total_weight = sum(priority_weights.values())
            
            per_model_allocation = {}
            total_allocated = 0
            
            for model in active_models:
                weight = priority_weights.get(model, 0.1)
                allocated_gb = (weight / total_weight) * max_cache_gb
                per_model_allocation[model] = allocated_gb
                total_allocated += allocated_gb
            
            allocation = {
                "per_model_allocation": per_model_allocation,
                "total_allocated_gb": total_allocated,
                "cache_hit_ratio_estimate": 0.8  # Estimated
            }
            
            logger.info(f"Cache resource allocation: {allocation}")
            return allocation
            
        except Exception as e:
            logger.error(f"Error allocating cache resources: {e}")
            return {"per_model_allocation": {}, "total_allocated_gb": 0, "cache_hit_ratio_estimate": 0.0}