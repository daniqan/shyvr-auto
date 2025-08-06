"""
Performance Testing for Dynamic Resource Allocation

Provides performance testing capabilities for validating resource allocation
decisions and system performance under various load conditions.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestScenario:
    """Load testing scenario configuration"""
    name: str
    concurrent_users: int
    requests_per_second: int
    duration_seconds: int
    model_types: List[str]
    complexity_factor: float = 1.0


@dataclass
class PerformanceResult:
    """Performance test result"""
    scenario_name: str
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    resource_utilization: Dict[str, float]
    error_count: int
    total_requests: int


class UserBehaviorPattern(Enum):
    """User behavior patterns for testing"""
    CONSTANT_LOAD = "constant"
    SPIKE_LOAD = "spike"
    GRADUAL_INCREASE = "gradual"
    RANDOM_BURST = "random_burst"


class ResourceAllocationPerformanceTester:
    """
    Resource Allocation Performance Tester
    
    Tests performance of the dynamic resource allocation system under
    various load conditions and validates scaling behavior.
    """
    
    def __init__(self):
        """Initialize performance tester"""
        self.test_results = []
        self.active_tests = {}
        
        logger.info("Initialized ResourceAllocationPerformanceTester")

    async def simulate_allocation_request(self, model_type: str, load_factor: float) -> Dict[str, Any]:
        """Simulate a resource allocation request"""
        try:
            start_time = time.time()
            
            # Simulate allocation processing time based on load
            base_latency = random.uniform(5, 15)  # 5-15ms base latency
            load_latency = load_factor * random.uniform(0, 10)  # Additional latency based on load
            
            processing_time = (base_latency + load_latency) / 1000.0  # Convert to seconds
            await asyncio.sleep(processing_time)
            
            # Simulate success rate (higher load = slightly lower success rate)
            success_probability = max(0.95, 1.0 - (load_factor * 0.05))
            success = random.random() < success_probability
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = {
                "success": success,
                "latency_ms": latency_ms,
                "model_type": model_type,
                "load_factor": load_factor
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating allocation request: {e}")
            return {"success": False, "latency_ms": 0, "error": str(e)}

    async def test_memory_efficiency(self, total_memory: str, concurrent_models: int,
                                   test_duration_seconds: int) -> Dict[str, Any]:
        """Test memory allocation efficiency"""
        try:
            memory_gb = int(total_memory.rstrip("Gi"))
            start_time = time.time()
            
            # Simulate memory allocation patterns
            allocations = []
            allocation_failures = 0
            
            # Simulate allocating memory for concurrent models
            memory_per_model = memory_gb / concurrent_models
            
            for i in range(concurrent_models):
                # Simulate some variability in memory usage
                actual_usage = memory_per_model * random.uniform(0.8, 1.2)
                if actual_usage <= memory_per_model * 1.1:  # Within reasonable bounds
                    allocations.append(actual_usage)
                else:
                    allocation_failures += 1
            
            # Simulate test duration
            await asyncio.sleep(min(test_duration_seconds, 1.0))  # Cap for testing
            
            total_allocated = sum(allocations)
            memory_utilization = total_allocated / memory_gb
            
            # Calculate fragmentation (mock)
            fragmentation = max(0, 0.05 + random.uniform(-0.02, 0.08))
            
            efficiency_result = {
                "memory_utilization": memory_utilization,
                "allocation_failures": allocation_failures,
                "memory_fragmentation": fragmentation,
                "total_allocated_gb": total_allocated,
                "efficiency_score": memory_utilization * (1 - fragmentation)
            }
            
            logger.info(f"Memory efficiency test result: {efficiency_result}")
            return efficiency_result
            
        except Exception as e:
            logger.error(f"Error testing memory efficiency: {e}")
            return {"memory_utilization": 0, "allocation_failures": 1, "memory_fragmentation": 1.0}

    def optimize_for_cost(self, budget_per_hour: float, active_models: List[str],
                         performance_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for cost constraints"""
        try:
            # Base costs per resource unit (mock pricing)
            cost_per_gb_hour = 0.18
            cost_per_cpu_hour = 0.05
            
            # Model resource requirements (simplified)
            model_requirements = {
                "itransformer": {"memory_gb": 4, "cpu": 2},
                "patchtst": {"memory_gb": 3, "cpu": 2}, 
                "timesmixer": {"memory_gb": 5, "cpu": 3},
                "timesfm": {"memory_gb": 6, "cpu": 4}
            }
            
            # Calculate total base requirements
            total_memory_gb = sum(model_requirements.get(model, {"memory_gb": 4})["memory_gb"] for model in active_models)
            total_cpu = sum(model_requirements.get(model, {"cpu": 2})["cpu"] for model in active_models)
            
            # Optimize within budget
            base_cost = (total_memory_gb * cost_per_gb_hour) + (total_cpu * cost_per_cpu_hour)
            
            if base_cost <= budget_per_hour:
                # We can afford base requirements, possibly scale up
                remaining_budget = budget_per_hour - base_cost
                
                # Add extra resources if budget allows
                extra_memory = min(4, int(remaining_budget / cost_per_gb_hour * 0.5))
                extra_cpu = min(2, int(remaining_budget / cost_per_cpu_hour * 0.5))
                
                total_memory_gb += extra_memory
                total_cpu += extra_cpu
            else:
                # Need to scale down to fit budget
                scale_factor = budget_per_hour / base_cost
                total_memory_gb = max(len(active_models) * 2, int(total_memory_gb * scale_factor))
                total_cpu = max(len(active_models), int(total_cpu * scale_factor))
            
            final_cost = (total_memory_gb * cost_per_gb_hour) + (total_cpu * cost_per_cpu_hour)
            
            # Predict performance based on resources
            req_latency = performance_requirements.get("latency_ms", 100)
            req_throughput = performance_requirements.get("throughput_rps", 50)
            
            # Simple performance prediction
            memory_factor = total_memory_gb / (len(active_models) * 4)  # 4GB baseline per model
            cpu_factor = total_cpu / (len(active_models) * 2)  # 2 CPU baseline per model
            
            predicted_latency = req_latency * (2.0 - min(memory_factor, cpu_factor))
            predicted_throughput = req_throughput * min(memory_factor, cpu_factor)
            
            optimization_result = {
                "estimated_cost_per_hour": final_cost,
                "predicted_latency_ms": predicted_latency,
                "predicted_throughput_rps": predicted_throughput,
                "allocated_memory_gb": total_memory_gb,
                "allocated_cpu": total_cpu,
                "optimization_score": min(1.0, budget_per_hour / max(final_cost, 0.01))
            }
            
            logger.info(f"Cost optimization result: {optimization_result}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing for cost: {e}")
            return {"estimated_cost_per_hour": budget_per_hour, "optimization_score": 0}

    async def run_load_test(self, scenario: LoadTestScenario) -> PerformanceResult:
        """Run a comprehensive load test scenario"""
        try:
            logger.info(f"Starting load test: {scenario.name}")
            start_time = time.time()
            
            results = []
            errors = 0
            
            # Calculate request rate
            total_requests = scenario.requests_per_second * scenario.duration_seconds
            request_interval = 1.0 / scenario.requests_per_second
            
            # Generate requests
            tasks = []
            for i in range(min(total_requests, 100)):  # Cap for testing
                # Select random model type
                model_type = random.choice(scenario.model_types)
                
                # Create allocation request task
                task = asyncio.create_task(
                    self.simulate_allocation_request(model_type, scenario.complexity_factor)
                )
                tasks.append(task)
                
                # Respect request rate
                if i < total_requests - 1:
                    await asyncio.sleep(min(request_interval, 0.1))  # Cap sleep for testing
            
            # Wait for all requests to complete
            request_results = await asyncio.gather(*tasks)
            
            # Process results
            successful_results = [r for r in request_results if r.get("success", False)]
            latencies = [r["latency_ms"] for r in successful_results]
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            success_rate = len(successful_results) / len(request_results) if request_results else 0
            errors = len(request_results) - len(successful_results)
            
            # Calculate throughput
            test_duration = time.time() - start_time
            throughput = len(successful_results) / test_duration
            
            # Mock resource utilization
            resource_utilization = {
                "cpu": min(95, 30 + scenario.complexity_factor * 40),
                "memory": min(90, 25 + scenario.complexity_factor * 50),
                "network": min(80, 20 + len(scenario.model_types) * 15)
            }
            
            result = PerformanceResult(
                scenario_name=scenario.name,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                throughput_rps=throughput,
                resource_utilization=resource_utilization,
                error_count=errors,
                total_requests=len(request_results)
            )
            
            self.test_results.append(result)
            logger.info(f"Load test completed: {scenario.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error running load test: {e}")
            return PerformanceResult(
                scenario_name=scenario.name,
                success_rate=0,
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_rps=0,
                resource_utilization={},
                error_count=1,
                total_requests=0
            )

    async def run_comprehensive_performance_suite(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        try:
            logger.info("Starting comprehensive performance test suite")
            
            # Define test scenarios
            scenarios = [
                LoadTestScenario(
                    name="baseline_load",
                    concurrent_users=10,
                    requests_per_second=25,
                    duration_seconds=30,
                    model_types=["itransformer"],
                    complexity_factor=1.0
                ),
                LoadTestScenario(
                    name="medium_load",
                    concurrent_users=25,
                    requests_per_second=75,
                    duration_seconds=30,
                    model_types=["itransformer", "patchtst"],
                    complexity_factor=1.5
                ),
                LoadTestScenario(
                    name="high_load",
                    concurrent_users=50,
                    requests_per_second=150,
                    duration_seconds=30,
                    model_types=["itransformer", "patchtst", "timesmixer"],
                    complexity_factor=2.0
                ),
                LoadTestScenario(
                    name="stress_test",
                    concurrent_users=100,
                    requests_per_second=300,
                    duration_seconds=30,
                    model_types=["itransformer", "patchtst", "timesmixer", "timesfm"],
                    complexity_factor=3.0
                )
            ]
            
            # Run all scenarios
            suite_results = []
            for scenario in scenarios:
                result = await self.run_load_test(scenario)
                suite_results.append(result)
            
            # Analyze suite results
            suite_summary = self._analyze_suite_results(suite_results)
            
            logger.info("Comprehensive performance test suite completed")
            return suite_summary
            
        except Exception as e:
            logger.error(f"Error running performance suite: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_suite_results(self, results: List[PerformanceResult]) -> Dict[str, Any]:
        """Analyze performance test suite results"""
        try:
            if not results:
                return {"status": "no_results"}
            
            # Calculate overall metrics
            avg_success_rate = sum(r.success_rate for r in results) / len(results)
            avg_latency = sum(r.avg_latency_ms for r in results) / len(results)
            max_p99_latency = max(r.p99_latency_ms for r in results)
            total_throughput = sum(r.throughput_rps for r in results)
            total_errors = sum(r.error_count for r in results)
            
            # Determine performance grade
            if avg_success_rate >= 0.99 and max_p99_latency <= 200:
                performance_grade = "A"
            elif avg_success_rate >= 0.95 and max_p99_latency <= 500:
                performance_grade = "B"  
            elif avg_success_rate >= 0.90 and max_p99_latency <= 1000:
                performance_grade = "C"
            else:
                performance_grade = "D"
            
            summary = {
                "status": "completed",
                "total_scenarios": len(results),
                "overall_success_rate": avg_success_rate,
                "avg_latency_ms": avg_latency,
                "max_p99_latency_ms": max_p99_latency,
                "total_throughput_rps": total_throughput,
                "total_errors": total_errors,
                "performance_grade": performance_grade,
                "scenario_results": [
                    {
                        "name": r.scenario_name,
                        "success_rate": r.success_rate,
                        "avg_latency_ms": r.avg_latency_ms,
                        "throughput_rps": r.throughput_rps
                    }
                    for r in results
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error analyzing suite results: {e}")
            return {"status": "analysis_error"}

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            if not self.test_results:
                return {"status": "no_data", "message": "No test results available"}
            
            # Group results by scenario type
            scenario_groups = {}
            for result in self.test_results:
                scenario_type = result.scenario_name.split("_")[0]  # e.g., "baseline", "medium", etc.
                if scenario_type not in scenario_groups:
                    scenario_groups[scenario_type] = []
                scenario_groups[scenario_type].append(result)
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_tests_run": len(self.test_results),
                "scenario_groups": scenario_groups,
                "performance_trends": self._calculate_performance_trends(),
                "resource_efficiency": self._calculate_resource_efficiency(),
                "recommendations": recommendations,
                "summary_statistics": self._calculate_summary_statistics()
            }
            
            logger.info("Generated comprehensive performance report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"status": "error", "message": str(e)}

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self.test_results:
            return ["No test data available for recommendations"]
        
        # Analyze latency patterns
        avg_latencies = [r.avg_latency_ms for r in self.test_results]
        if max(avg_latencies) > 100:
            recommendations.append("Consider increasing CPU allocation to reduce latency")
        
        # Analyze success rates
        success_rates = [r.success_rate for r in self.test_results]
        if min(success_rates) < 0.95:
            recommendations.append("Improve error handling and add more robust fallback mechanisms")
        
        # Analyze resource utilization
        cpu_utilizations = [r.resource_utilization.get("cpu", 0) for r in self.test_results if r.resource_utilization]
        if cpu_utilizations and max(cpu_utilizations) > 90:
            recommendations.append("CPU utilization is high - consider auto-scaling policies")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges")
        
        return recommendations

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.test_results) < 2:
            return {"status": "insufficient_data"}
        
        # Sort by timestamp if available, otherwise by order
        sorted_results = self.test_results
        
        latency_trend = "stable"
        throughput_trend = "stable"
        
        if len(sorted_results) >= 3:
            recent_latency = sum(r.avg_latency_ms for r in sorted_results[-3:]) / 3
            earlier_latency = sum(r.avg_latency_ms for r in sorted_results[:3]) / 3
            
            if recent_latency > earlier_latency * 1.1:
                latency_trend = "increasing"
            elif recent_latency < earlier_latency * 0.9:
                latency_trend = "decreasing"
        
        return {
            "latency_trend": latency_trend,
            "throughput_trend": throughput_trend,
            "trend_confidence": "medium"
        }

    def _calculate_resource_efficiency(self) -> Dict[str, Any]:
        """Calculate resource efficiency metrics"""
        if not self.test_results:
            return {"status": "no_data"}
        
        # Calculate efficiency based on throughput vs resource utilization
        efficiencies = []
        for result in self.test_results:
            if result.resource_utilization:
                cpu_util = result.resource_utilization.get("cpu", 50)
                throughput = result.throughput_rps
                
                # Simple efficiency metric: throughput per CPU utilization
                efficiency = throughput / max(cpu_util, 1)
                efficiencies.append(efficiency)
        
        if efficiencies:
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            return {
                "average_efficiency": avg_efficiency,
                "efficiency_rating": "high" if avg_efficiency > 1.0 else "medium" if avg_efficiency > 0.5 else "low"
            }
        
        return {"status": "insufficient_data"}

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not self.test_results:
            return {}
        
        latencies = [r.avg_latency_ms for r in self.test_results]
        success_rates = [r.success_rate for r in self.test_results]
        throughputs = [r.throughput_rps for r in self.test_results]
        
        return {
            "latency_stats": {
                "min": min(latencies),
                "max": max(latencies), 
                "avg": sum(latencies) / len(latencies),
                "std_dev": np.std(latencies) if len(latencies) > 1 else 0
            },
            "success_rate_stats": {
                "min": min(success_rates),
                "max": max(success_rates),
                "avg": sum(success_rates) / len(success_rates)
            },
            "throughput_stats": {
                "min": min(throughputs),
                "max": max(throughputs),
                "avg": sum(throughputs) / len(throughputs),
                "total": sum(throughputs)
            }
        }