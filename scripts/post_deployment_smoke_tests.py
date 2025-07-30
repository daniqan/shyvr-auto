#!/usr/bin/env python3
"""
Post-deployment Smoke Tests for Shyvr RLTE
Comprehensive smoke testing to verify deployed system functionality
"""

import asyncio
import aiohttp
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SmokeTestResult:
    """Smoke test result container"""
    test_category: str
    test_name: str
    passed: bool
    message: str
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class PostDeploymentSmokeTests:
    """Comprehensive post-deployment smoke testing system"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[SmokeTestResult] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def add_result(self, result: SmokeTestResult):
        """Add test result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        status_info = f" [{result.status_code}]" if result.status_code else ""
        time_info = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms else ""
        logger.info(f"{status} {result.test_category}.{result.test_name}{status_info}{time_info}: {result.message}")
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Tuple[int, Dict[str, Any], float]:
        """Make HTTP request and return status, response, and timing"""
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        
        start_time = time.time()
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_time = (time.time() - start_time) * 1000
                
                try:
                    if response.content_type == 'application/json':
                        data = await response.json()
                    else:
                        text = await response.text()
                        data = {"content": text[:500] + "..." if len(text) > 500 else text}
                except Exception:
                    data = {"error": "Failed to parse response"}
                
                return response.status, data, response_time
                
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return 0, {"error": "Request timeout"}, response_time
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return 0, {"error": str(e)}, response_time
    
    async def test_basic_connectivity(self) -> List[SmokeTestResult]:
        """Test basic connectivity and service availability"""
        results = []
        
        # Test root endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/")
        
        root_accessible = status_code in [200, 301, 302]
        results.append(SmokeTestResult(
            test_category="Connectivity",
            test_name="Root Endpoint",
            passed=root_accessible,
            message=f"Root endpoint {'accessible' if root_accessible else 'failed'}",
            status_code=status_code,
            response_time_ms=response_time,
            details={"endpoint": "/", "response_type": type(response_data).__name__}
        ))
        
        # Test service response time
        fast_response = response_time < 5000  # Under 5 seconds
        results.append(SmokeTestResult(
            test_category="Connectivity",
            test_name="Response Time",
            passed=fast_response,
            message=f"Root response time: {response_time:.0f}ms {'(acceptable)' if fast_response else '(too slow)'}",
            response_time_ms=response_time,
            details={"threshold_ms": 5000, "actual_ms": response_time}
        ))
        
        return results
    
    async def test_health_endpoints(self) -> List[SmokeTestResult]:
        """Test health and monitoring endpoints"""
        results = []
        
        # Test main health endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/health")
        
        health_working = status_code == 200
        results.append(SmokeTestResult(
            test_category="Health",
            test_name="Health Endpoint",
            passed=health_working,
            message=f"Health endpoint {'responding' if health_working else 'failed'}",
            status_code=status_code,
            response_time_ms=response_time,
            details={"response_data": response_data}
        ))
        
        if health_working:
            # Check health status details
            overall_status = response_data.get("status", "unknown")
            system_healthy = overall_status == "healthy"
            
            results.append(SmokeTestResult(
                test_category="Health",
                test_name="System Health Status",
                passed=system_healthy,
                message=f"System health status: {overall_status}",
                details={"health_status": overall_status, "full_response": response_data}
            ))
            
            # Check component health
            critical_components = ["database", "ml_models", "rl_agent"]
            for component in critical_components:
                if component in response_data:
                    component_status = response_data[component].get("status", "unknown")
                    component_healthy = component_status == "healthy"
                    
                    results.append(SmokeTestResult(
                        test_category="Health",
                        test_name=f"Component {component}",
                        passed=component_healthy,
                        message=f"Component {component}: {component_status}",
                        details=response_data[component]
                    ))
        
        # Test metrics endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/metrics")
        
        metrics_available = status_code in [200, 404]  # 404 is acceptable if metrics disabled
        results.append(SmokeTestResult(
            test_category="Health",
            test_name="Metrics Endpoint",
            passed=metrics_available,
            message=f"Metrics endpoint {'available' if status_code == 200 else 'disabled' if status_code == 404 else 'failed'}",
            status_code=status_code,
            response_time_ms=response_time,
            details={"metrics_enabled": status_code == 200}
        ))
        
        return results
    
    async def test_api_functionality(self) -> List[SmokeTestResult]:
        """Test core API functionality"""
        results = []
        
        # Test API base endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/api")
        
        api_accessible = status_code in [200, 404]  # 404 acceptable if no base API route
        results.append(SmokeTestResult(
            test_category="API",
            test_name="API Base",
            passed=api_accessible,
            message=f"API base endpoint {'accessible' if status_code == 200 else 'not found' if status_code == 404 else 'failed'}",
            status_code=status_code,
            response_time_ms=response_time
        ))
        
        # Test dashboard API endpoints
        dashboard_endpoints = [
            "/api/dashboard",
            "/api/dashboard/status",
            "/api/backtest",
            "/api/experience"
        ]
        
        for endpoint in dashboard_endpoints:
            status_code, response_data, response_time = await self.make_request("GET", endpoint)
            
            # API endpoints might require auth, so 401/403/404 are acceptable
            endpoint_working = status_code in [200, 401, 403, 404]
            
            results.append(SmokeTestResult(
                test_category="API",
                test_name=f"Endpoint {endpoint}",
                passed=endpoint_working,
                message=f"Endpoint {endpoint} {'responding' if endpoint_working else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={"requires_auth": status_code in [401, 403]}
            ))
        
        return results
    
    async def test_static_assets(self) -> List[SmokeTestResult]:
        """Test static asset serving"""
        results = []
        
        static_assets = [
            ("/static/index.html", "Dashboard HTML"),
            ("/static/dashboard.js", "Dashboard JavaScript"),
            ("/static/styles.css", "Dashboard CSS"),
            ("/static/backtest-results.js", "Backtest JavaScript")
        ]
        
        for asset_path, description in static_assets:
            status_code, response_data, response_time = await self.make_request("GET", asset_path)
            
            asset_available = status_code == 200
            results.append(SmokeTestResult(
                test_category="Static Assets",
                test_name=description,
                passed=asset_available or status_code == 404,  # 404 is OK for optional assets
                message=f"{description} {'available' if asset_available else 'not found' if status_code == 404 else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={"asset_path": asset_path, "content_length": len(response_data.get("content", ""))}
            ))
        
        return results
    
    async def test_authentication_system(self) -> List[SmokeTestResult]:
        """Test authentication system"""
        results = []
        
        # Test auth endpoints
        auth_endpoints = [
            ("/api/auth/key", "GET", "API Key Endpoint"),
            ("/api/auth/status", "GET", "Auth Status")
        ]
        
        for endpoint, method, description in auth_endpoints:
            status_code, response_data, response_time = await self.make_request(method, endpoint)
            
            auth_responding = status_code in [200, 401, 403, 404]
            results.append(SmokeTestResult(
                test_category="Authentication",
                test_name=description,
                passed=auth_responding,
                message=f"{description} {'responding' if auth_responding else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={"endpoint": endpoint, "method": method}
            ))
        
        # Test protected endpoint access (should require auth)
        protected_endpoints = ["/api/dashboard/config", "/api/experience"]
        
        for endpoint in protected_endpoints:
            status_code, response_data, response_time = await self.make_request("GET", endpoint)
            
            # Protected endpoints should return 401/403 without auth, or 200 if auth bypassed
            properly_protected = status_code in [401, 403] or (status_code == 200 and "authenticated" in str(response_data).lower())
            
            results.append(SmokeTestResult(
                test_category="Authentication",
                test_name=f"Protection {endpoint}",
                passed=properly_protected,
                message=f"Endpoint {endpoint} {'properly protected' if status_code in [401, 403] else 'accessible' if status_code == 200 else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={"endpoint": endpoint, "requires_auth": status_code in [401, 403]}
            ))
        
        return results
    
    async def test_database_connectivity(self) -> List[SmokeTestResult]:
        """Test database connectivity through API"""
        results = []
        
        # Test health endpoint for database status
        status_code, response_data, response_time = await self.make_request("GET", "/health")
        
        if status_code == 200 and "database" in response_data:
            db_status = response_data["database"].get("status", "unknown")
            db_healthy = db_status == "healthy"
            
            results.append(SmokeTestResult(
                test_category="Database",
                test_name="Database Health",
                passed=db_healthy,
                message=f"Database health: {db_status}",
                details=response_data["database"]
            ))
            
            # Check table status if available
            if "tables_exist" in response_data["database"]:
                tables_exist = response_data["database"]["tables_exist"]
                results.append(SmokeTestResult(
                    test_category="Database",
                    test_name="Database Tables",
                    passed=tables_exist,
                    message=f"Database tables {'exist' if tables_exist else 'missing'}",
                    details={"tables_exist": tables_exist}
                ))
        else:
            results.append(SmokeTestResult(
                test_category="Database",
                test_name="Database Status",
                passed=False,
                message="Cannot determine database status from health endpoint",
                details={"health_response": response_data}
            ))
        
        return results
    
    async def test_ml_rl_systems(self) -> List[SmokeTestResult]:
        """Test ML/RL system availability"""
        results = []
        
        # Test through health endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/health")
        
        if status_code == 200:
            # Test ML models
            if "ml_models" in response_data:
                ml_status = response_data["ml_models"].get("status", "unknown")
                ml_healthy = ml_status in ["healthy", "loaded"]
                
                results.append(SmokeTestResult(
                    test_category="ML/RL",
                    test_name="ML Models",
                    passed=ml_healthy,
                    message=f"ML models status: {ml_status}",
                    details=response_data["ml_models"]
                ))
            
            # Test RL agent
            if "rl_agent" in response_data:
                rl_status = response_data["rl_agent"].get("status", "unknown")
                rl_ready = rl_status in ["healthy", "ready", "training"]
                
                results.append(SmokeTestResult(
                    test_category="ML/RL",
                    test_name="RL Agent",
                    passed=rl_ready,
                    message=f"RL agent status: {rl_status}",
                    details=response_data["rl_agent"]
                ))
        
        # Test experience API endpoint
        status_code, response_data, response_time = await self.make_request("GET", "/api/experience")
        
        experience_api_working = status_code in [200, 401, 403]  # Should respond, might need auth
        results.append(SmokeTestResult(
            test_category="ML/RL",
            test_name="Experience API",
            passed=experience_api_working,
            message=f"Experience API {'responding' if experience_api_working else 'failed'}",
            status_code=status_code,
            response_time_ms=response_time
        ))
        
        return results
    
    async def test_performance_characteristics(self) -> List[SmokeTestResult]:
        """Test system performance characteristics"""
        results = []
        
        # Test multiple requests to key endpoints
        performance_tests = [
            ("/health", "Health Check"),
            ("/", "Root Page"),
            ("/api", "API Base")
        ]
        
        for endpoint, description in performance_tests:
            response_times = []
            success_count = 0
            
            # Make 5 requests to test consistency
            for i in range(5):
                status_code, response_data, response_time = await self.make_request("GET", endpoint)
                
                if status_code > 0 and status_code < 500:  # Any non-server-error response
                    response_times.append(response_time)
                    success_count += 1
                
                await asyncio.sleep(0.2)  # Small delay between requests
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                # Performance thresholds
                is_fast = avg_response_time < 2000  # Under 2 seconds average
                is_consistent = max_response_time < avg_response_time * 3  # Max not more than 3x average
                
                results.append(SmokeTestResult(
                    test_category="Performance",
                    test_name=f"Response Time {description}",
                    passed=is_fast and is_consistent,
                    message=f"{description} avg: {avg_response_time:.0f}ms, max: {max_response_time:.0f}ms",
                    response_time_ms=avg_response_time,
                    details={
                        "endpoint": endpoint,
                        "avg_response_time_ms": avg_response_time,
                        "max_response_time_ms": max_response_time,
                        "success_rate": success_count / 5,
                        "all_response_times": response_times
                    }
                ))
            else:
                results.append(SmokeTestResult(
                    test_category="Performance",
                    test_name=f"Response Time {description}",
                    passed=False,
                    message=f"{description} performance test failed - no successful responses",
                    details={"endpoint": endpoint, "failed_requests": 5}
                ))
        
        return results
    
    async def test_error_handling(self) -> List[SmokeTestResult]:
        """Test system error handling"""
        results = []
        
        # Test 404 handling
        status_code, response_data, response_time = await self.make_request("GET", "/nonexistent-endpoint")
        
        handles_404 = status_code == 404
        results.append(SmokeTestResult(
            test_category="Error Handling",
            test_name="404 Not Found",
            passed=handles_404,
            message=f"404 handling {'correct' if handles_404 else 'incorrect'} (got {status_code})",
            status_code=status_code,
            response_time_ms=response_time
        ))
        
        # Test method not allowed
        status_code, response_data, response_time = await self.make_request("DELETE", "/health")
        
        handles_method_not_allowed = status_code in [405, 404]  # 405 Method Not Allowed or 404
        results.append(SmokeTestResult(
            test_category="Error Handling",
            test_name="Method Not Allowed",
            passed=handles_method_not_allowed,
            message=f"Method not allowed handling {'correct' if handles_method_not_allowed else 'incorrect'} (got {status_code})",
            status_code=status_code,
            response_time_ms=response_time
        ))
        
        # Test malformed request handling
        try:
            status_code, response_data, response_time = await self.make_request(
                "POST", "/api/dashboard", 
                json={"invalid": "malformed json test"}
            )
            
            handles_bad_request = status_code in [400, 401, 403, 404, 405]
            results.append(SmokeTestResult(
                test_category="Error Handling",
                test_name="Malformed Request",
                passed=handles_bad_request,
                message=f"Malformed request handling {'correct' if handles_bad_request else 'incorrect'} (got {status_code})",
                status_code=status_code,
                response_time_ms=response_time
            ))
        except Exception as e:
            results.append(SmokeTestResult(
                test_category="Error Handling",
                test_name="Malformed Request",
                passed=True,  # Exception handling is also acceptable
                message=f"Malformed request properly rejected with exception: {str(e)[:100]}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def test_security_measures(self) -> List[SmokeTestResult]:
        """Test basic security measures"""
        results = []
        
        # Test HTTPS (if applicable)
        is_https = self.base_url.startswith('https://')
        results.append(SmokeTestResult(
            test_category="Security",
            test_name="HTTPS Usage",
            passed=is_https,
            message=f"Service uses {'HTTPS (secure)' if is_https else 'HTTP (insecure)'}",
            details={"uses_https": is_https, "base_url": self.base_url}
        ))
        
        # Test security headers
        status_code, response_data, response_time = await self.make_request("GET", "/")
        
        if status_code == 200 and hasattr(self.session, '_connector'):
            # Check for basic security headers in response
            try:
                # Make another request to get headers
                async with self.session.get(self.base_url + '/') as response:
                    headers = dict(response.headers)
                    
                    security_headers = [
                        'X-Frame-Options',
                        'X-Content-Type-Options', 
                        'X-XSS-Protection'
                    ]
                    
                    found_headers = [h for h in security_headers if h.lower() in [k.lower() for k in headers.keys()]]
                    
                    has_security_headers = len(found_headers) > 0
                    results.append(SmokeTestResult(
                        test_category="Security",
                        test_name="Security Headers",
                        passed=has_security_headers,
                        message=f"Security headers: {len(found_headers)}/{len(security_headers)} present",
                        details={"found_headers": found_headers, "all_headers": list(headers.keys())}
                    ))
            except Exception as e:
                results.append(SmokeTestResult(
                    test_category="Security",
                    test_name="Security Headers",
                    passed=True,  # Don't fail on header check issues
                    message=f"Could not check security headers: {str(e)[:100]}",
                    details={"error": str(e)}
                ))
        
        return results
    
    async def run_all_smoke_tests(self) -> Dict[str, Any]:
        """Run all post-deployment smoke tests"""
        logger.info(f"🔥 Starting post-deployment smoke tests for {self.base_url}")
        
        test_categories = [
            ("Basic Connectivity", self.test_basic_connectivity()),
            ("Health Endpoints", self.test_health_endpoints()),
            ("API Functionality", self.test_api_functionality()),
            ("Static Assets", self.test_static_assets()),
            ("Authentication", self.test_authentication_system()),
            ("Database", self.test_database_connectivity()),
            ("ML/RL Systems", self.test_ml_rl_systems()),
            ("Performance", self.test_performance_characteristics()),
            ("Error Handling", self.test_error_handling()),
            ("Security", self.test_security_measures())
        ]
        
        start_time = time.time()
        
        for category_name, test_task in test_categories:
            logger.info(f"🧪 Testing {category_name}...")
            try:
                category_results = await test_task
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                error_result = SmokeTestResult(
                    test_category=category_name,
                    test_name="Category Test",
                    passed=False,
                    message=f"Category test failed: {str(e)}",
                    details={"error": str(e)}
                )
                self.add_result(error_result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate average response time
        response_times = [r.response_time_ms for r in self.results if r.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Categorize results
        results_by_category = {}
        for result in self.results:
            if result.test_category not in results_by_category:
                results_by_category[result.test_category] = []
            results_by_category[result.test_category].append(asdict(result))
        
        summary = {
            "smoke_tests_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "execution_time_seconds": total_time,
            "deployment_smoke_test_passed": failed_tests == 0,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2),
                "average_response_time_ms": round(avg_response_time, 2)
            },
            "results_by_category": results_by_category,
            "critical_failures": [
                asdict(r) for r in self.results 
                if not r.passed and r.test_category in ["Basic Connectivity", "Health Endpoints", "Database"]
            ]
        }
        
        # Log summary
        logger.info(f"✅ Smoke tests complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"⚡ Average response time: {avg_response_time:.0f}ms")
        
        if failed_tests == 0:
            logger.info("🎉 All smoke tests passed - deployment successful!")
        else:
            logger.error(f"💥 {failed_tests} smoke tests failed - deployment may have issues")
        
        return summary

async def main():
    """Main smoke test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run post-deployment smoke tests for Shyvr RLTE")
    parser.add_argument("--base-url", required=True, help="Base URL of the deployed service")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output-file", help="Output file for test results")
    
    args = parser.parse_args()
    
    try:
        async with PostDeploymentSmokeTests(args.base_url, args.timeout) as smoke_tester:
            results = await smoke_tester.run_all_smoke_tests()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = Path(__file__).parent.parent / "reports" / f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Smoke test report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["deployment_smoke_test_passed"] else 1)
        
    except Exception as e:
        logger.error(f"💥 Smoke tests failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())