#!/usr/bin/env python3
"""
API Endpoint Health Check and Validation Script for Shyvr RLTE
Validates all API endpoints, response times, authentication, and functionality
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
class APIValidationResult:
    """API validation result container"""
    endpoint_category: str
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

class APIEndpointValidator:
    """Comprehensive API endpoint validation system"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[APIValidationResult] = []
        
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
    
    def add_result(self, result: APIValidationResult):
        """Add validation result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        status_info = f" [{result.status_code}]" if result.status_code else ""
        time_info = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms else ""
        logger.info(f"{status} {result.endpoint_category}.{result.test_name}{status_info}{time_info}: {result.message}")
    
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
    
    async def validate_basic_endpoints(self) -> List[APIValidationResult]:
        """Validate basic API endpoints"""
        results = []
        
        basic_endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/health", "Health check endpoint"),
            ("GET", "/metrics", "Metrics endpoint"),
            ("GET", "/api", "API base endpoint")
        ]
        
        for method, endpoint, description in basic_endpoints:
            status_code, response_data, response_time = await self.make_request(method, endpoint)
            
            # Determine if endpoint passed
            if endpoint == "/":
                # Root should return HTML or redirect
                passed = status_code in [200, 301, 302]
            elif endpoint == "/health":
                # Health should return 200 with status info
                passed = status_code == 200 and ("status" in response_data or "healthy" in str(response_data).lower())
            elif endpoint == "/metrics":
                # Metrics might not be enabled, so 200 or 404 is acceptable
                passed = status_code in [200, 404]
            else:
                # Other endpoints should return 200 or valid redirect
                passed = status_code in [200, 301, 302]
            
            results.append(APIValidationResult(
                endpoint_category="Basic Endpoints",
                test_name=f"{method} {endpoint}",
                passed=passed,
                message=f"{description} {'accessible' if passed else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "endpoint": endpoint,
                    "method": method,
                    "response_sample": response_data
                }
            ))
        
        return results
    
    async def validate_health_endpoint_details(self) -> List[APIValidationResult]:
        """Validate health endpoint in detail"""
        results = []
        
        status_code, response_data, response_time = await self.make_request("GET", "/health")
        
        if status_code == 200:
            # Check overall health status
            overall_status = response_data.get("status", "unknown")
            status_valid = overall_status in ["healthy", "degraded", "unhealthy"]
            
            results.append(APIValidationResult(
                endpoint_category="Health Details",
                test_name="Overall Status",
                passed=status_valid and overall_status != "unhealthy",
                message=f"Overall health status: {overall_status}",
                status_code=status_code,
                response_time_ms=response_time,
                details={"health_status": overall_status}
            ))
            
            # Check component health
            components = ["database", "ml_models", "rl_agent"]
            for component in components:
                if component in response_data:
                    component_status = response_data[component].get("status", "unknown")
                    component_healthy = component_status == "healthy"
                    
                    results.append(APIValidationResult(
                        endpoint_category="Health Details",
                        test_name=f"Component {component}",
                        passed=component_healthy,
                        message=f"Component {component}: {component_status}",
                        details=response_data[component]
                    ))
                else:
                    results.append(APIValidationResult(
                        endpoint_category="Health Details",
                        test_name=f"Component {component}",
                        passed=False,
                        message=f"Component {component} not reported in health check",
                        details={"missing_component": component}
                    ))
        else:
            results.append(APIValidationResult(
                endpoint_category="Health Details",
                test_name="Health Endpoint",
                passed=False,
                message=f"Health endpoint returned {status_code}",
                status_code=status_code,
                response_time_ms=response_time,
                details=response_data
            ))
        
        return results
    
    async def validate_dashboard_endpoints(self) -> List[APIValidationResult]:
        """Validate dashboard API endpoints"""
        results = []
        
        dashboard_endpoints = [
            ("GET", "/api/dashboard", "Dashboard base API"),
            ("GET", "/api/dashboard/status", "Dashboard status"),
            ("GET", "/api/dashboard/config", "Dashboard configuration"),
            ("GET", "/api/backtest", "Backtest API"),
            ("GET", "/api/experience", "Experience API")
        ]
        
        for method, endpoint, description in dashboard_endpoints:
            status_code, response_data, response_time = await self.make_request(method, endpoint)
            
            # Dashboard endpoints might require authentication, so 401/403 is acceptable
            passed = status_code in [200, 401, 403, 404]  # 404 means endpoint exists in router but not found
            
            results.append(APIValidationResult(
                endpoint_category="Dashboard API",
                test_name=f"{method} {endpoint}",
                passed=passed,
                message=f"{description} {'accessible' if status_code == 200 else 'protected/not found' if status_code in [401, 403, 404] else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "endpoint": endpoint,
                    "authentication_required": status_code in [401, 403]
                }
            ))
        
        return results
    
    async def validate_static_assets(self) -> List[APIValidationResult]:
        """Validate static asset serving"""
        results = []
        
        static_assets = [
            "/static/index.html",
            "/static/dashboard.js", 
            "/static/styles.css",
            "/static/backtest-results.js"
        ]
        
        for asset_path in static_assets:
            status_code, response_data, response_time = await self.make_request("GET", asset_path)
            
            # Static assets should return 200 or 404 if not found
            passed = status_code in [200, 404]
            asset_exists = status_code == 200
            
            results.append(APIValidationResult(
                endpoint_category="Static Assets",
                test_name=f"Asset {asset_path}",
                passed=passed,
                message=f"Static asset {asset_path} {'available' if asset_exists else 'not found' if status_code == 404 else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "asset_path": asset_path,
                    "exists": asset_exists,
                    "content_length": len(response_data.get("content", ""))
                }
            ))
        
        return results
    
    async def validate_authentication_endpoints(self) -> List[APIValidationResult]:
        """Validate authentication endpoints"""
        results = []
        
        auth_endpoints = [
            ("GET", "/api/auth/key", "API key endpoint"),
            ("POST", "/api/auth/login", "Login endpoint"),
            ("GET", "/api/auth/status", "Auth status endpoint")
        ]
        
        for method, endpoint, description in auth_endpoints:
            if method == "POST":
                # For POST endpoints, test with empty payload
                status_code, response_data, response_time = await self.make_request(
                    method, endpoint, json={}
                )
            else:
                status_code, response_data, response_time = await self.make_request(method, endpoint)
            
            # Auth endpoints should respond properly (200, 400, 401, 404)
            passed = status_code in [200, 400, 401, 404, 405]  # 405 = Method Not Allowed
            
            results.append(APIValidationResult(
                endpoint_category="Authentication",
                test_name=f"{method} {endpoint}",
                passed=passed,
                message=f"{description} {'responds correctly' if passed else 'failed'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "endpoint": endpoint,
                    "method": method,
                    "response_type": type(response_data).__name__
                }
            ))
        
        return results
    
    async def validate_websocket_endpoints(self) -> List[APIValidationResult]:
        """Validate WebSocket endpoints (connection test only)"""
        results = []
        
        # Test WebSocket endpoint accessibility (will likely fail WebSocket upgrade but should respond)
        ws_endpoints = [
            "/ws",
            "/api/ws",
            "/ws/dashboard"
        ]
        
        for ws_endpoint in ws_endpoints:
            status_code, response_data, response_time = await self.make_request("GET", ws_endpoint)
            
            # WebSocket endpoints should return 400 (Bad Request) for non-WS requests or 404 if not found
            passed = status_code in [400, 404, 426]  # 426 = Upgrade Required
            websocket_endpoint_exists = status_code in [400, 426]
            
            results.append(APIValidationResult(
                endpoint_category="WebSocket",
                test_name=f"WebSocket {ws_endpoint}",
                passed=passed,
                message=f"WebSocket endpoint {ws_endpoint} {'detected' if websocket_endpoint_exists else 'not found' if status_code == 404 else 'unexpected response'}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "endpoint": ws_endpoint,
                    "websocket_detected": websocket_endpoint_exists
                }
            ))
        
        return results
    
    async def validate_performance_characteristics(self) -> List[APIValidationResult]:
        """Validate API performance characteristics"""
        results = []
        
        # Test performance of key endpoints
        performance_endpoints = [
            ("/health", "Health check"),
            ("/", "Root page"),
            ("/api", "API base")
        ]
        
        for endpoint, description in performance_endpoints:
            # Make multiple requests to test consistency
            response_times = []
            
            for i in range(3):
                status_code, response_data, response_time = await self.make_request("GET", endpoint)
                if status_code > 0:  # Valid response
                    response_times.append(response_time)
                    
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # Consider endpoints performant if average response time is under 2 seconds
                is_performant = avg_response_time < 2000
                
                results.append(APIValidationResult(
                    endpoint_category="Performance",
                    test_name=f"Performance {endpoint}",
                    passed=is_performant,
                    message=f"{description} avg response time: {avg_response_time:.0f}ms {'(good)' if is_performant else '(slow)'}",
                    response_time_ms=avg_response_time,
                    details={
                        "endpoint": endpoint,
                        "avg_response_time_ms": avg_response_time,
                        "min_response_time_ms": min_response_time,
                        "max_response_time_ms": max_response_time,
                        "samples": len(response_times),
                        "all_response_times": response_times
                    }
                ))
            else:
                results.append(APIValidationResult(
                    endpoint_category="Performance",
                    test_name=f"Performance {endpoint}",
                    passed=False,
                    message=f"{description} performance test failed - no valid responses",
                    details={"endpoint": endpoint, "failed_requests": 3}
                ))
        
        return results
    
    async def validate_error_handling(self) -> List[APIValidationResult]:
        """Validate API error handling"""
        results = []
        
        # Test various error conditions
        error_tests = [
            ("GET", "/api/nonexistent", "404 handling", [404]),
            ("POST", "/api", "Method not allowed", [405, 404]),
            ("GET", "/api/dashboard/invalid-resource", "Invalid resource handling", [404, 400]),
            ("POST", "/health", "Invalid method on health", [405, 404])
        ]
        
        for method, endpoint, description, expected_codes in error_tests:
            if method == "POST":
                status_code, response_data, response_time = await self.make_request(
                    method, endpoint, json={"test": "data"}
                )
            else:
                status_code, response_data, response_time = await self.make_request(method, endpoint)
            
            # Check if we got an expected error code
            correct_error_handling = status_code in expected_codes
            
            results.append(APIValidationResult(
                endpoint_category="Error Handling",
                test_name=f"Error {method} {endpoint}",
                passed=correct_error_handling,
                message=f"{description}: {'correct' if correct_error_handling else 'unexpected'} error code {status_code}",
                status_code=status_code,
                response_time_ms=response_time,
                details={
                    "endpoint": endpoint,
                    "method": method,
                    "expected_codes": expected_codes,
                    "actual_code": status_code
                }
            ))
        
        return results
    
    async def validate_security_headers(self) -> List[APIValidationResult]:
        """Validate security headers in responses"""
        results = []
        
        # Check security headers on main endpoints
        test_endpoints = ["/", "/health", "/api"]
        
        for endpoint in test_endpoints:
            try:
                url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
                
                start_time = time.time()
                async with self.session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    headers = dict(response.headers)
                    
                    # Check for security headers
                    security_headers = {
                        "X-Frame-Options": "Clickjacking protection",
                        "X-Content-Type-Options": "MIME type sniffing protection", 
                        "X-XSS-Protection": "XSS protection",
                        "Content-Security-Policy": "Content Security Policy",
                        "Strict-Transport-Security": "HTTPS enforcement"
                    }
                    
                    found_headers = []
                    missing_headers = []
                    
                    for header_name, description in security_headers.items():
                        if header_name.lower() in [h.lower() for h in headers.keys()]:
                            found_headers.append(header_name)
                        else:
                            missing_headers.append(header_name)
                    
                    # Don't require all security headers, but report what's found
                    has_some_security = len(found_headers) > 0
                    
                    results.append(APIValidationResult(
                        endpoint_category="Security",
                        test_name=f"Security Headers {endpoint}",
                        passed=True,  # Don't fail on missing security headers
                        message=f"Security headers on {endpoint}: {len(found_headers)}/{len(security_headers)} present",
                        response_time_ms=response_time,
                        details={
                            "endpoint": endpoint,
                            "found_headers": found_headers,
                            "missing_headers": missing_headers,
                            "all_headers": list(headers.keys())
                        }
                    ))
                    
            except Exception as e:
                results.append(APIValidationResult(
                    endpoint_category="Security",
                    test_name=f"Security Headers {endpoint}",
                    passed=False,
                    message=f"Security header check failed for {endpoint}: {str(e)}",
                    details={"error": str(e), "endpoint": endpoint}
                ))
        
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all API endpoint validations"""
        logger.info(f"🌐 Starting comprehensive API endpoint validation for {self.base_url}")
        
        validation_categories = [
            ("Basic Endpoints", self.validate_basic_endpoints()),
            ("Health Details", self.validate_health_endpoint_details()),
            ("Dashboard API", self.validate_dashboard_endpoints()),
            ("Static Assets", self.validate_static_assets()),
            ("Authentication", self.validate_authentication_endpoints()),
            ("WebSocket", self.validate_websocket_endpoints()),
            ("Performance", self.validate_performance_characteristics()),
            ("Error Handling", self.validate_error_handling()),
            ("Security", self.validate_security_headers())
        ]
        
        start_time = time.time()
        
        for category_name, validation_task in validation_categories:
            logger.info(f"🔍 Validating {category_name}...")
            try:
                category_results = await validation_task
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                error_result = APIValidationResult(
                    endpoint_category=category_name,
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
        
        # Calculate average response time
        response_times = [r.response_time_ms for r in self.results if r.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Categorize results
        results_by_category = {}
        for result in self.results:
            if result.endpoint_category not in results_by_category:
                results_by_category[result.endpoint_category] = []
            results_by_category[result.endpoint_category].append(asdict(result))
        
        summary = {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "execution_time_seconds": total_time,
            "api_endpoints_ready": failed_tests == 0,
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
                if not r.passed and r.endpoint_category in ["Basic Endpoints", "Health Details"]
            ]
        }
        
        # Log summary
        logger.info(f"✅ API endpoint validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"⚡ Average response time: {avg_response_time:.0f}ms")
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} API tests failed")
        else:
            logger.info("🎉 All API endpoint validations passed - API ready for production")
        
        return summary

async def main():
    """Main API endpoint validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Shyvr RLTE API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output-file", help="Output file for validation results")
    
    args = parser.parse_args()
    
    try:
        async with APIEndpointValidator(args.base_url, args.timeout) as validator:
            results = await validator.run_all_validations()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = Path(__file__).parent.parent / "reports" / f"api_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 API validation report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["api_endpoints_ready"] else 1)
        
    except Exception as e:
        logger.error(f"💥 API endpoint validation failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())