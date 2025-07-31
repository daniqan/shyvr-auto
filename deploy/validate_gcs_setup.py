#!/usr/bin/env python3
"""
Shyvr RLTE - GCS Infrastructure Validation Script
Phase 1.1: GCS Infrastructure Setup Validation

This script validates that the GCS infrastructure has been set up correctly
for model preservation, including bucket configuration, permissions, and policies.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, Forbidden
    from google.api_core import exceptions as gcp_exceptions
except ImportError:
    print("ERROR: Google Cloud Storage client library not installed")
    print("Install with: pip install google-cloud-storage")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color


class GCSValidator:
    """GCS Infrastructure validation class"""
    
    def __init__(self):
        self.client = None
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.prod_bucket_name = os.getenv('GCS_PROD_BUCKET', 'shyvr-models-prod')
        self.staging_bucket_name = os.getenv('GCS_STAGING_BUCKET', 'shyvr-models-staging')
        self.service_account = os.getenv('CLOUD_RUN_SA')
        
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
        
    def log_info(self, message: str):
        """Log info message"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
        
    def log_success(self, message: str):
        """Log success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
        self.results['passed'] += 1
        
    def log_warning(self, message: str):
        """Log warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        self.results['warnings'] += 1
        
    def log_error(self, message: str):
        """Log error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        self.results['failed'] += 1
        
    def add_test_result(self, test_name: str, passed: bool, message: str):
        """Add test result to tracking"""
        self.results['tests'].append({
            'name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
    def initialize_client(self) -> bool:
        """Initialize GCS client"""
        try:
            self.client = storage.Client(project=self.project_id)
            self.log_success("GCS client initialized successfully")
            return True
        except Exception as e:
            self.log_error(f"Failed to initialize GCS client: {e}")
            return False
            
    def validate_bucket_existence(self, bucket_name: str) -> bool:
        """Validate that a bucket exists"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            self.log_success(f"✓ Bucket gs://{bucket_name} exists")
            self.add_test_result(f"bucket_exists_{bucket_name}", True, "Bucket exists")
            return True
        except NotFound:
            self.log_error(f"✗ Bucket gs://{bucket_name} does not exist")
            self.add_test_result(f"bucket_exists_{bucket_name}", False, "Bucket not found")
            return False
        except Exception as e:
            self.log_error(f"✗ Error checking bucket gs://{bucket_name}: {e}")
            self.add_test_result(f"bucket_exists_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_bucket_location(self, bucket_name: str, expected_location: str = "US-CENTRAL1") -> bool:
        """Validate bucket location"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            location = bucket.location
            if location.upper() == expected_location.upper():
                self.log_success(f"✓ Bucket gs://{bucket_name} location: {location}")
                self.add_test_result(f"bucket_location_{bucket_name}", True, f"Location: {location}")
                return True
            else:
                self.log_warning(f"⚠ Bucket gs://{bucket_name} location: {location} (expected: {expected_location})")
                self.add_test_result(f"bucket_location_{bucket_name}", False, f"Location mismatch: {location}")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking location for gs://{bucket_name}: {e}")
            self.add_test_result(f"bucket_location_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_versioning(self, bucket_name: str) -> bool:
        """Validate that versioning is enabled"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            if bucket.versioning_enabled:
                self.log_success(f"✓ Versioning enabled on gs://{bucket_name}")
                self.add_test_result(f"versioning_{bucket_name}", True, "Versioning enabled")
                return True
            else:
                self.log_error(f"✗ Versioning disabled on gs://{bucket_name}")
                self.add_test_result(f"versioning_{bucket_name}", False, "Versioning disabled")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking versioning for gs://{bucket_name}: {e}")
            self.add_test_result(f"versioning_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_lifecycle_policy(self, bucket_name: str) -> bool:
        """Validate lifecycle policy is configured"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            lifecycle_rules = bucket.lifecycle_rules
            if lifecycle_rules:
                self.log_success(f"✓ Lifecycle policy configured on gs://{bucket_name} ({len(lifecycle_rules)} rules)")
                self.add_test_result(f"lifecycle_{bucket_name}", True, f"{len(lifecycle_rules)} lifecycle rules")
                
                # Validate specific rules
                rule_types = [rule.get('action', {}).get('type') for rule in lifecycle_rules]
                expected_rules = ['SetStorageClass', 'Delete', 'AbortIncompleteMultipartUpload']
                
                for expected_rule in expected_rules:
                    if expected_rule in rule_types:
                        self.log_success(f"  ✓ {expected_rule} rule found")
                    else:
                        self.log_warning(f"  ⚠ {expected_rule} rule not found")
                        
                return True
            else:
                self.log_warning(f"⚠ No lifecycle policy on gs://{bucket_name}")
                self.add_test_result(f"lifecycle_{bucket_name}", False, "No lifecycle policy")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking lifecycle policy for gs://{bucket_name}: {e}")
            self.add_test_result(f"lifecycle_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_storage_class(self, bucket_name: str, expected_class: str = "STANDARD") -> bool:
        """Validate default storage class"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            storage_class = bucket.storage_class
            if storage_class == expected_class:
                self.log_success(f"✓ Storage class for gs://{bucket_name}: {storage_class}")
                self.add_test_result(f"storage_class_{bucket_name}", True, f"Storage class: {storage_class}")
                return True
            else:
                self.log_warning(f"⚠ Storage class for gs://{bucket_name}: {storage_class} (expected: {expected_class})")
                self.add_test_result(f"storage_class_{bucket_name}", False, f"Storage class mismatch: {storage_class}")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking storage class for gs://{bucket_name}: {e}")
            self.add_test_result(f"storage_class_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_uniform_bucket_access(self, bucket_name: str) -> bool:
        """Validate uniform bucket-level access is enabled"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            iam_config = bucket.iam_configuration
            if iam_config.uniform_bucket_level_access_enabled:
                self.log_success(f"✓ Uniform bucket-level access enabled on gs://{bucket_name}")
                self.add_test_result(f"uniform_access_{bucket_name}", True, "Uniform access enabled")
                return True
            else:
                self.log_warning(f"⚠ Uniform bucket-level access disabled on gs://{bucket_name}")
                self.add_test_result(f"uniform_access_{bucket_name}", False, "Uniform access disabled")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking uniform access for gs://{bucket_name}: {e}")
            self.add_test_result(f"uniform_access_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_public_access_prevention(self, bucket_name: str) -> bool:
        """Validate public access prevention is enforced"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            iam_config = bucket.iam_configuration
            if iam_config.public_access_prevention == "enforced":
                self.log_success(f"✓ Public access prevention enforced on gs://{bucket_name}")
                self.add_test_result(f"public_access_prevention_{bucket_name}", True, "Public access prevention enforced")
                return True
            else:
                self.log_warning(f"⚠ Public access prevention not enforced on gs://{bucket_name}")
                self.add_test_result(f"public_access_prevention_{bucket_name}", False, "Public access prevention not enforced")
                return False
        except Exception as e:
            self.log_error(f"✗ Error checking public access prevention for gs://{bucket_name}: {e}")
            self.add_test_result(f"public_access_prevention_{bucket_name}", False, f"Error: {e}")
            return False
            
    def test_read_write_permissions(self, bucket_name: str) -> bool:
        """Test read/write permissions on bucket"""
        try:
            bucket = self.client.bucket(bucket_name)
            
            # Test write permission
            test_blob_name = f"test/validation_{int(time.time())}.txt"
            test_content = f"GCS validation test - {datetime.now().isoformat()}"
            
            blob = bucket.blob(test_blob_name)
            blob.upload_from_string(test_content)
            self.log_success(f"✓ Write permission test passed for gs://{bucket_name}")
            
            # Test read permission
            downloaded_content = blob.download_as_text()
            if downloaded_content == test_content:
                self.log_success(f"✓ Read permission test passed for gs://{bucket_name}")
                
                # Test delete permission
                blob.delete()
                self.log_success(f"✓ Delete permission test passed for gs://{bucket_name}")
                
                self.add_test_result(f"permissions_{bucket_name}", True, "Read/write/delete permissions working")
                return True
            else:
                self.log_error(f"✗ Content mismatch in read test for gs://{bucket_name}")
                blob.delete()  # Cleanup
                self.add_test_result(f"permissions_{bucket_name}", False, "Content mismatch in read test")
                return False
                
        except Forbidden as e:
            self.log_error(f"✗ Permission denied for gs://{bucket_name}: {e}")
            self.add_test_result(f"permissions_{bucket_name}", False, f"Permission denied: {e}")
            return False
        except Exception as e:
            self.log_error(f"✗ Error testing permissions for gs://{bucket_name}: {e}")
            self.add_test_result(f"permissions_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_encryption(self, bucket_name: str) -> bool:
        """Validate encryption settings"""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.reload()
            
            # Check if default encryption is configured
            encryption_config = bucket.encryption_configuration
            if encryption_config:
                self.log_success(f"✓ Custom encryption configured on gs://{bucket_name}")
                self.add_test_result(f"encryption_{bucket_name}", True, "Custom encryption configured")
            else:
                self.log_info(f"ℹ Using Google-managed encryption on gs://{bucket_name}")
                self.add_test_result(f"encryption_{bucket_name}", True, "Google-managed encryption")
            
            return True
        except Exception as e:
            self.log_error(f"✗ Error checking encryption for gs://{bucket_name}: {e}")
            self.add_test_result(f"encryption_{bucket_name}", False, f"Error: {e}")
            return False
            
    def validate_monitoring_configuration(self) -> bool:
        """Validate monitoring configuration"""
        try:
            # This is a placeholder for monitoring validation
            # In a real implementation, you would check Cloud Monitoring metrics, alerts, etc.
            self.log_info("ℹ Monitoring configuration validation not implemented yet")
            self.add_test_result("monitoring", True, "Placeholder - monitoring validation needed")
            return True
        except Exception as e:
            self.log_error(f"✗ Error validating monitoring: {e}")
            self.add_test_result("monitoring", False, f"Error: {e}")
            return False
            
    def generate_report(self) -> Dict:
        """Generate validation report"""
        total_tests = self.results['passed'] + self.results['failed']
        success_rate = (self.results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': self.results['passed'],
                'failed': self.results['failed'],
                'warnings': self.results['warnings'],
                'success_rate': round(success_rate, 2)
            },
            'configuration': {
                'project_id': self.project_id,
                'prod_bucket': self.prod_bucket_name,
                'staging_bucket': self.staging_bucket_name,
                'service_account': self.service_account
            },
            'tests': self.results['tests']
        }
        
        return report
        
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        self.log_info("Starting GCS Infrastructure Validation")
        self.log_info("=" * 50)
        
        # Initialize client
        if not self.initialize_client():
            return False
            
        # Validate production bucket
        self.log_info(f"\n{Colors.CYAN}Validating Production Bucket: gs://{self.prod_bucket_name}{Colors.NC}")
        prod_bucket_exists = self.validate_bucket_existence(self.prod_bucket_name)
        
        if prod_bucket_exists:
            self.validate_bucket_location(self.prod_bucket_name)
            self.validate_versioning(self.prod_bucket_name)
            self.validate_lifecycle_policy(self.prod_bucket_name)
            self.validate_storage_class(self.prod_bucket_name, "STANDARD")
            self.validate_uniform_bucket_access(self.prod_bucket_name)
            self.validate_public_access_prevention(self.prod_bucket_name)
            self.validate_encryption(self.prod_bucket_name)
            self.test_read_write_permissions(self.prod_bucket_name)
            
        # Validate staging bucket
        self.log_info(f"\n{Colors.CYAN}Validating Staging Bucket: gs://{self.staging_bucket_name}{Colors.NC}")
        staging_bucket_exists = self.validate_bucket_existence(self.staging_bucket_name)
        
        if staging_bucket_exists:
            self.validate_bucket_location(self.staging_bucket_name)
            self.validate_versioning(self.staging_bucket_name)
            self.validate_lifecycle_policy(self.staging_bucket_name)
            # Staging bucket might use NEARLINE for cost optimization
            self.validate_storage_class(self.staging_bucket_name, "NEARLINE")
            self.validate_uniform_bucket_access(self.staging_bucket_name)
            self.validate_public_access_prevention(self.staging_bucket_name)
            self.validate_encryption(self.staging_bucket_name)
            self.test_read_write_permissions(self.staging_bucket_name)
            
        # Validate monitoring
        self.log_info(f"\n{Colors.CYAN}Validating Monitoring Configuration{Colors.NC}")
        self.validate_monitoring_configuration()
        
        # Generate and display report
        report = self.generate_report()
        
        self.log_info(f"\n{Colors.WHITE}Validation Summary{Colors.NC}")
        self.log_info("=" * 30)
        self.log_info(f"Total Tests: {report['summary']['total_tests']}")
        self.log_info(f"Passed: {Colors.GREEN}{report['summary']['passed']}{Colors.NC}")
        self.log_info(f"Failed: {Colors.RED}{report['summary']['failed']}{Colors.NC}")
        self.log_info(f"Warnings: {Colors.YELLOW}{report['summary']['warnings']}{Colors.NC}")
        self.log_info(f"Success Rate: {report['summary']['success_rate']}%")
        
        # Save detailed report
        report_file = f"gcs_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        self.log_info(f"Detailed report saved to: {report_file}")
        
        success = report['summary']['failed'] == 0
        if success:
            self.log_success("🎉 All validations passed! GCS infrastructure is ready.")
        else:
            self.log_error("❌ Some validations failed. Please review and fix issues.")
            
        return success


def main():
    """Main function"""
    validator = GCSValidator()
    
    # Check environment variables
    if not validator.project_id:
        print(f"{Colors.RED}ERROR:{Colors.NC} GCP_PROJECT_ID environment variable not set")
        sys.exit(1)
        
    try:
        success = validator.run_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}ERROR:{Colors.NC} Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()