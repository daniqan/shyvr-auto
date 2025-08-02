"""
Model Registry for Model Preservation System

This module provides comprehensive model registry functionality including:
- Model registration and metadata management
- Schema validation and evolution tracking
- Performance tracking and regression detection
- Access control and approval workflows
- Environment promotion and deployment tracking
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

from .base import PreservationError, ModelMetadata
from .versioning import SemanticVersion


class ModelStatus(Enum):
    """Model status enumeration"""
    DRAFT = "draft"
    REGISTERED = "registered"
    VALIDATED = "validated"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ApprovalStatus(Enum):
    """Approval status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ModelRegistration:
    """Model registration information"""
    model_name: str
    model_type: str
    version: str
    author: str
    description: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    registration_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate registration ID"""
        if not self.registration_id:
            content = f"{self.model_name}:{self.version}:{self.created_at.isoformat()}"
            self.registration_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ModelSchema:
    """Model input/output schema definition"""
    model_name: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    schema_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate schema ID"""
        if not self.schema_id:
            content = f"{self.model_name}:{self.version}:{self.schema_version}"
            self.schema_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class PerformanceBaseline:
    """Performance baseline for model"""
    model_name: str
    version: str
    benchmark_dataset: str
    metrics: Dict[str, float]
    environment: str = "validation"
    established_at: datetime = field(default_factory=datetime.now)
    baseline_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate baseline ID"""
        if not self.baseline_id:
            content = f"{self.model_name}:{self.version}:{self.benchmark_dataset}"
            self.baseline_id = hashlib.sha256(content.encode()).hexdigest()[:16]


class ModelRegistrySchemas:
    """Schema validation and management for model registry"""
    
    def __init__(self):
        """Initialize schema manager"""
        self.schemas: Dict[str, ModelSchema] = {}
    
    def register_schema(
        self,
        model_name: str,
        version: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        schema_version: str = "1.0"
    ) -> str:
        """
        Register input/output schemas for a model
        
        Args:
            model_name: Name of the model
            version: Model version
            input_schema: JSON schema for model inputs
            output_schema: JSON schema for model outputs
            schema_version: Schema version
            
        Returns:
            Schema ID
        """
        schema = ModelSchema(
            model_name=model_name,
            version=version,
            input_schema=input_schema,
            output_schema=output_schema,
            schema_version=schema_version
        )
        
        schema_key = f"{model_name}:{version}"
        self.schemas[schema_key] = schema
        
        return schema.schema_id
    
    def validate_input(
        self,
        model_name: str,
        version: str,
        input_data: Dict[str, Any]
    ) -> bool:
        """
        Validate input data against registered schema
        
        Args:
            model_name: Name of the model
            version: Model version
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        schema_key = f"{model_name}:{version}"
        if schema_key not in self.schemas:
            raise ValueError(f"No schema found for {model_name}:{version}")
        
        schema = self.schemas[schema_key]
        return self._validate_against_schema(input_data, schema.input_schema)
    
    def validate_output(
        self,
        model_name: str,
        version: str,
        output_data: Dict[str, Any]
    ) -> bool:
        """
        Validate output data against registered schema
        
        Args:
            model_name: Name of the model
            version: Model version
            output_data: Output data to validate
            
        Returns:
            True if valid, False otherwise
        """
        schema_key = f"{model_name}:{version}"
        if schema_key not in self.schemas:
            raise ValueError(f"No schema found for {model_name}:{version}")
        
        schema = self.schemas[schema_key]
        return self._validate_against_schema(output_data, schema.output_schema)
    
    def check_schema_compatibility(
        self,
        model_name: str,
        old_version: str,
        new_version: str
    ) -> Dict[str, Any]:
        """
        Check schema compatibility between versions
        
        Args:
            model_name: Name of the model
            old_version: Old version
            new_version: New version
            
        Returns:
            Compatibility analysis results
        """
        old_key = f"{model_name}:{old_version}"
        new_key = f"{model_name}:{new_version}"
        
        if old_key not in self.schemas or new_key not in self.schemas:
            raise ValueError("Both schemas must be registered for compatibility check")
        
        old_schema = self.schemas[old_key]
        new_schema = self.schemas[new_key]
        
        input_compatibility = self._check_schema_evolution(
            old_schema.input_schema, new_schema.input_schema
        )
        output_compatibility = self._check_schema_evolution(
            old_schema.output_schema, new_schema.output_schema
        )
        
        return {
            "input_compatibility": input_compatibility,
            "output_compatibility": output_compatibility,
            "is_backward_compatible": (
                input_compatibility["backward_compatible"] and
                output_compatibility["backward_compatible"]
            ),
            "breaking_changes": (
                input_compatibility.get("breaking_changes", []) +
                output_compatibility.get("breaking_changes", [])
            )
        }
    
    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> bool:
        """
        Simple JSON schema validation
        
        Args:
            data: Data to validate
            schema: JSON schema
            
        Returns:
            True if valid, False otherwise
        """
        # This is a simplified implementation
        # In production, you'd use jsonschema library
        
        if schema.get("type") == "object":
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            
            # Check required fields
            for field in required:
                if field not in data:
                    return False
            
            # Check field types
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    if not self._validate_field_type(value, field_schema):
                        return False
        
        return True
    
    def _validate_field_type(self, value: Any, field_schema: Dict[str, Any]) -> bool:
        """Validate field type against schema"""
        expected_type = field_schema.get("type")
        
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        
        return True
    
    def _check_schema_evolution(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check schema evolution for breaking changes"""
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))
        
        old_properties = old_schema.get("properties", {})
        new_properties = new_schema.get("properties", {})
        
        breaking_changes = []
        
        # Check for removed required fields
        removed_required = old_required - new_required
        if removed_required:
            breaking_changes.append(f"Removed required fields: {removed_required}")
        
        # Check for added required fields (breaking for inputs)
        added_required = new_required - old_required
        if added_required:
            breaking_changes.append(f"Added required fields: {added_required}")
        
        # Check for type changes
        for field in old_properties:
            if field in new_properties:
                old_type = old_properties[field].get("type")
                new_type = new_properties[field].get("type")
                if old_type != new_type:
                    breaking_changes.append(f"Type changed for {field}: {old_type} -> {new_type}")
        
        return {
            "backward_compatible": len(breaking_changes) == 0,
            "breaking_changes": breaking_changes,
            "added_fields": set(new_properties.keys()) - set(old_properties.keys()),
            "removed_fields": set(old_properties.keys()) - set(new_properties.keys())
        }


class PerformanceTracker:
    """Performance tracking and regression detection"""
    
    def __init__(self):
        """Initialize performance tracker"""
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def establish_baseline(
        self,
        model_name: str,
        version: str,
        benchmark_dataset: str,
        metrics: Dict[str, float],
        environment: str = "validation"
    ) -> str:
        """
        Establish performance baseline for a model
        
        Args:
            model_name: Name of the model
            version: Model version
            benchmark_dataset: Dataset used for benchmarking
            metrics: Performance metrics
            environment: Environment where benchmark was run
            
        Returns:
            Baseline ID
        """
        baseline = PerformanceBaseline(
            model_name=model_name,
            version=version,
            benchmark_dataset=benchmark_dataset,
            metrics=metrics,
            environment=environment
        )
        
        baseline_key = f"{model_name}:{version}:{benchmark_dataset}"
        self.baselines[baseline_key] = baseline
        
        return baseline.baseline_id
    
    def detect_regression(
        self,
        model_name: str,
        new_version: str,
        baseline_version: str,
        threshold_percentage: float = 5.0,
        benchmark_dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect performance regression between versions
        
        Args:
            model_name: Name of the model
            new_version: New version to compare
            baseline_version: Baseline version
            threshold_percentage: Regression threshold as percentage
            benchmark_dataset: Specific dataset to compare (optional)
            
        Returns:
            Regression analysis results
        """
        # Find baseline
        baseline_key = None
        for key in self.baselines:
            parts = key.split(":")
            if (parts[0] == model_name and parts[1] == baseline_version and
                (benchmark_dataset is None or parts[2] == benchmark_dataset)):
                baseline_key = key
                break
        
        if not baseline_key:
            raise ValueError(f"No baseline found for {model_name}:{baseline_version}")
        
        baseline = self.baselines[baseline_key]
        
        # Get current performance data (would be passed as parameter in real implementation)
        # For now, we'll simulate this
        new_metrics = self._get_current_metrics(model_name, new_version)
        
        regressions = []
        improvements = []
        
        for metric, baseline_value in baseline.metrics.items():
            if metric in new_metrics:
                new_value = new_metrics[metric]
                
                # Calculate percentage change
                if baseline_value != 0:
                    percentage_change = ((new_value - baseline_value) / baseline_value) * 100
                else:
                    percentage_change = 0
                
                # Determine if this is a regression (assuming higher is better for most metrics)
                is_regression = percentage_change < -threshold_percentage
                is_improvement = percentage_change > threshold_percentage
                
                if is_regression:
                    regressions.append({
                        "metric": metric,
                        "baseline_value": baseline_value,
                        "new_value": new_value,
                        "percentage_change": percentage_change
                    })
                elif is_improvement:
                    improvements.append({
                        "metric": metric,
                        "baseline_value": baseline_value,
                        "new_value": new_value,
                        "percentage_change": percentage_change
                    })
        
        return {
            "has_regression": len(regressions) > 0,
            "regressions": regressions,
            "improvements": improvements,
            "baseline_version": baseline_version,
            "new_version": new_version,
            "threshold_percentage": threshold_percentage
        }
    
    def _get_current_metrics(self, model_name: str, version: str) -> Dict[str, float]:
        """Get current metrics for a model (placeholder implementation)"""
        # In a real implementation, this would load actual performance data
        return {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "inference_time_ms": 45,
            "memory_usage_mb": 128
        }


class ModelBenchmarker:
    """Automated model benchmarking"""
    
    def run_benchmark_suite(
        self,
        model_name: str,
        version: str,
        benchmark_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run benchmark suite for a model
        
        Args:
            model_name: Name of the model
            version: Model version
            benchmark_configs: List of benchmark configurations
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for config in benchmark_configs:
            benchmark_name = config["name"]
            benchmark_result = self._run_single_benchmark(model_name, version, config)
            results[benchmark_name] = benchmark_result
        
        return {
            "model_name": model_name,
            "version": version,
            "benchmark_results": results,
            "executed_at": datetime.now().isoformat(),
            "overall_score": self._calculate_overall_score(results)
        }
    
    def _run_single_benchmark(
        self,
        model_name: str,
        version: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single benchmark"""
        benchmark_name = config["name"]
        
        # Simulate benchmark execution
        if benchmark_name == "latency":
            return {
                "avg_latency_ms": 45,
                "p50_latency_ms": 42,
                "p95_latency_ms": 65,
                "p99_latency_ms": 95,
                "iterations": config.get("iterations", 1000)
            }
        elif benchmark_name == "throughput":
            return {
                "requests_per_second": 850,
                "concurrent_requests": config.get("concurrent_requests", [1, 10, 50]),
                "duration_seconds": config.get("duration_seconds", 60)
            }
        elif benchmark_name == "memory":
            return {
                "peak_memory_mb": 128,
                "avg_memory_mb": 115,
                "memory_efficiency": 0.85,
                "batch_sizes": config.get("batch_sizes", [1, 32, 128])
            }
        
        return {"status": "completed"}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall benchmark score"""
        # Simple scoring based on common performance metrics
        scores = []
        
        if "latency" in results:
            latency_score = max(0, 100 - results["latency"]["avg_latency_ms"])
            scores.append(latency_score)
        
        if "throughput" in results:
            throughput_score = min(100, results["throughput"]["requests_per_second"] / 10)
            scores.append(throughput_score)
        
        if "memory" in results:
            memory_score = results["memory"]["memory_efficiency"] * 100
            scores.append(memory_score)
        
        return sum(scores) / len(scores) if scores else 0.0


class ModelRegistry:
    """Main model registry for managing model metadata and lifecycle"""
    
    def __init__(self):
        """Initialize model registry"""
        self.registrations: Dict[str, ModelRegistration] = {}
        self.schema_manager = ModelRegistrySchemas()
        self.performance_tracker = PerformanceTracker()
        self.benchmarker = ModelBenchmarker()
        self.approvals: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, Dict[str, List[str]]] = {}
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        author: str,
        description: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model in the registry
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            version: Model version
            author: Model author
            description: Model description
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Registration ID
        """
        registration = ModelRegistration(
            model_name=model_name,
            model_type=model_type,
            version=version,
            author=author,
            description=description,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        registration_key = f"{model_name}:{version}"
        self.registrations[registration_key] = registration
        
        return registration.registration_id
    
    def search_models(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for models in the registry
        
        Args:
            query: Search query
            filters: Optional filters
            sort_by: Field to sort by
            sort_order: Sort order ("asc" or "desc")
            limit: Maximum number of results
            
        Returns:
            List of matching models
        """
        results = []
        
        for key, registration in self.registrations.items():
            # Simple text search
            if (query.lower() in registration.model_name.lower() or
                query.lower() in registration.description.lower() or
                any(query.lower() in tag.lower() for tag in registration.tags)):
                
                # Apply filters
                if self._apply_filters(registration, filters or {}):
                    results.append(self._registration_to_dict(registration))
        
        # Sort results
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            results.sort(key=lambda x: x["created_at"], reverse=reverse)
        elif sort_by == "model_name":
            results.sort(key=lambda x: x["model_name"], reverse=reverse)
        
        return results[:limit]
    
    def get_model_by_name(
        self,
        model_name: str,
        version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Get model by name and version
        
        Args:
            model_name: Name of the model
            version: Version (or "latest")
            
        Returns:
            Model information or None if not found
        """
        if version == "latest":
            # Find latest version
            latest_registration = None
            latest_version = None
            
            for key, registration in self.registrations.items():
                if registration.model_name == model_name:
                    try:
                        sem_version = SemanticVersion(registration.version)
                        if latest_version is None or sem_version > latest_version:
                            latest_version = sem_version
                            latest_registration = registration
                    except:
                        continue
            
            return self._registration_to_dict(latest_registration) if latest_registration else None
        else:
            registration_key = f"{model_name}:{version}"
            registration = self.registrations.get(registration_key)
            return self._registration_to_dict(registration) if registration else None
    
    def list_model_versions(
        self,
        model_name: str,
        include_deprecated: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a model
        
        Args:
            model_name: Name of the model
            include_deprecated: Whether to include deprecated versions
            
        Returns:
            List of model versions
        """
        versions = []
        
        for key, registration in self.registrations.items():
            if registration.model_name == model_name:
                if include_deprecated or registration.status != ModelStatus.DEPRECATED:
                    versions.append(self._registration_to_dict(registration))
        
        # Sort by version
        versions.sort(key=lambda x: x["version"], reverse=True)
        return versions
    
    def update_model_metadata(
        self,
        model_name: str,
        version: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update model metadata
        
        Args:
            model_name: Name of the model
            version: Model version
            metadata_updates: Metadata updates
            
        Returns:
            True if successful, False otherwise
        """
        registration_key = f"{model_name}:{version}"
        if registration_key not in self.registrations:
            return False
        
        registration = self.registrations[registration_key]
        registration.metadata.update(metadata_updates)
        registration.updated_at = datetime.now()
        
        return True
    
    def deprecate_model(
        self,
        model_name: str,
        version: str,
        reason: str,
        replacement_version: Optional[str] = None
    ) -> bool:
        """
        Deprecate a model version
        
        Args:
            model_name: Name of the model
            version: Version to deprecate
            reason: Deprecation reason
            replacement_version: Optional replacement version
            
        Returns:
            True if successful, False otherwise
        """
        registration_key = f"{model_name}:{version}"
        if registration_key not in self.registrations:
            return False
        
        registration = self.registrations[registration_key]
        registration.status = ModelStatus.DEPRECATED
        registration.metadata.update({
            "deprecation_reason": reason,
            "deprecated_at": datetime.now().isoformat()
        })
        
        if replacement_version:
            registration.metadata["replacement_version"] = replacement_version
        
        return True
    
    def get_model_lineage(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Get model lineage and provenance information
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Lineage information
        """
        registration_key = f"{model_name}:{version}"
        if registration_key not in self.registrations:
            raise ValueError(f"Model {model_name}:{version} not found")
        
        registration = self.registrations[registration_key]
        
        # In a real implementation, this would track actual lineage
        return {
            "model_name": model_name,
            "version": version,
            "author": registration.author,
            "created_at": registration.created_at.isoformat(),
            "parent_models": registration.metadata.get("parent_models", []),
            "training_data": registration.metadata.get("training_data", []),
            "code_version": registration.metadata.get("code_version"),
            "environment": registration.metadata.get("environment", {}),
            "lineage_graph": self._build_lineage_graph(model_name, version)
        }
    
    def _apply_filters(self, registration: ModelRegistration, filters: Dict[str, Any]) -> bool:
        """Apply filters to registration"""
        for filter_key, filter_value in filters.items():
            if filter_key == "model_type":
                if registration.model_type != filter_value:
                    return False
            elif filter_key == "tags":
                if not any(tag in registration.tags for tag in filter_value):
                    return False
            elif filter_key == "created_after":
                if registration.created_at < filter_value:
                    return False
        
        return True
    
    def _registration_to_dict(self, registration: Optional[ModelRegistration]) -> Optional[Dict[str, Any]]:
        """Convert registration to dictionary"""
        if not registration:
            return None
        
        return {
            "registration_id": registration.registration_id,
            "model_name": registration.model_name,
            "model_type": registration.model_type,
            "version": registration.version,
            "author": registration.author,
            "description": registration.description,
            "tags": registration.tags,
            "metadata": registration.metadata,
            "status": registration.status.value,
            "created_at": registration.created_at.isoformat(),
            "updated_at": registration.updated_at.isoformat()
        }
    
    def _build_lineage_graph(self, model_name: str, version: str) -> Dict[str, Any]:
        """Build lineage graph for model"""
        # Simplified lineage graph
        return {
            "nodes": [
                {"id": f"{model_name}:{version}", "type": "model"},
            ],
            "edges": []
        }