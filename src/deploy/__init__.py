"""
Dynamic Resource Allocation for Transformer Models

This module provides dynamic resource allocation capabilities for transformer
models in GCP Cloud Run deployment environments. It includes:

- DynamicResourceAllocator: Core resource allocation with transformer optimization
- TransformerResourceManager: Model-specific resource profiles and management
- ResourceScalingPolicy: Auto-scaling policies for transformer workloads
- ResourceMetricsCollector: Real-time resource usage metrics collection
- AutoScalingController: Automated scaling based on performance metrics
- MemoryManager: Memory optimization and pressure management
- CPUManager: CPU allocation and optimization strategies
- ContainerResourceManager: Container resource limits and configuration
"""

from .dynamic_resource_allocator import (
    DynamicResourceAllocator,
    TransformerResourceManager,
    ResourceScalingPolicy,
    ResourceMetricsCollector,
    AutoScalingController,
    ResourceOptimizer,
    MemoryManager,
    CPUManager,
    ContainerResourceManager
)

__all__ = [
    "DynamicResourceAllocator",
    "TransformerResourceManager", 
    "ResourceScalingPolicy",
    "ResourceMetricsCollector",
    "AutoScalingController",
    "ResourceOptimizer",
    "MemoryManager",
    "CPUManager",
    "ContainerResourceManager"
]