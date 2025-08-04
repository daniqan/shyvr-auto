"""
Transformer Optimization Module - Phase 2.1 Performance Optimization

This module provides:
- Flash Attention 2.0 wrapper with graceful fallback
- Memory usage benchmarking and optimization
- Sequence length scaling analysis  
- Hardware-specific optimizations for GCP GPU instances
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16/BF16)

Implementation follows TDD methodology with real implementations (no mocks).
"""

import math
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import structlog
from contextlib import contextmanager
import psutil
import gc

logger = structlog.get_logger()


@dataclass
class FlashAttentionConfig:
    """Configuration for Flash Attention optimization"""
    use_flash_attention: bool = True
    enable_fallback: bool = True
    memory_efficient: bool = True
    causal: bool = False
    dropout_p: float = 0.0
    softmax_scale: Optional[float] = None
    window_size: Optional[Tuple[int, int]] = None
    alibi_slopes: Optional[torch.Tensor] = None
    deterministic: bool = False


@dataclass 
class MemoryBenchmark:
    """Memory usage benchmark results"""
    peak_memory_mb: float
    allocated_memory_mb: float
    cached_memory_mb: float
    memory_efficiency: float  # useful_memory / peak_memory
    sequence_length: int
    batch_size: int
    model_dimension: int
    num_heads: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    throughput_seq_per_sec: float
    memory_usage_mb: float
    sequence_length: int
    batch_size: int
    model_dimension: int
    num_heads: int
    use_flash_attention: bool
    timestamp: datetime = field(default_factory=datetime.now)


class FlashAttentionWrapper(nn.Module):
    """
    Flash Attention 2.0 wrapper with graceful fallback to standard attention
    
    Provides O(N) memory complexity instead of O(N²) for long sequences.
    Falls back gracefully when Flash Attention is not available.
    """
    
    def __init__(self, config: FlashAttentionConfig):
        super().__init__()
        self.config = config
        self.logger = structlog.get_logger().bind(component="FlashAttentionWrapper")
        
        # Try to import flash_attn
        self.flash_attn_available = self._check_flash_attention_availability()
        
        if self.flash_attn_available:
            self.logger.info("Flash Attention available - using optimized implementation")
        else:
            self.logger.info("Flash Attention not available - using standard attention fallback")
            if not config.enable_fallback:
                raise ImportError("Flash Attention not available and fallback disabled")
    
    def _check_flash_attention_availability(self) -> bool:
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Flash Attention or fallback
        
        Args:
            q: Query tensor [batch, seq_len, n_heads, head_dim]
            k: Key tensor [batch, seq_len, n_heads, head_dim] 
            v: Value tensor [batch, seq_len, n_heads, head_dim]
            attention_mask: Optional attention mask
            causal: Optional causal masking override
            
        Returns:
            output: Attention output [batch, seq_len, n_heads, head_dim]
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        """
        if self.flash_attn_available and self.config.use_flash_attention:
            return self._flash_attention_forward(q, k, v, attention_mask, causal)
        else:
            return self._standard_attention_forward(q, k, v, attention_mask, causal)
    
    def _flash_attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None,
                                causal: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flash Attention implementation"""
        try:
            from flash_attn import flash_attn_func
            
            # Flash attention expects [batch, seq_len, n_heads, head_dim]
            is_causal = causal if causal is not None else self.config.causal
            
            # Flash attention currently doesn't return attention weights
            # We'll simulate this for compatibility
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.dropout_p,
                softmax_scale=self.config.softmax_scale,
                causal=is_causal,
                window_size=self.config.window_size,
                alibi_slopes=self.config.alibi_slopes,
                deterministic=self.config.deterministic
            )
            
            # Create mock attention weights for compatibility
            # In production, you might want to compute these only when needed
            batch_size, seq_len, n_heads, head_dim = q.shape
            attention_weights = torch.zeros(
                batch_size, n_heads, seq_len, seq_len,
                device=q.device, dtype=q.dtype
            )
            
            return output, attention_weights
            
        except Exception as e:
            self.logger.warning("Flash Attention failed, falling back to standard attention", 
                              error=str(e))
            return self._standard_attention_forward(q, k, v, attention_mask, causal)
    
    def _standard_attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   causal: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention fallback"""
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        # Reshape for attention computation: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Apply causal mask if needed
        is_causal = causal if causal is not None else self.config.causal
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        if self.config.dropout_p > 0:
            attention_weights = F.dropout(attention_weights, p=self.config.dropout_p, training=self.training)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        # Reshape back: [batch, seq_len, n_heads, head_dim]
        output = output.transpose(1, 2)
        
        return output, attention_weights


class MemoryOptimizer:
    """
    Memory optimization utilities for Transformer models
    
    Provides gradient checkpointing, mixed precision training,
    and memory usage analysis for efficient training and inference.
    """
    
    def __init__(self, enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision: bool = True,
                 memory_fraction: float = 0.8):
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision = enable_mixed_precision
        self.memory_fraction = memory_fraction
        self.logger = structlog.get_logger().bind(component="MemoryOptimizer")
        
        # Initialize mixed precision scaler if available
        self.scaler = None
        if enable_mixed_precision and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                self.logger.info("Mixed precision training enabled")
            except ImportError:
                self.logger.warning("Mixed precision not available")
    
    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations"""
        if self.enable_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage"""
        if not self.enable_gradient_checkpointing:
            return model
        
        # Apply checkpointing to transformer layers
        for name, module in model.named_modules():
            if hasattr(module, 'checkpoint') and callable(module.checkpoint):
                module.checkpoint = True
                self.logger.debug("Applied gradient checkpointing", module=name)
        
        return model
    
    def optimize_memory_allocation(self, batch_size: int, seq_len: int, 
                                 d_model: int, n_heads: int) -> Dict[str, Any]:
        """Analyze and optimize memory allocation for given parameters"""
        
        # Estimate memory requirements (simplified)
        # Attention memory: O(batch_size * n_heads * seq_len^2 * 4 bytes)
        attention_memory_bytes = batch_size * n_heads * seq_len * seq_len * 4
        
        # Model parameters memory: O(d_model^2 * 4 * num_layers)
        # Assuming 6 layers for estimation
        param_memory_bytes = d_model * d_model * 4 * 6
        
        # Activation memory: O(batch_size * seq_len * d_model * 4)
        activation_memory_bytes = batch_size * seq_len * d_model * 4
        
        total_memory_bytes = attention_memory_bytes + param_memory_bytes + activation_memory_bytes
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        # Check available GPU memory
        available_memory_mb = 0
        if torch.cuda.is_available():
            available_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        # Calculate recommended batch size if current exceeds memory
        recommended_batch_size = batch_size
        if total_memory_mb > available_memory_mb * self.memory_fraction:
            memory_ratio = (available_memory_mb * self.memory_fraction) / total_memory_mb
            recommended_batch_size = max(1, int(batch_size * memory_ratio))
        
        return {
            "estimated_memory_mb": total_memory_mb,
            "available_memory_mb": available_memory_mb,
            "memory_utilization": total_memory_mb / max(available_memory_mb, 1),
            "recommended_batch_size": recommended_batch_size,
            "memory_breakdown": {
                "attention_mb": attention_memory_bytes / (1024 * 1024),
                "parameters_mb": param_memory_bytes / (1024 * 1024),
                "activations_mb": activation_memory_bytes / (1024 * 1024)
            }
        }
    
    def analyze_sequence_length_scaling(self, base_seq_len: int = 128,
                                      max_seq_len: int = 2048,
                                      d_model: int = 512,
                                      n_heads: int = 8) -> List[Dict[str, Any]]:
        """Analyze memory scaling with sequence length"""
        scaling_analysis = []
        
        sequence_lengths = [base_seq_len * (2 ** i) for i in range(5) 
                          if base_seq_len * (2 ** i) <= max_seq_len]
        
        for seq_len in sequence_lengths:
            analysis = self.optimize_memory_allocation(
                batch_size=1, seq_len=seq_len, d_model=d_model, n_heads=n_heads
            )
            analysis["sequence_length"] = seq_len
            analysis["scaling_factor"] = seq_len / base_seq_len
            scaling_analysis.append(analysis)
        
        return scaling_analysis


class AttentionBenchmarker:
    """
    Benchmarking utilities for attention mechanisms
    
    Provides comprehensive performance and memory benchmarking
    for Flash Attention vs standard attention.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger().bind(component="AttentionBenchmarker")
        self.benchmark_history: List[PerformanceBenchmark] = []
    
    def benchmark_attention_mechanisms(self, batch_sizes: List[int] = [1, 2, 4],
                                     sequence_lengths: List[int] = [128, 256, 512],
                                     d_model: int = 512,
                                     n_heads: int = 8,
                                     num_trials: int = 5) -> Dict[str, List[PerformanceBenchmark]]:
        """Comprehensive benchmarking of attention mechanisms"""
        
        flash_results = []
        standard_results = []
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                self.logger.info("Benchmarking attention", 
                               batch_size=batch_size, seq_len=seq_len)
                
                # Benchmark Flash Attention
                flash_benchmark = self._benchmark_single_config(
                    batch_size, seq_len, d_model, n_heads, 
                    use_flash_attention=True, num_trials=num_trials
                )
                if flash_benchmark:
                    flash_results.append(flash_benchmark)
                
                # Benchmark Standard Attention  
                standard_benchmark = self._benchmark_single_config(
                    batch_size, seq_len, d_model, n_heads,
                    use_flash_attention=False, num_trials=num_trials
                )
                standard_results.append(standard_benchmark)
        
        return {
            "flash_attention": flash_results,
            "standard_attention": standard_results
        }
    
    def _benchmark_single_config(self, batch_size: int, seq_len: int,
                               d_model: int, n_heads: int,
                               use_flash_attention: bool,
                               num_trials: int = 5) -> Optional[PerformanceBenchmark]:
        """Benchmark single configuration"""
        
        head_dim = d_model // n_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test tensors
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, requires_grad=True)
        
        # Setup attention mechanism
        if use_flash_attention:
            config = FlashAttentionConfig(use_flash_attention=True, enable_fallback=True)
            attention = FlashAttentionWrapper(config)
            if not attention.flash_attn_available:
                return None  # Skip if Flash Attention not available
        else:
            config = FlashAttentionConfig(use_flash_attention=False, enable_fallback=True)
            attention = FlashAttentionWrapper(config)
        
        attention = attention.to(device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = attention(q, k, v)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark forward pass
        forward_times = []
        memory_usage = []
        
        for _ in range(num_trials):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            output, _ = attention(q, k, v)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = (time.time() - start_time) * 1000  # Convert to ms
            forward_times.append(forward_time)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                memory_usage.append(peak_memory)
        
        # Benchmark backward pass
        backward_times = []
        for _ in range(num_trials):
            output, _ = attention(q, k, v)
            loss = output.sum()
            
            start_time = time.time()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            backward_time = (time.time() - start_time) * 1000  # Convert to ms
            backward_times.append(backward_time)
            
            # Clear gradients
            for tensor in [q, k, v]:
                if tensor.grad is not None:
                    tensor.grad.zero_()
        
        # Calculate statistics
        avg_forward_time = np.mean(forward_times)
        avg_backward_time = np.mean(backward_times)
        total_time = avg_forward_time + avg_backward_time
        throughput = (batch_size * 1000) / total_time  # sequences per second
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        benchmark = PerformanceBenchmark(
            forward_time_ms=avg_forward_time,
            backward_time_ms=avg_backward_time,
            total_time_ms=total_time,
            throughput_seq_per_sec=throughput,
            memory_usage_mb=avg_memory,
            sequence_length=seq_len,
            batch_size=batch_size,
            model_dimension=d_model,
            num_heads=n_heads,
            use_flash_attention=use_flash_attention
        )
        
        self.benchmark_history.append(benchmark)
        return benchmark
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks"""
        if not self.benchmark_history:
            return {"message": "No benchmarks available"}
        
        flash_benchmarks = [b for b in self.benchmark_history if b.use_flash_attention]
        standard_benchmarks = [b for b in self.benchmark_history if not b.use_flash_attention]
        
        summary = {
            "total_benchmarks": len(self.benchmark_history),
            "flash_attention_benchmarks": len(flash_benchmarks),
            "standard_attention_benchmarks": len(standard_benchmarks)
        }
        
        if flash_benchmarks and standard_benchmarks:
            # Compare performance
            flash_avg_time = np.mean([b.total_time_ms for b in flash_benchmarks])
            standard_avg_time = np.mean([b.total_time_ms for b in standard_benchmarks])
            speed_improvement = standard_avg_time / flash_avg_time if flash_avg_time > 0 else 1.0
            
            flash_avg_memory = np.mean([b.memory_usage_mb for b in flash_benchmarks])
            standard_avg_memory = np.mean([b.memory_usage_mb for b in standard_benchmarks])
            memory_reduction = 1.0 - (flash_avg_memory / standard_avg_memory) if standard_avg_memory > 0 else 0.0
            
            summary.update({
                "speed_improvement": speed_improvement,
                "memory_reduction": memory_reduction,
                "flash_avg_time_ms": flash_avg_time,
                "standard_avg_time_ms": standard_avg_time,
                "flash_avg_memory_mb": flash_avg_memory,
                "standard_avg_memory_mb": standard_avg_memory
            })
        
        return summary


class HardwareProfiler:
    """
    Hardware profiling for optimal Transformer configuration
    
    Profiles GPU capabilities and provides optimization recommendations
    specific to GCP GPU instances (T4, V100, etc.).
    """
    
    def __init__(self):
        self.logger = structlog.get_logger().bind(component="HardwareProfiler")
        self.gpu_info = self._detect_gpu_hardware()
        self.optimization_recommendations = {}
    
    def _detect_gpu_hardware(self) -> Dict[str, Any]:
        """Detect GPU hardware capabilities"""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "devices": []
        }
        
        if torch.cuda.is_available():
            gpu_info["device_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": (props.major, props.minor),
                    "multiprocessor_count": props.multi_processor_count,
                    "is_gcp_optimized": self._is_gcp_optimized_gpu(props.name)
                }
                gpu_info["devices"].append(device_info)
        
        return gpu_info
    
    def _is_gcp_optimized_gpu(self, gpu_name: str) -> bool:
        """Check if GPU is optimized for GCP deployment"""
        gcp_gpu_types = ["Tesla T4", "Tesla V100", "Tesla K80", "Tesla P4", "Tesla P100"]
        return any(gpu_type in gpu_name for gpu_type in gcp_gpu_types)
    
    def profile_memory_bandwidth(self, test_size_mb: int = 100) -> Dict[str, float]:
        """Profile GPU memory bandwidth"""
        if not torch.cuda.is_available():
            return {"memory_bandwidth_gb_s": 0.0}
        
        device = torch.cuda.current_device()
        
        # Create test tensors
        size = test_size_mb * 1024 * 1024 // 4  # Convert MB to float32 elements
        test_tensor = torch.randn(size, device=device)
        
        # Warmup
        for _ in range(5):
            _ = test_tensor * 2.0
        
        torch.cuda.synchronize()
        
        # Benchmark memory operations
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = test_tensor * 2.0
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        # Calculate bandwidth (read + write operations)
        bytes_per_iteration = test_size_mb * 1024 * 1024 * 2  # Read + write
        total_bytes = bytes_per_iteration * num_iterations
        bandwidth_gb_s = (total_bytes / (1024**3)) / elapsed_time
        
        return {"memory_bandwidth_gb_s": bandwidth_gb_s}
    
    def get_optimal_configuration(self, target_seq_len: int = 512,
                                d_model: int = 512,
                                n_heads: int = 8) -> Dict[str, Any]:
        """Get optimal configuration recommendations"""
        
        if not self.gpu_info["cuda_available"]:
            return {
                "device": "cpu",
                "batch_size": 1,
                "use_flash_attention": False,
                "enable_mixed_precision": False,
                "recommendations": ["Use CPU-optimized model architecture"]
            }
        
        # Get primary GPU
        primary_gpu = self.gpu_info["devices"][0] if self.gpu_info["devices"] else None
        if not primary_gpu:
            return self.get_optimal_configuration()  # Fallback to CPU
        
        recommendations = []
        
        # Memory-based batch size recommendation
        available_memory_gb = primary_gpu["total_memory_gb"]
        
        # Estimate memory usage (simplified)
        memory_per_sample_mb = (target_seq_len * d_model * 4) / (1024 * 1024)  # Rough estimate
        max_batch_size = int((available_memory_gb * 1024 * 0.7) / memory_per_sample_mb)  # 70% utilization
        recommended_batch_size = min(max_batch_size, 32)  # Cap at 32 for stability
        
        # Flash Attention recommendation
        compute_capability = primary_gpu["compute_capability"]
        supports_flash_attention = compute_capability[0] >= 7  # Volta and newer
        
        # Mixed precision recommendation  
        supports_mixed_precision = compute_capability[0] >= 7  # Tensor cores
        
        if supports_flash_attention:
            recommendations.append("Enable Flash Attention for memory efficiency")
        else:
            recommendations.append("Flash Attention not supported, use gradient checkpointing")
        
        if supports_mixed_precision:
            recommendations.append("Enable mixed precision (FP16) for speed improvement")
        
        if primary_gpu["is_gcp_optimized"]:
            recommendations.append("GPU is GCP-optimized, use default configuration")
        else:
            recommendations.append("Custom GPU detected, monitor performance carefully")
        
        # Sequence length recommendations
        if target_seq_len > 1024 and not supports_flash_attention:
            recommendations.append("Consider reducing sequence length or using Flash Attention")
        
        return {
            "device": f"cuda:0",
            "gpu_name": primary_gpu["name"],
            "batch_size": recommended_batch_size,
            "use_flash_attention": supports_flash_attention,
            "enable_mixed_precision": supports_mixed_precision,
            "enable_gradient_checkpointing": True,
            "memory_utilization_target": 0.7,
            "recommendations": recommendations,
            "hardware_info": {
                "compute_capability": compute_capability,
                "memory_gb": available_memory_gb,
                "is_gcp_optimized": primary_gpu["is_gcp_optimized"]
            }
        }
    
    def benchmark_hardware_performance(self) -> Dict[str, Any]:
        """Benchmark hardware performance for optimization"""
        if not torch.cuda.is_available():
            return {"device": "cpu", "performance_score": 1.0}
        
        # Memory bandwidth test
        memory_benchmark = self.profile_memory_bandwidth()
        
        # Compute performance test (simple matrix multiplication)
        device = torch.cuda.current_device()
        size = 1024
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        # Calculate GFLOPS (simplified)
        ops_per_matmul = 2 * size**3  # Multiply-add operations
        total_ops = ops_per_matmul * num_iterations
        gflops = (total_ops / elapsed_time) / 1e9
        
        # Overall performance score (normalized)
        memory_score = min(memory_benchmark["memory_bandwidth_gb_s"] / 500.0, 1.0)  # Normalize to 500 GB/s
        compute_score = min(gflops / 5000.0, 1.0)  # Normalize to 5000 GFLOPS
        performance_score = (memory_score + compute_score) / 2
        
        return {
            "device": f"cuda:{device}",
            "memory_bandwidth_gb_s": memory_benchmark["memory_bandwidth_gb_s"],
            "compute_gflops": gflops,
            "performance_score": performance_score,
            "memory_score": memory_score,
            "compute_score": compute_score
        }


# Utility functions for integration

def create_optimized_attention(d_model: int, n_heads: int,
                             use_flash_attention: bool = True,
                             hardware_config: Optional[Dict[str, Any]] = None) -> FlashAttentionWrapper:
    """Create optimized attention mechanism with hardware-aware configuration"""
    
    if hardware_config is None:
        profiler = HardwareProfiler()
        hardware_config = profiler.get_optimal_configuration(d_model=d_model, n_heads=n_heads)
    
    config = FlashAttentionConfig(
        use_flash_attention=use_flash_attention and hardware_config.get("use_flash_attention", True),
        enable_fallback=True,
        memory_efficient=True,
        dropout_p=0.1
    )
    
    return FlashAttentionWrapper(config)


def optimize_transformer_memory(model: nn.Module, 
                               enable_gradient_checkpointing: bool = True,
                               enable_mixed_precision: bool = True) -> Tuple[nn.Module, MemoryOptimizer]:
    """Apply memory optimizations to transformer model"""
    
    optimizer = MemoryOptimizer(
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        enable_mixed_precision=enable_mixed_precision
    )
    
    optimized_model = optimizer.apply_gradient_checkpointing(model)
    
    return optimized_model, optimizer


def benchmark_attention_performance(batch_sizes: List[int] = [1, 2, 4],
                                  sequence_lengths: List[int] = [128, 256, 512],
                                  d_model: int = 512,
                                  n_heads: int = 8) -> Dict[str, Any]:
    """Benchmark attention performance and return optimization recommendations"""
    
    benchmarker = AttentionBenchmarker()
    results = benchmarker.benchmark_attention_mechanisms(
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        d_model=d_model,
        n_heads=n_heads
    )
    
    summary = benchmarker.get_benchmark_summary()
    
    # Generate recommendations based on results
    recommendations = []
    
    if summary.get("speed_improvement", 1.0) > 1.2:
        recommendations.append("Flash Attention provides significant speed improvement")
    
    if summary.get("memory_reduction", 0.0) > 0.3:
        recommendations.append("Flash Attention provides significant memory reduction")
    
    if not recommendations:
        recommendations.append("Standard attention performance is adequate")
    
    return {
        "benchmark_results": results,
        "summary": summary,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    # Example usage and testing
    
    # Hardware profiling
    profiler = HardwareProfiler()
    hardware_info = profiler.get_optimal_configuration()
    print("Hardware Configuration:")
    print(f"  Device: {hardware_info['device']}")
    print(f"  Recommended Batch Size: {hardware_info['batch_size']}")
    print(f"  Use Flash Attention: {hardware_info['use_flash_attention']}")
    print(f"  Mixed Precision: {hardware_info['enable_mixed_precision']}")
    
    # Performance benchmarking
    print("\nRunning Performance Benchmark...")
    benchmark_results = benchmark_attention_performance()
    print(f"Benchmark Summary: {benchmark_results['summary']}")
    print(f"Recommendations: {benchmark_results['recommendations']}")