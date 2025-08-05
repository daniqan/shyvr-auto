"""
Transformer Model Preservation Framework

This module provides specialized preservation functionality for transformer models
including attention weight preservation, architecture versioning, and migration support.

Features:
- Efficient serialization/deserialization of transformer models
- Attention pattern preservation and analysis
- Large model sharding for models > 2GB
- Architecture versioning and compatibility checking
- Migration support between transformer architectures
- GCP-optimized storage and loading
"""

import asyncio
import hashlib
import json
import pickle
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

from .base import (
    ModelMetadata, 
    PreservationError, 
    PreservationPriority,
    ModelState,
    generate_model_id,
    calculate_checksum
)
# Import TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import PreservationManager


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    SPARSE_ATTENTION = "sparse_attention"
    CAUSAL_ATTENTION = "causal_attention"


class TransformerArchitecture(Enum):
    """Supported transformer architectures"""
    VANILLA_TRANSFORMER = "vanilla_transformer"
    ITRANSFORMER = "itransformer"
    PATCHTST = "patchtst"
    TIMESMIXER = "timesmixer"
    TIMESFM = "timesfm"
    CUSTOM = "custom"


@dataclass
class AttentionWeights:
    """Container for attention weights with metadata"""
    layer_idx: int
    head_idx: int
    attention_type: AttentionType
    weights: np.ndarray
    input_sequence_length: int
    output_sequence_length: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get attention weight shape"""
        return self.weights.shape
    
    @property
    def entropy(self) -> float:
        """Calculate attention entropy"""
        # Normalize weights to probabilities
        probs = self.weights / (self.weights.sum(axis=-1, keepdims=True) + 1e-8)
        # Calculate entropy
        log_probs = np.log(probs + 1e-8)
        entropy = -np.sum(probs * log_probs, axis=-1).mean()
        return float(entropy)
    
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Extract attention pattern characteristics"""
        return {
            "entropy": self.entropy,
            "max_attention": float(self.weights.max()),
            "min_attention": float(self.weights.min()),
            "attention_spread": float(self.weights.std()),
            "diagonal_dominance": self._calculate_diagonal_dominance(),
            "locality_score": self._calculate_locality_score()
        }
    
    def _calculate_diagonal_dominance(self) -> float:
        """Calculate how much attention focuses on diagonal (self-attention)"""
        if self.weights.ndim < 2:
            return 0.0
        
        min_dim = min(self.weights.shape[-2:])
        diagonal_sum = np.trace(self.weights[..., :min_dim, :min_dim], axis1=-2, axis2=-1)
        total_sum = self.weights.sum(axis=(-2, -1))
        
        return float(np.mean(diagonal_sum / (total_sum + 1e-8)))
    
    def _calculate_locality_score(self) -> float:
        """Calculate how local vs global the attention patterns are"""
        if self.weights.ndim < 2 or self.weights.shape[-1] < 3:
            return 0.0
        
        # Calculate attention within a local window (±2 positions)
        local_attention = 0.0
        seq_len = self.weights.shape[-1]
        
        for i in range(seq_len):
            start = max(0, i - 2)
            end = min(seq_len, i + 3)
            local_attention += self.weights[..., i, start:end].sum()
        
        total_attention = self.weights.sum()
        return float(local_attention / (total_attention + 1e-8))


@dataclass
class TransformerMetadata:
    """Extended metadata for transformer models"""
    architecture: TransformerArchitecture
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    attention_types: List[AttentionType] = field(default_factory=list)
    layer_configs: List[Dict[str, Any]] = field(default_factory=list)
    attention_patterns_preserved: bool = False
    model_shards: List[str] = field(default_factory=list)
    total_parameters: int = 0
    activation_function: str = "gelu"
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    compatibility_version: str = "1.0"
    
    def get_architecture_signature(self) -> str:
        """Get unique signature for architecture compatibility"""
        signature_data = {
            "architecture": self.architecture.value,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_types": [at.value for at in self.attention_types]
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]


class AttentionPreservationManager:
    """Manager for preserving and analyzing attention weights"""
    
    def __init__(self):
        self.attention_cache: Dict[str, List[AttentionWeights]] = {}
        self.pattern_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def extract_attention_weights(
        self,
        model: nn.Module,
        model_id: str,
        input_data: Optional[torch.Tensor] = None
    ) -> List[AttentionWeights]:
        """
        Extract attention weights from transformer model
        
        Args:
            model: Transformer model
            model_id: Unique model identifier
            input_data: Optional input data for attention computation
            
        Returns:
            List of AttentionWeights objects
        """
        attention_weights = []
        
        # Set model to evaluation mode
        model.eval()
        
        # Hook function to capture attention weights
        def attention_hook(module, input_data, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                # Handle different attention output formats
                attentions = output.attentions
                if isinstance(attentions, tuple):
                    attentions = attentions[0]
                
                # Extract weights from each head
                if attentions.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    batch_size, num_heads, seq_len, _ = attentions.shape
                    
                    for head_idx in range(num_heads):
                        attention_weight = AttentionWeights(
                            layer_idx=getattr(module, '_layer_idx', 0),
                            head_idx=head_idx,
                            attention_type=AttentionType.MULTI_HEAD,
                            weights=attentions[0, head_idx].detach().cpu().numpy(),
                            input_sequence_length=seq_len,
                            output_sequence_length=seq_len
                        )
                        attention_weights.append(attention_weight)
        
        # Register hooks for attention layers
        hooks = []
        layer_idx = 0
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                module._layer_idx = layer_idx
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
                layer_idx += 1
        
        try:
            # Run forward pass to capture attention weights
            if input_data is not None:
                with torch.no_grad():
                    model(input_data)
            
            # Cache attention weights
            self.attention_cache[model_id] = attention_weights
            
            return attention_weights
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    async def preserve_attention_patterns(
        self,
        model_id: str,
        attention_weights: List[AttentionWeights],
        storage_path: str
    ) -> str:
        """
        Preserve attention patterns to storage
        
        Args:
            model_id: Model identifier
            attention_weights: List of attention weights to preserve
            storage_path: Base storage path
            
        Returns:
            Path to preserved attention patterns
        """
        patterns_data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "attention_patterns": []
        }
        
        for attention_weight in attention_weights:
            pattern_data = {
                "layer_idx": attention_weight.layer_idx,
                "head_idx": attention_weight.head_idx,
                "attention_type": attention_weight.attention_type.value,
                "shape": attention_weight.shape,
                "patterns": attention_weight.get_attention_patterns(),
                "weights_compressed": self._compress_attention_weights(attention_weight.weights)
            }
            patterns_data["attention_patterns"].append(pattern_data)
        
        # Serialize and compress the patterns data
        serialized_data = pickle.dumps(patterns_data)
        compressed_data = zlib.compress(serialized_data, level=6)
        
        # Save to storage path
        attention_path = f"{storage_path}/attention_patterns.pkl.gz"
        
        # In a real implementation, this would save to GCS
        # For now, we'll return the path and data size
        return attention_path
    
    def _compress_attention_weights(self, weights: np.ndarray) -> bytes:
        """Compress attention weights using quantization and compression"""
        # Quantize to 16-bit float to reduce size
        quantized = weights.astype(np.float16)
        # Serialize and compress
        serialized = pickle.dumps(quantized)
        compressed = zlib.compress(serialized, level=9)
        return compressed
    
    async def analyze_attention_evolution(
        self,
        model_id: str,
        versions: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze how attention patterns evolve across model versions
        
        Args:
            model_id: Base model identifier
            versions: List of model versions to compare
            
        Returns:
            Analysis of attention pattern evolution
        """
        evolution_analysis = {
            "model_id": model_id,
            "versions_analyzed": versions,
            "pattern_changes": [],
            "stability_metrics": {},
            "anomaly_detection": {}
        }
        
        # Compare attention patterns across versions
        for i in range(len(versions) - 1):
            current_version = versions[i]
            next_version = versions[i + 1]
            
            # Get attention patterns for both versions
            current_patterns = self.pattern_history.get(f"{model_id}-{current_version}", [])
            next_patterns = self.pattern_history.get(f"{model_id}-{next_version}", [])
            
            if current_patterns and next_patterns:
                change_analysis = self._analyze_pattern_changes(current_patterns, next_patterns)
                evolution_analysis["pattern_changes"].append({
                    "from_version": current_version,
                    "to_version": next_version,
                    "changes": change_analysis
                })
        
        return evolution_analysis
    
    def _analyze_pattern_changes(
        self,
        patterns1: List[Dict[str, Any]],
        patterns2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze changes between two sets of attention patterns"""
        changes = {
            "entropy_change": 0.0,
            "locality_change": 0.0,
            "attention_spread_change": 0.0,
            "significant_changes": []
        }
        
        # Compare patterns layer by layer
        min_layers = min(len(patterns1), len(patterns2))
        
        for i in range(min_layers):
            p1 = patterns1[i]["patterns"]
            p2 = patterns2[i]["patterns"]
            
            entropy_diff = p2["entropy"] - p1["entropy"]
            locality_diff = p2["locality_score"] - p1["locality_score"]
            spread_diff = p2["attention_spread"] - p1["attention_spread"]
            
            changes["entropy_change"] += entropy_diff
            changes["locality_change"] += locality_diff
            changes["attention_spread_change"] += spread_diff
            
            # Flag significant changes
            if abs(entropy_diff) > 0.5:
                changes["significant_changes"].append(f"Layer {i}: Large entropy change ({entropy_diff:.3f})")
            if abs(locality_diff) > 0.2:
                changes["significant_changes"].append(f"Layer {i}: Locality change ({locality_diff:.3f})")
        
        # Average changes across layers
        changes["entropy_change"] /= min_layers
        changes["locality_change"] /= min_layers
        changes["attention_spread_change"] /= min_layers
        
        return changes


class AttentionPatternAnalyzer:
    """Analyzer for attention patterns and behaviors"""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_attention_patterns(
        self,
        attention_weights: List[AttentionWeights],
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of attention patterns
        
        Args:
            attention_weights: List of attention weights
            analysis_config: Optional configuration for analysis
            
        Returns:
            Detailed attention pattern analysis
        """
        config = analysis_config or {}
        
        analysis = {
            "summary": {},
            "layer_analysis": [],
            "head_analysis": [],
            "pattern_types": {},
            "anomalies": []
        }
        
        # Analyze each layer
        layers = {}
        for attention_weight in attention_weights:
            layer_idx = attention_weight.layer_idx
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(attention_weight)
        
        for layer_idx, layer_weights in layers.items():
            layer_analysis = await self._analyze_layer_patterns(layer_idx, layer_weights)
            analysis["layer_analysis"].append(layer_analysis)
        
        # Generate summary statistics
        analysis["summary"] = self._generate_summary_statistics(attention_weights)
        
        # Detect pattern types
        analysis["pattern_types"] = self._classify_attention_patterns(attention_weights)
        
        # Detect anomalies
        analysis["anomalies"] = self._detect_attention_anomalies(attention_weights)
        
        return analysis
    
    async def _analyze_layer_patterns(
        self,
        layer_idx: int,
        layer_weights: List[AttentionWeights]
    ) -> Dict[str, Any]:
        """Analyze attention patterns for a specific layer"""
        layer_analysis = {
            "layer_idx": layer_idx,
            "num_heads": len(layer_weights),
            "head_patterns": [],
            "layer_statistics": {}
        }
        
        entropies = []
        localities = []
        spreads = []
        
        for attention_weight in layer_weights:
            patterns = attention_weight.get_attention_patterns()
            entropies.append(patterns["entropy"])
            localities.append(patterns["locality_score"])
            spreads.append(patterns["attention_spread"])
            
            head_analysis = {
                "head_idx": attention_weight.head_idx,
                "patterns": patterns,
                "specialization": self._analyze_head_specialization(attention_weight)
            }
            layer_analysis["head_patterns"].append(head_analysis)
        
        # Layer-level statistics
        layer_analysis["layer_statistics"] = {
            "avg_entropy": np.mean(entropies),
            "entropy_variance": np.var(entropies),
            "avg_locality": np.mean(localities),
            "locality_variance": np.var(localities),
            "avg_spread": np.mean(spreads),
            "head_diversity": np.std(entropies)  # How diverse are the heads
        }
        
        return layer_analysis
    
    def _analyze_head_specialization(self, attention_weight: AttentionWeights) -> Dict[str, Any]:
        """Analyze what this attention head specializes in"""
        patterns = attention_weight.get_attention_patterns()
        
        specialization = {
            "type": "unknown",
            "confidence": 0.0,
            "characteristics": []
        }
        
        # Classify head specialization based on patterns
        if patterns["diagonal_dominance"] > 0.7:
            specialization["type"] = "positional"
            specialization["confidence"] = patterns["diagonal_dominance"]
            specialization["characteristics"].append("Strong positional bias")
        
        elif patterns["locality_score"] > 0.8:
            specialization["type"] = "local"
            specialization["confidence"] = patterns["locality_score"]
            specialization["characteristics"].append("Local attention patterns")
        
        elif patterns["entropy"] > 3.0:
            specialization["type"] = "global"
            specialization["confidence"] = min(patterns["entropy"] / 4.0, 1.0)
            specialization["characteristics"].append("Broad attention distribution")
        
        elif patterns["attention_spread"] < 0.1:
            specialization["type"] = "focused"
            specialization["confidence"] = 1.0 - patterns["attention_spread"]
            specialization["characteristics"].append("Highly focused attention")
        
        return specialization
    
    def _generate_summary_statistics(self, attention_weights: List[AttentionWeights]) -> Dict[str, Any]:
        """Generate summary statistics across all attention weights"""
        all_patterns = [aw.get_attention_patterns() for aw in attention_weights]
        
        summary = {
            "total_heads": len(attention_weights),
            "avg_entropy": np.mean([p["entropy"] for p in all_patterns]),
            "avg_locality": np.mean([p["locality_score"] for p in all_patterns]),
            "avg_spread": np.mean([p["attention_spread"] for p in all_patterns]),
            "entropy_distribution": {
                "min": min([p["entropy"] for p in all_patterns]),
                "max": max([p["entropy"] for p in all_patterns]),
                "std": np.std([p["entropy"] for p in all_patterns])
            }
        }
        
        return summary
    
    def _classify_attention_patterns(self, attention_weights: List[AttentionWeights]) -> Dict[str, Any]:
        """Classify the types of attention patterns present"""
        pattern_types = {
            "positional": 0,
            "local": 0,
            "global": 0,
            "focused": 0,
            "dispersed": 0
        }
        
        for attention_weight in attention_weights:
            patterns = attention_weight.get_attention_patterns()
            
            if patterns["diagonal_dominance"] > 0.6:
                pattern_types["positional"] += 1
            if patterns["locality_score"] > 0.7:
                pattern_types["local"] += 1
            if patterns["entropy"] > 2.5:
                pattern_types["global"] += 1
            if patterns["attention_spread"] < 0.2:
                pattern_types["focused"] += 1
            if patterns["attention_spread"] > 0.8:
                pattern_types["dispersed"] += 1
        
        # Convert to percentages
        total = len(attention_weights)
        return {k: (v / total) * 100 for k, v in pattern_types.items()}
    
    def _detect_attention_anomalies(self, attention_weights: List[AttentionWeights]) -> List[Dict[str, Any]]:
        """Detect anomalous attention patterns"""
        anomalies = []
        
        entropies = [aw.get_attention_patterns()["entropy"] for aw in attention_weights]
        entropy_mean = np.mean(entropies)
        entropy_std = np.std(entropies)
        
        for i, attention_weight in enumerate(attention_weights):
            patterns = attention_weight.get_attention_patterns()
            
            # Check for entropy anomalies
            z_score = abs(patterns["entropy"] - entropy_mean) / (entropy_std + 1e-8)
            if z_score > 2.5:
                anomalies.append({
                    "type": "entropy_anomaly",
                    "layer": attention_weight.layer_idx,
                    "head": attention_weight.head_idx,
                    "severity": min(z_score / 2.5, 2.0),
                    "description": f"Unusual entropy: {patterns['entropy']:.3f} (z-score: {z_score:.2f})"
                })
            
            # Check for attention collapse
            if patterns["max_attention"] > 0.95:
                anomalies.append({
                    "type": "attention_collapse",
                    "layer": attention_weight.layer_idx,
                    "head": attention_weight.head_idx,
                    "severity": 2.0,
                    "description": f"Attention collapse detected: max weight {patterns['max_attention']:.3f}"
                })
        
        return anomalies


class TransformerArchitectureManager:
    """Manager for transformer architecture versioning and compatibility"""
    
    def __init__(self):
        self.architecture_registry: Dict[str, TransformerMetadata] = {}
        self.compatibility_matrix: Dict[Tuple[str, str], bool] = {}
    
    async def register_architecture(
        self,
        model_id: str,
        architecture: TransformerArchitecture,
        model_config: Dict[str, Any]
    ) -> TransformerMetadata:
        """
        Register a transformer architecture
        
        Args:
            model_id: Model identifier
            architecture: Transformer architecture type
            model_config: Model configuration dictionary
            
        Returns:
            TransformerMetadata object
        """
        metadata = TransformerMetadata(
            architecture=architecture,
            num_layers=model_config.get("num_layers", 12),
            num_heads=model_config.get("num_heads", 12),
            hidden_size=model_config.get("hidden_size", 768),
            intermediate_size=model_config.get("intermediate_size", 3072),
            vocab_size=model_config.get("vocab_size"),
            max_position_embeddings=model_config.get("max_position_embeddings"),
            attention_types=self._infer_attention_types(model_config),
            total_parameters=model_config.get("total_parameters", 0),
            activation_function=model_config.get("activation_function", "gelu"),
            dropout_rate=model_config.get("dropout_rate", 0.1),
            layer_norm_eps=model_config.get("layer_norm_eps", 1e-12)
        )
        
        self.architecture_registry[model_id] = metadata
        return metadata
    
    def _infer_attention_types(self, model_config: Dict[str, Any]) -> List[AttentionType]:
        """Infer attention types from model configuration"""
        attention_types = []
        
        # Based on common transformer configurations
        if model_config.get("num_heads", 1) > 1:
            attention_types.append(AttentionType.MULTI_HEAD)
        
        if model_config.get("is_decoder", False):
            attention_types.append(AttentionType.CAUSAL_ATTENTION)
        
        if model_config.get("cross_attention", False):
            attention_types.append(AttentionType.CROSS_ATTENTION)
        else:
            attention_types.append(AttentionType.SELF_ATTENTION)
        
        return attention_types
    
    async def check_compatibility(
        self,
        source_model_id: str,
        target_model_id: str
    ) -> Dict[str, Any]:
        """
        Check compatibility between two transformer architectures
        
        Args:
            source_model_id: Source model identifier
            target_model_id: Target model identifier
            
        Returns:
            Compatibility analysis
        """
        if source_model_id not in self.architecture_registry:
            raise ValueError(f"Source model {source_model_id} not registered")
        
        if target_model_id not in self.architecture_registry:
            raise ValueError(f"Target model {target_model_id} not registered")
        
        source_metadata = self.architecture_registry[source_model_id]
        target_metadata = self.architecture_registry[target_model_id]
        
        compatibility = {
            "compatible": True,
            "compatibility_score": 1.0,
            "warnings": [],
            "blocking_issues": [],
            "migration_required": False
        }
        
        # Check architecture compatibility
        if source_metadata.architecture != target_metadata.architecture:
            compatibility["warnings"].append(
                f"Different architectures: {source_metadata.architecture.value} -> {target_metadata.architecture.value}"
            )
            compatibility["compatibility_score"] *= 0.8
            compatibility["migration_required"] = True
        
        # Check dimension compatibility
        if source_metadata.hidden_size != target_metadata.hidden_size:
            compatibility["blocking_issues"].append(
                f"Hidden size mismatch: {source_metadata.hidden_size} -> {target_metadata.hidden_size}"
            )
            compatibility["compatible"] = False
        
        if source_metadata.num_heads != target_metadata.num_heads:
            compatibility["warnings"].append(
                f"Number of heads changed: {source_metadata.num_heads} -> {target_metadata.num_heads}"
            )
            compatibility["compatibility_score"] *= 0.9
        
        if source_metadata.num_layers != target_metadata.num_layers:
            compatibility["warnings"].append(
                f"Number of layers changed: {source_metadata.num_layers} -> {target_metadata.num_layers}"
            )
            compatibility["compatibility_score"] *= 0.9
        
        # Check attention type compatibility
        source_attention_set = set(source_metadata.attention_types)
        target_attention_set = set(target_metadata.attention_types)
        
        if source_attention_set != target_attention_set:
            missing_types = source_attention_set - target_attention_set
            added_types = target_attention_set - source_attention_set
            
            if missing_types:
                compatibility["warnings"].append(f"Removed attention types: {missing_types}")
            if added_types:
                compatibility["warnings"].append(f"Added attention types: {added_types}")
            
            compatibility["compatibility_score"] *= 0.85
        
        return compatibility
    
    async def generate_migration_plan(
        self,
        source_model_id: str,
        target_model_id: str
    ) -> Dict[str, Any]:
        """
        Generate migration plan between architectures
        
        Args:
            source_model_id: Source model identifier
            target_model_id: Target model identifier
            
        Returns:
            Migration plan
        """
        compatibility = await self.check_compatibility(source_model_id, target_model_id)
        
        if not compatibility["migration_required"]:
            return {"migration_needed": False, "plan": []}
        
        source_metadata = self.architecture_registry[source_model_id]
        target_metadata = self.architecture_registry[target_model_id]
        
        plan = {
            "migration_needed": True,
            "complexity": "medium",
            "estimated_time_minutes": 30,
            "steps": [],
            "risks": [],
            "rollback_plan": []
        }
        
        # Add migration steps based on differences
        if source_metadata.architecture != target_metadata.architecture:
            plan["steps"].append({
                "step": "architecture_conversion",
                "description": f"Convert from {source_metadata.architecture.value} to {target_metadata.architecture.value}",
                "estimated_time": 15,
                "complexity": "high"
            })
            plan["complexity"] = "high"
            plan["estimated_time_minutes"] += 15
        
        if source_metadata.num_layers != target_metadata.num_layers:
            if source_metadata.num_layers > target_metadata.num_layers:
                plan["steps"].append({
                    "step": "layer_pruning",
                    "description": f"Prune layers from {source_metadata.num_layers} to {target_metadata.num_layers}",
                    "estimated_time": 10
                })
            else:
                plan["steps"].append({
                    "step": "layer_expansion",
                    "description": f"Expand layers from {source_metadata.num_layers} to {target_metadata.num_layers}",
                    "estimated_time": 15
                })
        
        # Add risks
        if not compatibility["compatible"]:
            plan["risks"].append("Model may lose accuracy due to incompatible architectures")
        
        if len(compatibility["warnings"]) > 2:
            plan["risks"].append("Multiple architectural changes may affect model performance")
        
        return plan


class TransformerMigrationManager:
    """Manager for migrating transformer models between versions and architectures"""
    
    def __init__(self, preservation_manager: 'PreservationManager'):
        self.preservation_manager = preservation_manager
        self.migration_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def migrate_transformer_model(
        self,
        source_model_type: str,
        source_version: str,
        target_architecture: TransformerArchitecture,
        migration_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Migrate a transformer model to a new architecture
        
        Args:
            source_model_type: Source model type
            source_version: Source model version
            target_architecture: Target transformer architecture
            migration_config: Optional migration configuration
            
        Returns:
            New model ID after migration
        """
        config = migration_config or {}
        
        # Load source model
        source_data, source_metadata = await self.preservation_manager.load_model(
            model_type=source_model_type,
            version=source_version
        )
        
        # Deserialize model
        source_model = pickle.loads(source_data)
        
        # Perform migration based on target architecture
        migrated_model = await self._perform_migration(
            source_model,
            target_architecture,
            config
        )
        
        # Serialize migrated model
        migrated_data = pickle.dumps(migrated_model)
        
        # Generate new version
        target_model_type = f"{source_model_type}_{target_architecture.value}"
        
        # Save migrated model
        new_model_id = await self.preservation_manager.save_model(
            model_data=migrated_data,
            model_type=target_model_type,
            mode=source_metadata.get("mode", "analysis"),
            tags=["migrated", f"from_{source_model_type}"],
            metadata={
                "migration": {
                    "source_model_type": source_model_type,
                    "source_version": source_version,
                    "target_architecture": target_architecture.value,
                    "migration_timestamp": datetime.now().isoformat(),
                    "migration_config": config
                }
            },
            priority=PreservationPriority.HIGH
        )
        
        # Record migration
        migration_record = {
            "migration_id": new_model_id,
            "source_model": f"{source_model_type}:{source_version}",
            "target_architecture": target_architecture.value,
            "timestamp": datetime.now(),
            "success": True,
            "config": config
        }
        
        if source_model_type not in self.migration_history:
            self.migration_history[source_model_type] = []
        self.migration_history[source_model_type].append(migration_record)
        
        return new_model_id
    
    async def _perform_migration(
        self,
        source_model: nn.Module,
        target_architecture: TransformerArchitecture,
        config: Dict[str, Any]
    ) -> nn.Module:
        """
        Perform the actual model migration
        
        Args:
            source_model: Source transformer model
            target_architecture: Target architecture
            config: Migration configuration
            
        Returns:
            Migrated model
        """
        # This is a simplified implementation
        # In practice, this would involve complex model transformations
        
        if target_architecture == TransformerArchitecture.ITRANSFORMER:
            return await self._migrate_to_itransformer(source_model, config)
        elif target_architecture == TransformerArchitecture.PATCHTST:
            return await self._migrate_to_patchtst(source_model, config)
        else:
            # For now, return the source model
            # In practice, implement specific migration logic
            return source_model
    
    async def _migrate_to_itransformer(
        self,
        source_model: nn.Module,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Migrate to iTransformer architecture"""
        # Placeholder implementation
        # In practice, this would involve architectural transformations
        return source_model
    
    async def _migrate_to_patchtst(
        self,
        source_model: nn.Module,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Migrate to PatchTST architecture"""
        # Placeholder implementation
        return source_model
    
    async def get_migration_history(
        self,
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Get migration history for a model type"""
        return self.migration_history.get(model_type, [])


class TransformerPreservationManager:
    """Main manager for transformer model preservation"""
    
    def __init__(self, preservation_manager: 'PreservationManager'):
        self.preservation_manager = preservation_manager
        self.attention_manager = AttentionPreservationManager()
        self.pattern_analyzer = AttentionPatternAnalyzer()
        self.architecture_manager = TransformerArchitectureManager()
        self.migration_manager = TransformerMigrationManager(preservation_manager)
        
        # Sharding configuration for large models
        self.shard_size_mb = 500  # 500MB per shard
        self.max_model_size_gb = 2  # 2GB threshold for sharding
    
    async def save_transformer_model(
        self,
        model: nn.Module,
        model_type: str,
        version: Optional[str] = None,
        mode: str = "analysis",
        preserve_attention: bool = True,
        architecture_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Save transformer model with specialized preservation
        
        Args:
            model: Transformer model to save
            model_type: Type of transformer model
            version: Model version
            mode: Operational mode
            preserve_attention: Whether to preserve attention patterns
            architecture_config: Architecture configuration
            **kwargs: Additional arguments for preservation
            
        Returns:
            Model ID
        """
        # Serialize model
        model_data = pickle.dumps(model)
        model_size_mb = len(model_data) / (1024 * 1024)
        
        # Check if sharding is needed
        needs_sharding = model_size_mb > (self.max_model_size_gb * 1024)
        
        if needs_sharding:
            model_data, shard_info = await self._shard_model(model_data)
            kwargs.setdefault("metadata", {})["sharding"] = shard_info
        
        # Register architecture if provided
        if architecture_config:
            transformer_architecture = TransformerArchitecture(
                architecture_config.get("architecture", "vanilla_transformer")
            )
            await self.architecture_manager.register_architecture(
                model_id=f"{model_type}-{version or 'latest'}",
                architecture=transformer_architecture,
                model_config=architecture_config
            )
        
        # Extract and preserve attention patterns if requested
        if preserve_attention:
            try:
                attention_weights = await self.attention_manager.extract_attention_weights(
                    model=model,
                    model_id=f"{model_type}-{version or 'latest'}"
                )
                
                # Analyze patterns
                pattern_analysis = await self.pattern_analyzer.analyze_attention_patterns(
                    attention_weights
                )
                
                kwargs.setdefault("metadata", {})["attention_analysis"] = pattern_analysis
                kwargs.setdefault("tags", []).append("attention_preserved")
                
            except Exception as e:
                # Log warning but don't fail the save operation
                print(f"Warning: Failed to preserve attention patterns: {e}")
        
        # Add transformer-specific metadata
        transformer_metadata = {
            "model_size_mb": model_size_mb,
            "needs_sharding": needs_sharding,
            "transformer_type": model_type,
            "preservation_timestamp": datetime.now().isoformat()
        }
        
        kwargs.setdefault("metadata", {}).update(transformer_metadata)
        kwargs.setdefault("tags", []).extend(["transformer", model_type])
        
        # Save using base preservation manager
        return await self.preservation_manager.save_model(
            model_data=model_data,
            model_type=model_type,
            version=version,
            mode=mode,
            **kwargs
        )
    
    async def load_transformer_model(
        self,
        model_type: str,
        version: Optional[str] = None,
        mode: str = "analysis",
        load_attention_patterns: bool = True,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load transformer model with specialized handling
        
        Args:
            model_type: Type of transformer model
            version: Model version
            mode: Operational mode
            load_attention_patterns: Whether to load attention patterns
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load using base preservation manager
        model_data, metadata = await self.preservation_manager.load_model(
            model_type=model_type,
            version=version,
            mode=mode,
            **kwargs
        )
        
        # Handle sharding if present
        if metadata.get("sharding"):
            model_data = await self._reconstruct_sharded_model(model_data, metadata["sharding"])
        
        # Deserialize model
        model = pickle.loads(model_data)
        
        # Load attention patterns if available and requested
        if load_attention_patterns and "attention_analysis" in metadata:
            # Attention patterns are already in metadata
            pass
        
        return model, metadata
    
    async def _shard_model(self, model_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Shard large model into smaller chunks
        
        Args:
            model_data: Serialized model data
            
        Returns:
            Tuple of (shard_metadata, shard_info)
        """
        shard_size_bytes = self.shard_size_mb * 1024 * 1024
        total_size = len(model_data)
        num_shards = (total_size + shard_size_bytes - 1) // shard_size_bytes
        
        shards = []
        shard_checksums = []
        
        for i in range(num_shards):
            start_idx = i * shard_size_bytes
            end_idx = min(start_idx + shard_size_bytes, total_size)
            shard_data = model_data[start_idx:end_idx]
            
            # Compress shard
            compressed_shard = zlib.compress(shard_data, level=6)
            shards.append(compressed_shard)
            
            # Calculate checksum
            shard_checksum = calculate_checksum(shard_data)
            shard_checksums.append(shard_checksum)
        
        shard_info = {
            "num_shards": num_shards,
            "shard_size_mb": self.shard_size_mb,
            "total_size_bytes": total_size,
            "shard_checksums": shard_checksums,
            "compression_ratio": len(b''.join(shards)) / total_size
        }
        
        # Serialize shard metadata instead of original data
        shard_metadata = {
            "shards": shards,
            "shard_info": shard_info
        }
        
        return pickle.dumps(shard_metadata), shard_info
    
    async def _reconstruct_sharded_model(
        self,
        shard_metadata_data: bytes,
        shard_info: Dict[str, Any]
    ) -> bytes:
        """
        Reconstruct model from shards
        
        Args:
            shard_metadata_data: Serialized shard metadata
            shard_info: Shard information
            
        Returns:
            Reconstructed model data
        """
        shard_metadata = pickle.loads(shard_metadata_data)
        shards = shard_metadata["shards"]
        
        # Reconstruct original data
        reconstructed_parts = []
        for i, compressed_shard in enumerate(shards):
            # Decompress shard
            shard_data = zlib.decompress(compressed_shard)
            
            # Verify checksum
            expected_checksum = shard_info["shard_checksums"][i]
            actual_checksum = calculate_checksum(shard_data)
            
            if expected_checksum != actual_checksum:
                raise PreservationError(f"Shard {i} checksum mismatch")
            
            reconstructed_parts.append(shard_data)
        
        return b''.join(reconstructed_parts)
    
    async def compare_transformer_models(
        self,
        model1_type: str,
        model1_version: str,
        model2_type: str,
        model2_version: str,
        comparison_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare two transformer models including attention patterns
        
        Args:
            model1_type: First model type
            model1_version: First model version
            model2_type: Second model type
            model2_version: Second model version
            comparison_config: Optional comparison configuration
            
        Returns:
            Detailed comparison results
        """
        config = comparison_config or {}
        
        # Load both models
        _, metadata1 = await self.preservation_manager.load_model(
            model_type=model1_type,
            version=model1_version
        )
        
        _, metadata2 = await self.preservation_manager.load_model(
            model_type=model2_type,
            version=model2_version
        )
        
        comparison = {
            "models": {
                "model1": f"{model1_type}:{model1_version}",
                "model2": f"{model2_type}:{model2_version}"
            },
            "architecture_comparison": {},
            "attention_comparison": {},
            "performance_comparison": {},
            "compatibility": {}
        }
        
        # Compare architectures if available
        model1_id = f"{model1_type}-{model1_version}"
        model2_id = f"{model2_type}-{model2_version}"
        
        if (model1_id in self.architecture_manager.architecture_registry and
            model2_id in self.architecture_manager.architecture_registry):
            
            compatibility = await self.architecture_manager.check_compatibility(
                model1_id, model2_id
            )
            comparison["compatibility"] = compatibility
        
        # Compare attention patterns if available
        if ("attention_analysis" in metadata1 and 
            "attention_analysis" in metadata2):
            
            attention_comparison = self._compare_attention_analyses(
                metadata1["attention_analysis"],
                metadata2["attention_analysis"]
            )
            comparison["attention_comparison"] = attention_comparison
        
        # Compare basic metrics
        comparison["size_comparison"] = {
            "model1_size_mb": metadata1.get("model_size_mb", 0),
            "model2_size_mb": metadata2.get("model_size_mb", 0),
            "size_ratio": metadata2.get("model_size_mb", 1) / max(metadata1.get("model_size_mb", 1), 1)
        }
        
        return comparison
    
    def _compare_attention_analyses(
        self,
        analysis1: Dict[str, Any],
        analysis2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two attention pattern analyses"""
        comparison = {
            "summary_differences": {},
            "pattern_evolution": {},
            "significant_changes": []
        }
        
        # Compare summary statistics
        summary1 = analysis1.get("summary", {})
        summary2 = analysis2.get("summary", {})
        
        for metric in ["avg_entropy", "avg_locality", "avg_spread"]:
            if metric in summary1 and metric in summary2:
                diff = summary2[metric] - summary1[metric]
                comparison["summary_differences"][metric] = {
                    "change": diff,
                    "relative_change": diff / max(abs(summary1[metric]), 1e-8),
                    "model1_value": summary1[metric],
                    "model2_value": summary2[metric]
                }
                
                # Flag significant changes
                if abs(diff) > 0.5:
                    comparison["significant_changes"].append(
                        f"Large {metric} change: {diff:.3f}"
                    )
        
        return comparison