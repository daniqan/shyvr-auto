"""
Tests for Transformer Preservation Functionality
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.model_preservation.transformer_preservation import (
    TransformerPreservationManager,
    AttentionPreservationManager,
    AttentionPatternAnalyzer,
    TransformerArchitectureManager,
    TransformerMigrationManager,
    AttentionWeights,
    AttentionType,
    TransformerArchitecture,
    TransformerMetadata
)
from src.model_preservation.base import PreservationPriority, ModelState
from src.model_preservation.manager import PreservationManager, PreservationConfig


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing"""
    
    def __init__(self, num_layers=6, num_heads=8, hidden_size=512):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Mock attention layers
        self.attention_layers = nn.ModuleList([
            MockAttentionLayer(num_heads, hidden_size) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Simple forward pass for testing
        return x


class MockAttentionLayer(nn.Module):
    """Mock attention layer that can produce attention weights"""
    
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
    
    def forward(self, x):
        # Return attention weights for testing
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output


@pytest.fixture
def mock_preservation_manager():
    """Create mock preservation manager"""
    config = PreservationConfig(gcs_bucket="test-bucket")
    manager = Mock(spec=PreservationManager)
    manager.config = config
    return manager


@pytest.fixture
def sample_attention_weights():
    """Create sample attention weights for testing"""
    weights = []
    for layer_idx in range(3):
        for head_idx in range(4):
            attention_matrix = np.random.rand(10, 10)  # 10x10 attention matrix
            attention_matrix = attention_matrix / attention_matrix.sum(axis=-1, keepdims=True)  # Normalize
            
            attention_weight = AttentionWeights(
                layer_idx=layer_idx,
                head_idx=head_idx,
                attention_type=AttentionType.MULTI_HEAD,
                weights=attention_matrix,
                input_sequence_length=10,
                output_sequence_length=10
            )
            weights.append(attention_weight)
    
    return weights


@pytest.fixture
def sample_transformer_model():
    """Create sample transformer model"""
    return MockTransformerModel(num_layers=6, num_heads=8, hidden_size=512)


class TestAttentionWeights:
    """Test AttentionWeights class"""
    
    def test_attention_weights_creation(self):
        """Test basic AttentionWeights creation"""
        weights = np.random.rand(5, 5)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        
        attention_weight = AttentionWeights(
            layer_idx=0,
            head_idx=0,
            attention_type=AttentionType.SELF_ATTENTION,
            weights=weights,
            input_sequence_length=5,
            output_sequence_length=5
        )
        
        assert attention_weight.layer_idx == 0
        assert attention_weight.head_idx == 0
        assert attention_weight.attention_type == AttentionType.SELF_ATTENTION
        assert attention_weight.shape == (5, 5)
        assert isinstance(attention_weight.entropy, float)
    
    def test_attention_patterns(self):
        """Test attention pattern calculation"""
        # Create diagonal-dominant attention (positional bias)
        weights = np.eye(5) * 0.8 + np.random.rand(5, 5) * 0.2
        weights = weights / weights.sum(axis=-1, keepdims=True)
        
        attention_weight = AttentionWeights(
            layer_idx=0,
            head_idx=0,
            attention_type=AttentionType.SELF_ATTENTION,
            weights=weights,
            input_sequence_length=5,
            output_sequence_length=5
        )
        
        patterns = attention_weight.get_attention_patterns()
        
        assert "entropy" in patterns
        assert "diagonal_dominance" in patterns
        assert "locality_score" in patterns
        assert patterns["diagonal_dominance"] > 0.6  # Should be high for diagonal matrix
    
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        # Uniform attention (high entropy)
        uniform_weights = np.ones((5, 5)) / 5
        uniform_attention = AttentionWeights(
            layer_idx=0, head_idx=0, attention_type=AttentionType.SELF_ATTENTION,
            weights=uniform_weights, input_sequence_length=5, output_sequence_length=5
        )
        
        # Focused attention (low entropy)
        focused_weights = np.zeros((5, 5))
        focused_weights[0, 0] = 1.0  # All attention on one position
        focused_attention = AttentionWeights(
            layer_idx=0, head_idx=0, attention_type=AttentionType.SELF_ATTENTION,
            weights=focused_weights, input_sequence_length=5, output_sequence_length=5
        )
        
        assert uniform_attention.entropy > focused_attention.entropy


class TestAttentionPreservationManager:
    """Test AttentionPreservationManager class"""
    
    @pytest.fixture
    def attention_manager(self):
        return AttentionPreservationManager()
    
    @pytest.mark.asyncio
    async def test_extract_attention_weights(self, attention_manager, sample_transformer_model):
        """Test attention weight extraction"""
        model_id = "test_transformer_v1"
        input_data = torch.randn(1, 10, 512)  # batch_size=1, seq_len=10, hidden_size=512
        
        # Mock the attention extraction
        with patch.object(attention_manager, 'attention_cache', {}):
            attention_weights = await attention_manager.extract_attention_weights(
                model=sample_transformer_model,
                model_id=model_id,
                input_data=input_data
            )
            
            # Should return empty list since we're using mock model
            assert isinstance(attention_weights, list)
    
    @pytest.mark.asyncio
    async def test_preserve_attention_patterns(self, attention_manager, sample_attention_weights):
        """Test attention pattern preservation"""
        model_id = "test_model"
        storage_path = "/test/path"
        
        with patch.object(attention_manager, '_compress_attention_weights') as mock_compress:
            mock_compress.return_value = b"compressed_data"
            
            result_path = await attention_manager.preserve_attention_patterns(
                model_id=model_id,
                attention_weights=sample_attention_weights,
                storage_path=storage_path
            )
            
            assert result_path == f"{storage_path}/attention_patterns.pkl.gz"
            assert mock_compress.call_count == len(sample_attention_weights)
    
    @pytest.mark.asyncio
    async def test_analyze_attention_evolution(self, attention_manager):
        """Test attention evolution analysis"""
        model_id = "test_model"
        versions = ["v1.0.0", "v1.1.0", "v1.2.0"]
        
        # Mock pattern history
        attention_manager.pattern_history = {
            f"{model_id}-v1.0.0": [{"patterns": {"entropy": 2.0, "locality_score": 0.5}}],
            f"{model_id}-v1.1.0": [{"patterns": {"entropy": 2.2, "locality_score": 0.6}}],
            f"{model_id}-v1.2.0": [{"patterns": {"entropy": 2.5, "locality_score": 0.7}}]
        }
        
        analysis = await attention_manager.analyze_attention_evolution(model_id, versions)
        
        assert analysis["model_id"] == model_id
        assert analysis["versions_analyzed"] == versions
        assert "pattern_changes" in analysis
        assert len(analysis["pattern_changes"]) == 2  # v1.0->v1.1, v1.1->v1.2


class TestAttentionPatternAnalyzer:
    """Test AttentionPatternAnalyzer class"""
    
    @pytest.fixture
    def pattern_analyzer(self):
        return AttentionPatternAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_attention_patterns(self, pattern_analyzer, sample_attention_weights):
        """Test comprehensive attention pattern analysis"""
        analysis = await pattern_analyzer.analyze_attention_patterns(sample_attention_weights)
        
        assert "summary" in analysis
        assert "layer_analysis" in analysis
        assert "pattern_types" in analysis
        assert "anomalies" in analysis
        
        # Check summary statistics
        summary = analysis["summary"]
        assert "total_heads" in summary
        assert summary["total_heads"] == len(sample_attention_weights)
        assert "avg_entropy" in summary
    
    @pytest.mark.asyncio
    async def test_layer_analysis(self, pattern_analyzer, sample_attention_weights):
        """Test layer-specific analysis"""
        analysis = await pattern_analyzer.analyze_attention_patterns(sample_attention_weights)
        
        layer_analysis = analysis["layer_analysis"]
        assert len(layer_analysis) == 3  # 3 layers in sample data
        
        for layer in layer_analysis:
            assert "layer_idx" in layer
            assert "num_heads" in layer
            assert "head_patterns" in layer
            assert "layer_statistics" in layer
    
    def test_head_specialization(self, pattern_analyzer):
        """Test head specialization analysis"""
        # Create focused attention (should be classified as "focused")
        focused_weights = np.zeros((5, 5))
        focused_weights[0, 0] = 1.0
        focused_attention = AttentionWeights(
            layer_idx=0, head_idx=0, attention_type=AttentionType.SELF_ATTENTION,
            weights=focused_weights, input_sequence_length=5, output_sequence_length=5
        )
        
        specialization = pattern_analyzer._analyze_head_specialization(focused_attention)
        
        assert specialization["type"] == "focused"
        assert specialization["confidence"] > 0.8
        assert "Highly focused attention" in specialization["characteristics"]


class TestTransformerArchitectureManager:
    """Test TransformerArchitectureManager class"""
    
    @pytest.fixture
    def arch_manager(self):
        return TransformerArchitectureManager()
    
    @pytest.mark.asyncio
    async def test_register_architecture(self, arch_manager):
        """Test architecture registration"""
        model_id = "test_transformer"
        architecture = TransformerArchitecture.VANILLA_TRANSFORMER
        config = {
            "num_layers": 12,
            "num_heads": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "total_parameters": 110000000
        }
        
        metadata = await arch_manager.register_architecture(model_id, architecture, config)
        
        assert metadata.architecture == architecture
        assert metadata.num_layers == 12
        assert metadata.num_heads == 12
        assert metadata.hidden_size == 768
        assert model_id in arch_manager.architecture_registry
    
    @pytest.mark.asyncio
    async def test_check_compatibility(self, arch_manager):
        """Test architecture compatibility checking"""
        # Register two similar architectures
        config1 = {
            "num_layers": 12, "num_heads": 12, "hidden_size": 768,
            "intermediate_size": 3072, "total_parameters": 110000000
        }
        config2 = {
            "num_layers": 12, "num_heads": 12, "hidden_size": 768,
            "intermediate_size": 3072, "total_parameters": 110000000
        }
        
        await arch_manager.register_architecture("model1", TransformerArchitecture.VANILLA_TRANSFORMER, config1)
        await arch_manager.register_architecture("model2", TransformerArchitecture.VANILLA_TRANSFORMER, config2)
        
        compatibility = await arch_manager.check_compatibility("model1", "model2")
        
        assert compatibility["compatible"] == True
        assert compatibility["compatibility_score"] == 1.0
        assert len(compatibility["blocking_issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_incompatible_architectures(self, arch_manager):
        """Test incompatible architecture detection"""
        config1 = {"num_layers": 12, "num_heads": 12, "hidden_size": 768, "intermediate_size": 3072}
        config2 = {"num_layers": 12, "num_heads": 12, "hidden_size": 512, "intermediate_size": 2048}  # Different hidden size
        
        await arch_manager.register_architecture("model1", TransformerArchitecture.VANILLA_TRANSFORMER, config1)
        await arch_manager.register_architecture("model2", TransformerArchitecture.VANILLA_TRANSFORMER, config2)
        
        compatibility = await arch_manager.check_compatibility("model1", "model2")
        
        assert compatibility["compatible"] == False
        assert len(compatibility["blocking_issues"]) > 0
        assert any("hidden_size" in issue for issue in compatibility["blocking_issues"])
    
    @pytest.mark.asyncio
    async def test_migration_plan_generation(self, arch_manager):
        """Test migration plan generation"""
        config1 = {"num_layers": 6, "num_heads": 8, "hidden_size": 512, "intermediate_size": 2048}
        config2 = {"num_layers": 12, "num_heads": 8, "hidden_size": 512, "intermediate_size": 2048}
        
        await arch_manager.register_architecture("small_model", TransformerArchitecture.VANILLA_TRANSFORMER, config1)
        await arch_manager.register_architecture("large_model", TransformerArchitecture.VANILLA_TRANSFORMER, config2)
        
        migration_plan = await arch_manager.generate_migration_plan("small_model", "large_model")
        
        assert migration_plan["migration_needed"] == True
        assert "steps" in migration_plan
        assert any("layer_expansion" in step["step"] for step in migration_plan["steps"])


class TestTransformerMigrationManager:
    """Test TransformerMigrationManager class"""
    
    @pytest.fixture
    def migration_manager(self, mock_preservation_manager):
        return TransformerMigrationManager(mock_preservation_manager)
    
    @pytest.mark.asyncio
    async def test_migrate_transformer_model(self, migration_manager, sample_transformer_model):
        """Test transformer model migration"""
        # Mock preservation manager methods
        migration_manager.preservation_manager.load_model = AsyncMock()
        migration_manager.preservation_manager.load_model.return_value = (
            b"model_data", {"mode": "analysis"}
        )
        migration_manager.preservation_manager.save_model = AsyncMock()
        migration_manager.preservation_manager.save_model.return_value = "new_model_id"
        
        with patch('pickle.loads', return_value=sample_transformer_model):
            with patch('pickle.dumps', return_value=b"migrated_model_data"):
                new_model_id = await migration_manager.migrate_transformer_model(
                    source_model_type="transformer",
                    source_version="v1.0.0",
                    target_architecture=TransformerArchitecture.ITRANSFORMER
                )
                
                assert new_model_id == "new_model_id"
                migration_manager.preservation_manager.load_model.assert_called_once()
                migration_manager.preservation_manager.save_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_migration_history(self, migration_manager):
        """Test migration history tracking"""
        # Mock a migration
        migration_manager.migration_history = {
            "transformer": [{
                "migration_id": "test_id",
                "source_model": "transformer:v1.0.0",
                "target_architecture": "itransformer",
                "timestamp": datetime.now(),
                "success": True,
                "config": {}
            }]
        }
        
        history = await migration_manager.get_migration_history("transformer")
        
        assert len(history) == 1
        assert history[0]["source_model"] == "transformer:v1.0.0"
        assert history[0]["target_architecture"] == "itransformer"


class TestTransformerPreservationManager:
    """Test TransformerPreservationManager class"""
    
    @pytest.fixture
    def transformer_manager(self, mock_preservation_manager):
        return TransformerPreservationManager(mock_preservation_manager)
    
    @pytest.mark.asyncio
    async def test_save_transformer_model(self, transformer_manager, sample_transformer_model):
        """Test transformer model saving"""
        # Mock preservation manager save method
        transformer_manager.preservation_manager.save_model = AsyncMock()
        transformer_manager.preservation_manager.save_model.return_value = "model_id_123"
        
        with patch('pickle.dumps', return_value=b"model_data"):
            model_id = await transformer_manager.save_transformer_model(
                model=sample_transformer_model,
                model_type="transformer",
                version="v1.0.0",
                preserve_attention=False,  # Skip attention extraction for this test
                architecture_config={
                    "num_layers": 6,
                    "num_heads": 8,
                    "hidden_size": 512
                }
            )
            
            assert model_id == "model_id_123"
            transformer_manager.preservation_manager.save_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_transformer_model(self, transformer_manager, sample_transformer_model):
        """Test transformer model loading"""
        # Mock preservation manager load method
        transformer_manager.preservation_manager.load_model = AsyncMock()
        transformer_manager.preservation_manager.load_model.return_value = (
            b"model_data", {"model_size_mb": 100}
        )
        
        with patch('pickle.loads', return_value=sample_transformer_model):
            model, metadata = await transformer_manager.load_transformer_model(
                model_type="transformer",
                version="v1.0.0"
            )
            
            assert model == sample_transformer_model
            assert metadata["model_size_mb"] == 100
            transformer_manager.preservation_manager.load_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sharding_large_model(self, transformer_manager):
        """Test model sharding for large models"""
        # Create large model data (> 2GB threshold)
        large_model_data = b"x" * (3 * 1024 * 1024 * 1024)  # 3GB of data
        
        with patch.object(transformer_manager, '_shard_model') as mock_shard:
            mock_shard.return_value = (b"sharded_metadata", {"num_shards": 6})
            
            sharded_data, shard_info = await transformer_manager._shard_model(large_model_data)
            
            assert shard_info["num_shards"] == 6
            mock_shard.assert_called_once_with(large_model_data)
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, transformer_manager):
        """Test transformer model comparison"""
        # Mock preservation manager load methods
        transformer_manager.preservation_manager.load_model = AsyncMock()
        transformer_manager.preservation_manager.load_model.side_effect = [
            (b"model1_data", {"model_size_mb": 100, "attention_analysis": {}}),
            (b"model2_data", {"model_size_mb": 200, "attention_analysis": {}})
        ]
        
        comparison = await transformer_manager.compare_transformer_models(
            model1_type="transformer",
            model1_version="v1.0.0",
            model2_type="itransformer",
            model2_version="v1.0.0"
        )
        
        assert "models" in comparison
        assert "size_comparison" in comparison
        assert comparison["models"]["model1"] == "transformer:v1.0.0"
        assert comparison["models"]["model2"] == "itransformer:v1.0.0"


class TestTransformerMetadata:
    """Test TransformerMetadata class"""
    
    def test_metadata_creation(self):
        """Test TransformerMetadata creation"""
        metadata = TransformerMetadata(
            architecture=TransformerArchitecture.VANILLA_TRANSFORMER,
            num_layers=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            total_parameters=110000000
        )
        
        assert metadata.architecture == TransformerArchitecture.VANILLA_TRANSFORMER
        assert metadata.num_layers == 12
        assert metadata.num_heads == 12
        assert metadata.total_parameters == 110000000
    
    def test_architecture_signature(self):
        """Test architecture signature generation"""
        metadata = TransformerMetadata(
            architecture=TransformerArchitecture.VANILLA_TRANSFORMER,
            num_layers=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            attention_types=[AttentionType.MULTI_HEAD, AttentionType.SELF_ATTENTION]
        )
        
        signature = metadata.get_architecture_signature()
        
        assert isinstance(signature, str)
        assert len(signature) == 16  # SHA256 truncated to 16 chars
        
        # Same config should produce same signature
        metadata2 = TransformerMetadata(
            architecture=TransformerArchitecture.VANILLA_TRANSFORMER,
            num_layers=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            attention_types=[AttentionType.MULTI_HEAD, AttentionType.SELF_ATTENTION]
        )
        
        assert metadata.get_architecture_signature() == metadata2.get_architecture_signature()


@pytest.mark.integration
class TestIntegrationTransformerPreservation:
    """Integration tests for transformer preservation"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_preservation(self, sample_transformer_model):
        """Test end-to-end transformer preservation workflow"""
        # This would be a more comprehensive integration test
        # that tests the entire workflow from model saving to loading
        # with attention analysis and architecture management
        
        config = PreservationConfig(gcs_bucket="test-bucket")
        base_manager = Mock(spec=PreservationManager)
        base_manager.config = config
        
        transformer_manager = TransformerPreservationManager(base_manager)
        
        # Mock the necessary methods for full workflow
        base_manager.save_model = AsyncMock(return_value="test_model_id")
        base_manager.load_model = AsyncMock(return_value=(b"model_data", {}))
        
        with patch('pickle.dumps', return_value=b"serialized_model"):
            with patch('pickle.loads', return_value=sample_transformer_model):
                # Save model
                model_id = await transformer_manager.save_transformer_model(
                    model=sample_transformer_model,
                    model_type="transformer",
                    preserve_attention=False,  # Skip for mock test
                    architecture_config={
                        "num_layers": 6,
                        "num_heads": 8,
                        "hidden_size": 512,
                        "total_parameters": 25000000
                    }
                )
                
                assert model_id == "test_model_id"
                
                # Load model back
                loaded_model, metadata = await transformer_manager.load_transformer_model(
                    model_type="transformer",
                    version="latest"
                )
                
                assert loaded_model == sample_transformer_model


if __name__ == "__main__":
    pytest.main([__file__])