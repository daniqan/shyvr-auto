"""
Test Transformer-specific feature engineering enhancements
Following TDD principles - these tests should FAIL initially
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml_analysis.feature_engineer import FeatureEngineer
from src.ml_analysis.base import TechnicalIndicators, MarketFeatures
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestFeatureEngineerTransformerIntegration:
    """Test Transformer-specific feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        """FeatureEngineer instance for testing"""
        return FeatureEngineer(
            cache_ttl_minutes=5,
            enable_live_data=False  # Use placeholder data for tests
        )
    
    @pytest.fixture
    def sample_token(self):
        """Sample token for testing"""
        return DiscoveredToken(
            address="0x123...",
            name="Test Token",
            symbol="TEST",
            chain=Chain.ETHEREUM,
            price_usd=100.0,
            market_cap=1000000,
            volume_24h=100000,
            discovered_at=datetime.now(),
            discovery_source="test"
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.02)  # 2% hourly volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [1000 + np.random.normal(0, 100) for _ in prices]
        })
    
    def test_transformer_sequence_features(self, feature_engineer, sample_token, sample_price_data):
        """Test creation of sequence-based features for Transformers"""
        # This should FAIL initially - sequence features not implemented
        
        # Should have method to create sequence features
        assert hasattr(feature_engineer, 'create_transformer_sequences'), "Should have transformer sequence creation"
        
        sequences = feature_engineer.create_transformer_sequences(
            token=sample_token,
            price_data=sample_price_data,
            sequence_length=60,
            prediction_horizons=[1, 4, 24]
        )
        
        assert isinstance(sequences, dict), "Should return dictionary of sequences"
        assert 'input_sequences' in sequences, "Should contain input sequences"
        assert 'target_sequences' in sequences, "Should contain target sequences"
        
        # Input sequences should be properly shaped
        input_seq = sequences['input_sequences']
        assert input_seq.shape[1] == 60, "Should have correct sequence length"
        assert input_seq.shape[2] > 0, "Should have features"
        
        # Target sequences for multi-horizon prediction
        target_seq = sequences['target_sequences']
        assert target_seq.shape[1] == 3, "Should have 3 prediction horizons (1h, 4h, 24h)"
    
    def test_multivariate_feature_preparation(self, feature_engineer, sample_price_data):
        """Test preparation of multivariate features for iTransformer"""
        # This should FAIL initially
        
        # Should handle multiple tokens/assets
        tokens_data = {
            'BTC': sample_price_data.copy(),
            'ETH': sample_price_data.copy() * 0.05,  # Different price scale
            'SOL': sample_price_data.copy() * 0.001
        }
        
        assert hasattr(feature_engineer, 'prepare_multivariate_features'), "Should have multivariate preparation"
        
        multivariate_features = feature_engineer.prepare_multivariate_features(
            tokens_data=tokens_data,
            sequence_length=60
        )
        
        assert isinstance(multivariate_features, np.ndarray), "Should return numpy array"
        assert multivariate_features.shape[0] > 0, "Should have samples"
        assert multivariate_features.shape[1] == 60, "Should have correct sequence length"
        assert multivariate_features.shape[2] == len(tokens_data), "Should have correct number of variates"
    
    def test_patch_based_features(self, feature_engineer, sample_token, sample_price_data):
        """Test patch-based feature preparation for PatchTST"""
        # This should FAIL initially
        
        assert hasattr(feature_engineer, 'create_patch_features'), "Should have patch feature creation"
        
        patch_features = feature_engineer.create_patch_features(
            token=sample_token,
            price_data=sample_price_data,
            patch_len=16,
            stride=8
        )
        
        assert isinstance(patch_features, dict), "Should return dictionary"
        assert 'patches' in patch_features, "Should contain patches"
        assert 'patch_info' in patch_features, "Should contain patch metadata"
        
        patches = patch_features['patches']
        assert patches.shape[1] == 16, "Should have correct patch length"
        assert patches.shape[0] > 0, "Should have patches"
    
    def test_temporal_embedding_features(self, feature_engineer, sample_price_data):
        """Test temporal embedding features for Transformers"""
        # This should FAIL initially
        
        assert hasattr(feature_engineer, 'extract_temporal_embeddings'), "Should have temporal embedding extraction"
        
        temporal_features = feature_engineer.extract_temporal_embeddings(
            price_data=sample_price_data,
            embedding_dim=128
        )
        
        assert isinstance(temporal_features, np.ndarray), "Should return numpy array"
        assert temporal_features.shape[1] == 128, "Should have correct embedding dimension"
        assert temporal_features.shape[0] == len(sample_price_data), "Should match data length"
    
    @pytest.mark.asyncio
    async def test_attention_friendly_normalization(self, feature_engineer, sample_token, sample_price_data):
        """Test attention-friendly feature normalization"""
        # This should FAIL initially
        
        # Calculate technical indicators first
        indicators = await feature_engineer.calculate_technical_indicators(sample_token, sample_price_data)
        
        assert hasattr(feature_engineer, 'normalize_for_attention'), "Should have attention normalization"
        
        normalized_features = feature_engineer.normalize_for_attention(
            indicators=indicators,
            market_features=await feature_engineer.calculate_market_features()
        )
        
        assert isinstance(normalized_features, np.ndarray), "Should return numpy array"
        assert not np.any(np.isnan(normalized_features)), "Should not contain NaN values"
        assert not np.any(np.isinf(normalized_features)), "Should not contain infinite values"
        
        # Values should be in reasonable range for attention mechanisms
        assert np.all(normalized_features >= -10), "Values should not be extremely negative"
        assert np.all(normalized_features <= 10), "Values should not be extremely positive"
    
    def test_cross_asset_correlation_features(self, feature_engineer):
        """Test cross-asset correlation features for multivariate models"""
        # This should FAIL initially
        
        # Mock multiple asset data
        asset_data = {
            'BTC': pd.Series([100, 101, 99, 102, 98]),
            'ETH': pd.Series([5, 5.1, 4.9, 5.2, 4.8]),
            'SOL': pd.Series([0.1, 0.11, 0.09, 0.12, 0.08])
        }
        
        assert hasattr(feature_engineer, 'calculate_cross_asset_correlations'), "Should have correlation calculation"
        
        correlations = feature_engineer.calculate_cross_asset_correlations(asset_data)
        
        assert isinstance(correlations, dict), "Should return dictionary"
        assert 'correlation_matrix' in correlations, "Should contain correlation matrix"
        assert 'rolling_correlations' in correlations, "Should contain rolling correlations"
        
        corr_matrix = correlations['correlation_matrix']
        assert corr_matrix.shape == (3, 3), "Should be 3x3 matrix for 3 assets"
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1.0"
    
    def test_market_regime_features(self, feature_engineer, sample_price_data):
        """Test market regime detection features for Transformers"""
        # This should FAIL initially
        
        assert hasattr(feature_engineer, 'detect_market_regime'), "Should have regime detection"
        
        regime_features = feature_engineer.detect_market_regime(
            price_data=sample_price_data,
            window_size=20
        )
        
        assert isinstance(regime_features, dict), "Should return dictionary"
        assert 'regime' in regime_features, "Should contain regime classification"
        assert 'volatility_regime' in regime_features, "Should contain volatility regime"
        assert 'trend_strength' in regime_features, "Should contain trend strength"
        
        regime = regime_features['regime']
        assert regime in ['bull', 'bear', 'sideways'], f"Invalid regime: {regime}"
    
    def test_sequence_padding_and_masking(self, feature_engineer, sample_token):
        """Test sequence padding and masking for variable-length inputs"""
        # This should FAIL initially
        
        # Create data of different lengths
        short_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='1H'),
            'close': [100.0] * 30,
            'volume': [1000] * 30
        })
        
        long_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=150, freq='1H'),
            'close': [100.0] * 150,
            'volume': [1000] * 150
        })
        
        assert hasattr(feature_engineer, 'pad_and_mask_sequences'), "Should have padding and masking"
        
        # Should handle both short and long sequences
        for data in [short_data, long_data]:
            result = feature_engineer.pad_and_mask_sequences(
                token=sample_token,
                price_data=data,
                target_length=100
            )
            
            assert isinstance(result, dict), "Should return dictionary"
            assert 'padded_sequence' in result, "Should contain padded sequence"
            assert 'attention_mask' in result, "Should contain attention mask"
            
            padded_seq = result['padded_sequence']
            mask = result['attention_mask']
            
            assert padded_seq.shape[0] == 100, "Should match target length"
            assert mask.shape[0] == 100, "Mask should match target length"
            assert mask.dtype == bool, "Mask should be boolean"
    
    def test_multi_scale_temporal_features(self, feature_engineer, sample_price_data):
        """Test multi-scale temporal feature extraction"""
        # This should FAIL initially
        
        assert hasattr(feature_engineer, 'extract_multiscale_features'), "Should have multiscale extraction"
        
        multiscale_features = feature_engineer.extract_multiscale_features(
            price_data=sample_price_data,
            scales=['5min', '1H', '4H', '1D']
        )
        
        assert isinstance(multiscale_features, dict), "Should return dictionary"
        
        for scale in ['5min', '1H', '4H', '1D']:
            assert scale in multiscale_features, f"Should contain {scale} features"
            scale_features = multiscale_features[scale]
            assert isinstance(scale_features, np.ndarray), f"{scale} features should be numpy array"
    
    @pytest.mark.asyncio
    async def test_enhanced_feature_matrix_for_transformers(self, feature_engineer, sample_token, sample_price_data):
        """Test enhanced feature matrix creation for Transformer models"""
        # This should FAIL initially
        
        tokens = [sample_token]
        price_data_dict = {sample_token.address: sample_price_data}
        
        # Should have enhanced feature matrix creation
        assert hasattr(feature_engineer, 'create_transformer_feature_matrix'), "Should have transformer feature matrix creation"
        
        feature_matrix, feature_names, metadata = await feature_engineer.create_transformer_feature_matrix(
            tokens=tokens,
            price_data=price_data_dict,
            sequence_length=60,
            include_sequences=True,
            include_multivariate=True
        )
        
        assert isinstance(feature_matrix, np.ndarray), "Should return numpy array"
        assert isinstance(feature_names, list), "Should return feature names list"
        assert isinstance(metadata, dict), "Should return metadata dictionary"
        
        # Should include sequence dimension
        assert len(feature_matrix.shape) == 3, "Should be 3D array (samples, sequence, features)"
        assert feature_matrix.shape[1] == 60, "Should have correct sequence length"
        
        # Metadata should contain useful information
        assert 'sequence_length' in metadata, "Should contain sequence length"
        assert 'feature_count' in metadata, "Should contain feature count"
        assert 'normalization_params' in metadata, "Should contain normalization parameters"


class TestTransformerFeatureValidation:
    """Test validation of Transformer-specific features"""
    
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer(cache_ttl_minutes=5, enable_live_data=False)
    
    def test_sequence_length_validation(self, feature_engineer):
        """Test validation of sequence lengths for different Transformer models"""
        # This should FAIL initially
        
        assert hasattr(feature_engineer, 'validate_sequence_length'), "Should have sequence length validation"
        
        # Test different model requirements
        model_requirements = {
            'transformer': {'min_seq_len': 10, 'max_seq_len': 1000},
            'itransformer': {'min_seq_len': 20, 'max_seq_len': 500},
            'patchtst': {'min_seq_len': 32, 'max_seq_len': 1000}  # Must be divisible by patch_len
        }
        
        for model_type, requirements in model_requirements.items():
            # Valid sequence lengths
            assert feature_engineer.validate_sequence_length(
                sequence_length=100,
                model_type=model_type
            ), f"Valid sequence should be accepted for {model_type}"
            
            # Invalid sequence lengths
            assert not feature_engineer.validate_sequence_length(
                sequence_length=5,  # Too short
                model_type=model_type
            ), f"Too short sequence should be rejected for {model_type}"
    
    def test_feature_dimensionality_consistency(self, feature_engineer):
        """Test that feature dimensions are consistent across batches"""
        # This should FAIL initially
        
        # Mock data with different tokens but same structure
        token1_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'close': [100.0] * 100,
            'volume': [1000] * 100
        })
        
        token2_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'close': [50.0] * 100,
            'volume': [2000] * 100
        })
        
        assert hasattr(feature_engineer, 'validate_feature_consistency'), "Should have consistency validation"
        
        is_consistent = feature_engineer.validate_feature_consistency([
            token1_data, token2_data
        ])
        
        assert is_consistent, "Features should be consistent across tokens"
    
    def test_nan_and_inf_handling(self, feature_engineer):
        """Test handling of NaN and infinite values in transformer features"""
        # This should FAIL initially
        
        # Create data with problematic values
        problematic_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1H'),
            'close': [100.0] * 25 + [np.nan] * 10 + [np.inf] * 10 + [100.0] * 5,
            'volume': [1000] * 30 + [0] * 10 + [1000] * 10  # Zero volume
        })
        
        assert hasattr(feature_engineer, 'clean_transformer_features'), "Should have feature cleaning"
        
        cleaned_data = feature_engineer.clean_transformer_features(problematic_data)
        
        assert not np.any(np.isnan(cleaned_data.select_dtypes(include=[np.number]).values)), "Should not contain NaN"
        assert not np.any(np.isinf(cleaned_data.select_dtypes(include=[np.number]).values)), "Should not contain Inf"


if __name__ == "__main__":
    pytest.main([__file__])