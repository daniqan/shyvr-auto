"""
Enhanced Trading Explanation Manager with Transformer Support.

This module extends the basic trading explanation manager with comprehensive
transformer-specific explanation generation, attention analysis, and regulatory compliance features.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np
import torch

from .trading_integration import TradingExplanationManager as BaseTradingExplanationManager
from .trading_integration import TradingExplanation
from .transformers.interpretable_trading_signals import (
    InterpretableTradingSignalGenerator,
    TradingSignalExplanation,
    MarketEventAttribution,
    TemporalContribution,
    AttentionBasedFeatureImportance
)

logger = logging.getLogger(__name__)


class TradingExplanationManager(BaseTradingExplanationManager):
    """
    Enhanced Trading Explanation Manager with comprehensive transformer support.
    
    Extends the base manager with attention-based explanations, interpretable signals,
    real-time streaming, regulatory compliance, and advanced caching mechanisms.
    """
    
    def __init__(
        self,
        explainer_factory=None,
        metrics_collector=None,
        cache_size: int = 1000,
        explanation_timeout: float = 5.0
    ):
        """Initialize enhanced trading explanation manager."""
        super().__init__(explainer_factory, metrics_collector, cache_size, explanation_timeout)
        
        # Enhanced transformer-specific configuration
        self._signal_generators_cache: Dict[str, InterpretableTradingSignalGenerator] = {}
        self._attention_pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._attention_cache_size = 200
        self._attention_drift_baseline: Dict[str, List[float]] = {}
        
        # Performance optimization
        self._enable_parallel_processing = True
        self._max_concurrent_explanations = 5
        
        # Regulatory compliance
        self._regulatory_frameworks = ['MiFID_II', 'SEC_US', 'FCA_UK']
        self._compliance_templates = self._load_compliance_templates()
        
        logger.info("Enhanced TradingExplanationManager initialized with transformer support")
    
    def _is_transformer_model(self, model: Any) -> bool:
        """Enhanced transformer model detection."""
        # Check explicit model type
        model_type = getattr(model, 'model_type', None)
        if model_type:
            transformer_types = [
                'itransformer', 'patchtst', 'timesmixer', 'timesfm', 
                'transformer', 'attention', 'bert', 'gpt'
            ]
            return any(t_type in model_type.lower() for t_type in transformer_types)
        
        # Check class name
        class_name = model.__class__.__name__.lower()
        transformer_indicators = [
            'transformer', 'attention', 'itransformer', 'patchtst', 
            'timesmixer', 'timesfm', 'bert', 'gpt'
        ]
        return any(indicator in class_name for indicator in transformer_indicators)
    
    def _get_transformer_model_type(self, model: Any) -> str:
        """Get specific transformer model type."""
        model_type = getattr(model, 'model_type', model.__class__.__name__)
        return model_type.lower()
    
    def _supports_attention_analysis(self, model: Any) -> bool:
        """Check if model supports comprehensive attention analysis."""
        required_methods = ['get_attention_weights']
        optional_methods = ['get_layer_attention_weights', 'forward_with_attention']
        
        has_required = any(hasattr(model, method) for method in required_methods)
        has_optional = any(hasattr(model, method) for method in optional_methods)
        
        return has_required or has_optional
    
    async def generate_attention_based_explanation(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        decision_type: str,
        symbol: str,
        include_temporal_analysis: bool = True,
        include_market_events: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attention-based explanation for transformer models.
        
        Args:
            model: Transformer model
            feature_data: Input feature data
            feature_names: List of feature names
            decision_type: Trading decision type
            symbol: Trading symbol
            include_temporal_analysis: Include temporal analysis
            include_market_events: Include market event analysis
            
        Returns:
            Comprehensive explanation dictionary
        """
        try:
            # Get or create signal generator
            generator_key = f"{id(model)}_{symbol}"
            if generator_key not in self._signal_generators_cache:
                generator = InterpretableTradingSignalGenerator(
                    model=model,
                    feature_names=feature_names,
                    config={
                        'explanation_timeout': 0.5,
                        'enable_market_events': include_market_events,
                        'enable_temporal_analysis': include_temporal_analysis,
                        'narrative_generation': True,
                        'enable_caching': True
                    }
                )
                self._signal_generators_cache[generator_key] = generator
            else:
                generator = self._signal_generators_cache[generator_key]
            
            # Convert feature data to numpy array
            if isinstance(feature_data, list):
                feature_data = np.array(feature_data)
            
            # Generate timestamps for temporal analysis
            timestamps = [
                datetime.now() - timedelta(hours=i) 
                for i in range(len(feature_data), 0, -1)
            ] if include_temporal_analysis else None
            
            # Generate full explanation
            explanation = generator.generate_full_explanation(
                input_data=feature_data,
                timestamps=timestamps,
                include_narrative=True,
                include_market_events=include_market_events,
                include_temporal_analysis=include_temporal_analysis
            )
            
            # Extract attention weights for analysis
            attention_weights = model.get_attention_weights(feature_data)
            
            # Enhanced attention analysis
            attention_analysis = {
                'attention_weights': attention_weights.detach().cpu().numpy().tolist() if isinstance(attention_weights, torch.Tensor) else attention_weights,
                'head_analysis': self._analyze_attention_heads(attention_weights),
                'attention_entropy': explanation['feature_importance']['attention_entropy'],
                'attention_sparsity': explanation['feature_importance']['attention_sparsity']
            }
            
            # Combine results
            enhanced_explanation = {
                **explanation,
                'attention_analysis': attention_analysis,
                'temporal_contributions': explanation.get('temporal_contributions', {}),
                'trading_narrative': explanation.get('narrative', {}),
                'confidence_breakdown': {
                    'attention_confidence': explanation['confidence_score'],
                    'feature_confidence': self._calculate_feature_confidence(explanation['feature_importance']),
                    'temporal_confidence': self._calculate_temporal_confidence(explanation.get('temporal_contributions'))
                }
            }
            
            return enhanced_explanation
            
        except Exception as e:
            logger.error(f"Error generating attention-based explanation: {str(e)}")
            raise
    
    async def explain_trading_decision(
        self,
        decision_id: str,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        decision_type: str,
        symbol: str,
        model_type: str = 'ml_model',
        explainer_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enable_attention_analysis: bool = True,
        enable_interpretable_signals: bool = True,
        include_market_context: bool = True,
        cache_attention_patterns: bool = True,
        enable_anomaly_detection: bool = False,
        regulatory_compliance: Optional[str] = None,
        fallback_to_basic_explanation: bool = True
    ) -> Optional[TradingExplanation]:
        """
        Enhanced trading decision explanation with transformer support.
        
        Args:
            decision_id: Unique decision identifier
            model: Trading model
            feature_data: Input features
            feature_names: Feature names
            decision_type: Decision type
            symbol: Trading symbol
            model_type: Model type
            explainer_type: Explainer type
            metadata: Additional metadata
            enable_attention_analysis: Enable attention analysis
            enable_interpretable_signals: Enable interpretable signals
            include_market_context: Include market context
            cache_attention_patterns: Cache attention patterns
            enable_anomaly_detection: Enable anomaly detection
            regulatory_compliance: Regulatory framework
            fallback_to_basic_explanation: Use fallback on failure
            
        Returns:
            Enhanced TradingExplanation
        """
        try:
            # First try base explanation
            base_explanation = await super().explain_trading_decision(
                decision_id=decision_id,
                model=model,
                feature_data=feature_data,
                feature_names=feature_names,
                decision_type=decision_type,
                symbol=symbol,
                model_type=model_type,
                explainer_type=explainer_type,
                metadata=metadata,
                enable_attention_analysis=enable_attention_analysis
            )
            
            if base_explanation is None and not fallback_to_basic_explanation:
                return None
            
            # Enhance with transformer-specific analysis if applicable
            if enable_interpretable_signals and self._is_transformer_model(model):
                try:
                    # Generate interpretable signals
                    enhanced_explanation = await self.generate_attention_based_explanation(
                        model=model,
                        feature_data=feature_data,
                        feature_names=feature_names,
                        decision_type=decision_type,
                        symbol=symbol,
                        include_temporal_analysis=True,
                        include_market_events=include_market_context
                    )
                    
                    # Add interpretable signals to metadata
                    if base_explanation:
                        base_explanation.metadata['interpretable_signals'] = enhanced_explanation
                        
                        # Update attention data
                        base_explanation.attention_data = enhanced_explanation.get('attention_analysis', {})
                    else:
                        # Create new explanation from enhanced analysis
                        base_explanation = TradingExplanation(
                            decision_id=decision_id,
                            timestamp=datetime.utcnow().isoformat() + 'Z',
                            decision_type=decision_type,
                            symbol=symbol,
                            explanation_data=self._create_explanation_data_from_enhanced(enhanced_explanation, feature_names),
                            model_type=model_type,
                            confidence=enhanced_explanation['confidence_score'],
                            metadata={'interpretable_signals': enhanced_explanation},
                            attention_data=enhanced_explanation.get('attention_analysis', {})
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to generate interpretable signals: {str(e)}")
                    if not fallback_to_basic_explanation:
                        return None
            
            # Add anomaly detection if enabled
            if enable_anomaly_detection and base_explanation and base_explanation.attention_data:
                anomaly_analysis = self._detect_attention_anomalies(
                    base_explanation.attention_data, symbol
                )
                base_explanation.metadata['attention_anomaly'] = anomaly_analysis
                
                # Reduce confidence for anomalous patterns
                if anomaly_analysis.get('is_anomalous', False):
                    base_explanation.confidence *= 0.7  # Reduce confidence by 30%
            
            # Add regulatory compliance if requested
            if regulatory_compliance and base_explanation:
                compliance_data = self._generate_regulatory_compliance(
                    base_explanation, regulatory_compliance
                )
                base_explanation.metadata['regulatory_compliance'] = compliance_data
            
            # Add model-specific analysis
            if base_explanation and self._is_transformer_model(model):
                model_type_detected = self._get_transformer_model_type(model)
                base_explanation.model_type = model_type_detected
                
                model_analysis = self._generate_model_specific_analysis(model, model_type_detected)
                base_explanation.metadata['model_specific_analysis'] = model_analysis
            
            # Cache attention patterns if enabled
            if cache_attention_patterns and base_explanation and base_explanation.attention_data:
                self._cache_attention_pattern(symbol, base_explanation.attention_data)
            
            return base_explanation
            
        except Exception as e:
            logger.error(f"Error in enhanced explain_trading_decision: {str(e)}")
            
            if fallback_to_basic_explanation:
                try:
                    # Return basic explanation as fallback
                    fallback_explanation = await super().explain_trading_decision(
                        decision_id=decision_id,
                        model=model,
                        feature_data=feature_data,
                        feature_names=feature_names,
                        decision_type=decision_type,
                        symbol=symbol,
                        model_type=model_type,
                        explainer_type=explainer_type,
                        metadata=metadata,
                        enable_attention_analysis=False
                    )
                    
                    if fallback_explanation:
                        fallback_explanation.metadata['transformer_failure'] = {
                            'failed': True,
                            'error_message': str(e),
                            'fallback_explanation_used': True
                        }
                        fallback_explanation.confidence *= 0.5  # Reduce confidence
                    
                    return fallback_explanation
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback explanation also failed: {str(fallback_error)}")
                    return None
            
            return None
    
    async def generate_attention_based_narrative(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        decision_type: str,
        symbol: str,
        market_events: Optional[List[Dict[str, Any]]] = None,
        prediction_confidence: float = 0.85
    ) -> Dict[str, str]:
        """
        Generate attention-based trading narratives.
        
        Args:
            model: Transformer model
            feature_data: Feature data
            feature_names: Feature names
            decision_type: Decision type
            symbol: Trading symbol
            market_events: Market events
            prediction_confidence: Prediction confidence
            
        Returns:
            Narrative dictionary
        """
        try:
            # Get or create signal generator
            generator_key = f"{id(model)}_{symbol}"
            if generator_key not in self._signal_generators_cache:
                generator = InterpretableTradingSignalGenerator(
                    model=model,
                    feature_names=feature_names
                )
                self._signal_generators_cache[generator_key] = generator
            else:
                generator = self._signal_generators_cache[generator_key]
            
            # Convert feature data
            if isinstance(feature_data, list):
                feature_data = np.array(feature_data)
            
            # Generate timestamps
            timestamps = [
                datetime.now() - timedelta(hours=i) 
                for i in range(len(feature_data), 0, -1)
            ]
            
            # Generate narrative
            narrative = generator.generate_trading_narrative(
                input_data=feature_data,
                prediction_value=prediction_confidence,
                timestamps=timestamps,
                market_events=market_events
            )
            
            # Add attention insights
            attention_weights = model.get_attention_weights(feature_data)
            attention_insights = self._generate_attention_insights(
                attention_weights, feature_names, decision_type
            )
            
            # Add market event analysis
            market_analysis = self._generate_market_event_analysis(
                market_events, attention_weights, timestamps
            ) if market_events else "No significant market events detected in analysis window."
            
            enhanced_narrative = {
                **narrative,
                'attention_insights': attention_insights,
                'market_event_analysis': market_analysis
            }
            
            return enhanced_narrative
            
        except Exception as e:
            logger.error(f"Error generating attention-based narrative: {str(e)}")
            return {
                'executive_summary': f"Unable to generate narrative for {decision_type} decision on {symbol}.",
                'attention_insights': "Attention analysis unavailable due to processing error.",
                'temporal_analysis': "Temporal analysis unavailable.",
                'market_event_analysis': "Market event analysis unavailable.",
                'risk_assessment': "Risk assessment incomplete due to technical issues.",
                'recommendation': "Manual review recommended."
            }
    
    async def generate_multi_asset_explanations(
        self,
        model: Any,
        feature_data: Union[np.ndarray, List[float]],
        feature_names: List[str],
        asset_names: List[str],
        decision_id: str
    ) -> Dict[str, TradingExplanation]:
        """
        Generate explanations for multi-asset scenarios.
        
        Args:
            model: Multi-asset model
            feature_data: Multi-asset feature data
            feature_names: Feature names
            asset_names: Asset names
            decision_id: Decision ID base
            
        Returns:
            Dictionary mapping assets to explanations
        """
        try:
            # Get or create multi-asset signal generator
            generator_key = f"{id(model)}_multiasset"
            if generator_key not in self._signal_generators_cache:
                generator = InterpretableTradingSignalGenerator(
                    model=model,
                    feature_names=feature_names,
                    config={'multi_asset_mode': True}
                )
                self._signal_generators_cache[generator_key] = generator
            else:
                generator = self._signal_generators_cache[generator_key]
            
            # Convert feature data
            if isinstance(feature_data, list):
                feature_data = np.array(feature_data)
            
            # Generate multi-asset explanations
            multi_explanations = generator.generate_multi_asset_explanations(
                input_data=feature_data,
                asset_names=asset_names
            )
            
            # Convert to TradingExplanation objects
            trading_explanations = {}
            
            for asset in asset_names:
                if asset in multi_explanations:
                    explanation_data = multi_explanations[asset]
                    
                    # Create TradingExplanation
                    trading_explanation = TradingExplanation(
                        decision_id=f"{decision_id}_{asset}",
                        timestamp=datetime.utcnow().isoformat() + 'Z',
                        decision_type=explanation_data['signal_interpretation']['signal_type'],
                        symbol=asset,
                        explanation_data=self._create_explanation_data_from_enhanced(
                            explanation_data, feature_names
                        ),
                        model_type=self._get_transformer_model_type(model),
                        confidence=explanation_data['confidence_score'],
                        metadata={'cross_asset_influences': explanation_data.get('cross_asset_influences', {})},
                        attention_data=explanation_data.get('attention_analysis', {})
                    )
                    
                    trading_explanations[asset] = trading_explanation
            
            return trading_explanations
            
        except Exception as e:
            logger.error(f"Error generating multi-asset explanations: {str(e)}")
            return {}
    
    async def generate_streaming_explanations(
        self,
        model: Any,
        streaming_data: List[Dict[str, Any]],
        feature_names: List[str],
        max_latency_ms: int = 100
    ) -> AsyncIterator[TradingExplanation]:
        """
        Generate real-time streaming explanations.
        
        Args:
            model: Trading model
            streaming_data: Stream of market data
            feature_names: Feature names
            max_latency_ms: Maximum latency in milliseconds
            
        Yields:
            TradingExplanation objects
        """
        try:
            # Sort by timestamp (most recent first)
            sorted_data = sorted(
                streaming_data, 
                key=lambda x: x['timestamp'], 
                reverse=True
            )
            
            for i, data_point in enumerate(sorted_data):
                start_time = time.time()
                
                try:
                    # Generate explanation
                    explanation = await self.explain_trading_decision(
                        decision_id=f"stream_{i}_{int(time.time())}",
                        model=model,
                        feature_data=data_point['feature_data'],
                        feature_names=feature_names,
                        decision_type=data_point['decision_type'],
                        symbol=data_point['symbol'],
                        enable_attention_analysis=True,
                        enable_interpretable_signals=True
                    )
                    
                    if explanation:
                        # Check latency requirement
                        elapsed_ms = (time.time() - start_time) * 1000
                        if elapsed_ms > max_latency_ms:
                            logger.warning(f"Explanation exceeded latency requirement: {elapsed_ms:.1f}ms")
                        
                        yield explanation
                
                except Exception as e:
                    logger.error(f"Error in streaming explanation {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in streaming explanations: {str(e)}")
    
    def get_attention_cache_statistics(self) -> Dict[str, Any]:
        """Get attention-specific cache statistics."""
        base_stats = self.get_cache_stats()
        
        attention_stats = {
            'attention_cache_hits': self._cache_stats.get('attention_cache_hits', 0),
            'attention_cache_misses': self._cache_stats.get('attention_cache_misses', 0),
            'attention_patterns_cached': len(self._attention_pattern_cache),
            'signal_generators_cached': len(self._signal_generators_cache)
        }
        
        return {**base_stats, **attention_stats}
    
    def get_attention_drift_statistics(
        self,
        symbol: str,
        time_window_hours: int = 1
    ) -> Dict[str, Any]:
        """Get attention drift statistics."""
        if symbol not in self._attention_drift_baseline:
            return {
                'average_drift_score': 0.0,
                'max_drift_score': 0.0,
                'drift_trend': 'stable',
                'attention_stability': 1.0
            }
        
        baseline = self._attention_drift_baseline[symbol]
        recent_scores = baseline[-time_window_hours:] if len(baseline) >= time_window_hours else baseline
        
        if not recent_scores:
            return {
                'average_drift_score': 0.0,
                'max_drift_score': 0.0,
                'drift_trend': 'stable',
                'attention_stability': 1.0
            }
        
        avg_drift = np.mean(recent_scores)
        max_drift = np.max(recent_scores)
        
        # Determine trend
        if len(recent_scores) >= 3:
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            if trend_slope > 0.1:
                drift_trend = 'increasing'
            elif trend_slope < -0.1:
                drift_trend = 'decreasing'
            else:
                drift_trend = 'stable'
        else:
            drift_trend = 'stable'
        
        attention_stability = max(0.0, 1.0 - avg_drift)
        
        return {
            'average_drift_score': avg_drift,
            'max_drift_score': max_drift,
            'drift_trend': drift_trend,
            'attention_stability': attention_stability
        }
    
    def generate_attention_visualization_data(
        self,
        attention_weights: torch.Tensor,
        feature_names: List[str],
        timestamps: List[datetime],
        top_k_features: int = 10
    ) -> Dict[str, Any]:
        """Generate data for attention pattern visualization."""
        try:
            # Convert to numpy
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]  # Take first batch
            
            # Generate attention heatmap
            avg_attention = np.mean(attention_weights, axis=0)
            heatmap_data = {
                'data': avg_attention.tolist(),
                'labels': [ts.strftime('%H:%M') for ts in timestamps] if timestamps else [],
                'shape': avg_attention.shape
            }
            
            # Generate feature importance chart
            feature_importance = np.mean(avg_attention, axis=1)
            top_indices = np.argsort(feature_importance)[-top_k_features:]
            
            feature_chart = {
                'features': [feature_names[i] for i in top_indices] if len(feature_names) > max(top_indices) else [],
                'importance_scores': [feature_importance[i] for i in top_indices]
            }
            
            # Generate temporal attention chart
            temporal_attention = np.mean(avg_attention, axis=0)
            temporal_chart = {
                'timestamps': [ts.isoformat() for ts in timestamps] if timestamps else [],
                'attention_values': temporal_attention.tolist()
            }
            
            # Generate head analysis chart
            head_analysis = {}
            for head in range(attention_weights.shape[0]):
                head_pattern = np.mean(attention_weights[head], axis=0)
                head_analysis[f'head_{head}'] = {
                    'specialization_score': np.std(head_pattern),
                    'focus_pattern': head_pattern.tolist()
                }
            
            return {
                'attention_heatmap': heatmap_data,
                'feature_importance_chart': feature_chart,
                'temporal_attention_chart': temporal_chart,
                'head_analysis_chart': head_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating attention visualization data: {str(e)}")
            return {}
    
    def calculate_attention_summary_statistics(
        self,
        attention_data: List[Dict[str, Any]],
        time_window_hours: int = 1
    ) -> Dict[str, Any]:
        """Calculate attention summary statistics."""
        try:
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_data = [
                data for data in attention_data
                if data['timestamp'] >= cutoff_time
            ]
            
            if not recent_data:
                return {}
            
            # Calculate statistics
            entropies = []
            sparsities = []
            specializations = []
            consistency_scores = []
            
            for data in recent_data:
                attention_weights = data['attention_weights']
                if isinstance(attention_weights, torch.Tensor):
                    attention_weights = attention_weights.detach().cpu().numpy()
                
                # Calculate entropy
                flat_attention = attention_weights.flatten()
                flat_attention = flat_attention / (np.sum(flat_attention) + 1e-8)
                entropy = -np.sum(flat_attention * np.log(flat_attention + 1e-8))
                entropies.append(entropy)
                
                # Calculate sparsity
                threshold = 0.1 * np.mean(attention_weights)
                sparsity = np.mean(attention_weights < threshold)
                sparsities.append(sparsity)
                
                # Calculate head specialization
                if attention_weights.ndim >= 3:
                    head_patterns = [np.mean(attention_weights[i], axis=0) for i in range(attention_weights.shape[0])]
                    correlations = []
                    for i in range(len(head_patterns)):
                        for j in range(i+1, len(head_patterns)):
                            corr = np.corrcoef(head_patterns[i], head_patterns[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    specialization = 1.0 - np.mean(correlations) if correlations else 0.0
                    specializations.append(specialization)
            
            # Calculate temporal patterns
            temporal_patterns = []
            if len(entropies) > 3:
                # Check for trends
                entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
                if abs(entropy_trend) > 0.1:
                    temporal_patterns.append('entropy_trend')
                
                # Check for periodicity
                if len(entropies) >= 12:  # At least 12 data points
                    fft = np.fft.fft(entropies)
                    frequencies = np.fft.fftfreq(len(entropies))
                    dominant_freq = frequencies[np.argmax(np.abs(fft[1:]))+1]
                    if abs(dominant_freq) > 0.1:
                        temporal_patterns.append('periodic_pattern')
            
            return {
                'average_attention_entropy': np.mean(entropies) if entropies else 0.0,
                'attention_sparsity_trend': np.mean(sparsities) if sparsities else 0.0,
                'head_specialization_stability': np.mean(specializations) if specializations else 0.0,
                'temporal_focus_patterns': temporal_patterns,
                'attention_consistency_score': 1.0 - np.std(entropies) if len(entropies) > 1 else 1.0,
                'data_points_analyzed': len(recent_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating attention summary statistics: {str(e)}")
            return {}
    
    def export_explanation_to_json(self, explanation: TradingExplanation) -> str:
        """Export explanation to JSON format."""
        try:
            explanation_dict = self.to_dict(explanation)
            return json.dumps(explanation_dict, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting explanation to JSON: {str(e)}")
            return "{}"
    
    def export_explanation_for_reporting(
        self,
        explanation: TradingExplanation,
        format: str = 'structured_dict'
    ) -> Dict[str, Any]:
        """Export explanation for reporting purposes."""
        try:
            if format == 'structured_dict':
                return {
                    'executive_summary': self._generate_executive_summary(explanation),
                    'technical_analysis': self._generate_technical_analysis(explanation),
                    'attention_insights': self._generate_attention_report(explanation),
                    'risk_assessment': self._generate_risk_report(explanation)
                }
            elif format == 'pdf_ready':
                return {
                    'title': f"Trading Decision Analysis - {explanation.symbol}",
                    'subtitle': f"{explanation.decision_type} Signal on {explanation.timestamp}",
                    'sections': [
                        {
                            'title': 'Executive Summary',
                            'content': self._generate_executive_summary(explanation)
                        },
                        {
                            'title': 'Technical Analysis',
                            'content': self._generate_technical_analysis(explanation)
                        },
                        {
                            'title': 'Attention Analysis',
                            'content': self._generate_attention_report(explanation)
                        },
                        {
                            'title': 'Risk Assessment',
                            'content': self._generate_risk_report(explanation)
                        }
                    ]
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting explanation for reporting: {str(e)}")
            return {}
    
    # Helper methods
    
    def _analyze_attention_heads(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze individual attention heads."""
        try:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            if attention_weights.ndim < 3:
                return {}
            
            num_heads = attention_weights.shape[0] if attention_weights.ndim == 3 else attention_weights.shape[1]
            head_analysis = {}
            
            for head in range(num_heads):
                if attention_weights.ndim == 4:
                    head_attention = attention_weights[0, head]
                else:
                    head_attention = attention_weights[head]
                
                # Calculate head-specific metrics
                entropy = -np.sum(head_attention * np.log(head_attention + 1e-8))
                sparsity = np.mean(head_attention < 0.1 * np.mean(head_attention))
                focus_strength = np.max(head_attention) / np.mean(head_attention)
                
                head_analysis[f'head_{head}'] = {
                    'entropy': float(entropy),
                    'sparsity': float(sparsity),
                    'focus_strength': float(focus_strength),
                    'specialization': float(np.std(head_attention))
                }
            
            return head_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing attention heads: {str(e)}")
            return {}
    
    def _calculate_feature_confidence(self, feature_importance: Dict[str, Any]) -> float:
        """Calculate confidence based on feature importance distribution."""
        try:
            scores = list(feature_importance.get('feature_scores', {}).values())
            if not scores:
                return 0.5
            
            # Higher confidence for more concentrated importance
            entropy = feature_importance.get('attention_entropy', 0)
            sparsity = feature_importance.get('attention_sparsity', 0)
            
            confidence = 0.5 + 0.3 * sparsity + 0.2 * (1.0 - min(1.0, entropy / 10.0))
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating feature confidence: {str(e)}")
            return 0.5
    
    def _calculate_temporal_confidence(self, temporal_contrib: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence based on temporal contribution patterns."""
        try:
            if not temporal_contrib or 'time_importance' not in temporal_contrib:
                return 0.5
            
            time_importance = temporal_contrib['time_importance']
            if not time_importance:
                return 0.5
            
            # Higher confidence for more focused temporal attention
            max_importance = max(time_importance)
            avg_importance = np.mean(time_importance)
            
            focus_ratio = max_importance / (avg_importance + 1e-8)
            confidence = 0.5 + 0.5 * min(1.0, (focus_ratio - 1.0) / 3.0)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating temporal confidence: {str(e)}")
            return 0.5
    
    def _create_explanation_data_from_enhanced(
        self,
        enhanced_explanation: Dict[str, Any],
        feature_names: List[str]
    ):
        """Create ExplanationData from enhanced explanation."""
        from .data_models import ExplanationData
        
        feature_scores = enhanced_explanation.get('feature_importance', {}).get('feature_scores', {})
        
        return ExplanationData(
            feature_importance=feature_scores,
            explanation_type='attention',
            instance_data=[0.0] * len(feature_names),  # Placeholder
            model_prediction=enhanced_explanation.get('confidence_score', 0.5),
            confidence_score=enhanced_explanation.get('confidence_score', 0.5)
        )
    
    def _detect_attention_anomalies(
        self,
        attention_data: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Detect anomalous attention patterns."""
        try:
            # Simple anomaly detection based on entropy and sparsity
            entropy = attention_data.get('attention_entropy', 0)
            sparsity = attention_data.get('attention_sparsity', 0)
            
            # Define normal ranges (these would be learned from historical data)
            normal_entropy_range = (1.0, 8.0)
            normal_sparsity_range = (0.1, 0.8)
            
            is_anomalous = (
                entropy < normal_entropy_range[0] or entropy > normal_entropy_range[1] or
                sparsity < normal_sparsity_range[0] or sparsity > normal_sparsity_range[1]
            )
            
            anomaly_score = 0.0
            if entropy < normal_entropy_range[0]:
                anomaly_score += (normal_entropy_range[0] - entropy) / normal_entropy_range[0]
            elif entropy > normal_entropy_range[1]:
                anomaly_score += (entropy - normal_entropy_range[1]) / normal_entropy_range[1]
            
            if sparsity < normal_sparsity_range[0]:
                anomaly_score += (normal_sparsity_range[0] - sparsity) / normal_sparsity_range[0]
            elif sparsity > normal_sparsity_range[1]:
                anomaly_score += (sparsity - normal_sparsity_range[1]) / normal_sparsity_range[1]
            
            anomaly_description = ""
            if is_anomalous:
                if entropy < normal_entropy_range[0]:
                    anomaly_description += "Extremely focused attention pattern detected. "
                elif entropy > normal_entropy_range[1]:
                    anomaly_description += "Highly dispersed attention pattern detected. "
                
                if sparsity < normal_sparsity_range[0]:
                    anomaly_description += "Unusually dense attention distribution. "
                elif sparsity > normal_sparsity_range[1]:
                    anomaly_description += "Extremely sparse attention pattern. "
            
            return {
                'is_anomalous': is_anomalous,
                'anomaly_score': min(1.0, anomaly_score),
                'anomaly_description': anomaly_description.strip(),
                'entropy': entropy,
                'sparsity': sparsity
            }
            
        except Exception as e:
            logger.error(f"Error detecting attention anomalies: {str(e)}")
            return {
                'is_anomalous': False,
                'anomaly_score': 0.0,
                'anomaly_description': 'Anomaly detection failed',
                'entropy': 0.0,
                'sparsity': 0.0
            }
    
    def _generate_regulatory_compliance(
        self,
        explanation: TradingExplanation,
        framework: str
    ) -> Dict[str, Any]:
        """Generate regulatory compliance data."""
        compliance_template = self._compliance_templates.get(framework, {})
        
        return {
            'framework': framework,
            'decision_rationale': self._generate_compliance_rationale(explanation, framework),
            'risk_disclosure': self._generate_risk_disclosure(explanation, framework),
            'model_transparency': self._generate_model_transparency(explanation, framework),
            'audit_trail': {
                'decision_id': explanation.decision_id,
                'timestamp': explanation.timestamp,
                'model_type': explanation.model_type,
                'confidence_level': explanation.confidence,
                'explanation_method': 'attention_based_transformer_analysis'
            }
        }
    
    def _generate_model_specific_analysis(self, model: Any, model_type: str) -> Dict[str, Any]:
        """Generate model-specific analysis."""
        analysis = {'model_type': model_type}
        
        if 'itransformer' in model_type:
            analysis['multivariate_attention'] = "Supports cross-variate attention analysis"
            analysis['variate_embeddings'] = "Uses independent variate embeddings"
        elif 'patchtst' in model_type:
            analysis['patch_attention'] = "Processes data in patches for efficiency"
            analysis['channel_independence'] = "Maintains channel independence"
        elif 'timesmixer' in model_type:
            analysis['decomposable_mixing'] = "Uses decomposable mixing operators"
            analysis['multi_scale_patterns'] = "Captures multi-scale temporal patterns"
        elif 'timesfm' in model_type:
            analysis['foundation_model'] = "Large-scale foundation model"
            analysis['zero_shot_prediction'] = "Supports zero-shot forecasting"
        
        return analysis
    
    def _cache_attention_pattern(self, symbol: str, attention_data: Dict[str, Any]) -> None:
        """Cache attention pattern for future reference."""
        if len(self._attention_pattern_cache) >= self._attention_cache_size:
            # Remove oldest entry
            oldest_key = min(self._attention_pattern_cache.keys())
            del self._attention_pattern_cache[oldest_key]
        
        cache_key = f"{symbol}_{int(time.time())}"
        self._attention_pattern_cache[cache_key] = attention_data
    
    def _generate_attention_insights(
        self,
        attention_weights: torch.Tensor,
        feature_names: List[str],
        decision_type: str
    ) -> str:
        """Generate human-readable attention insights."""
        try:
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            if attention_weights.ndim == 4:
                attention_weights = attention_weights[0]
            
            # Calculate key statistics
            avg_attention = np.mean(attention_weights, axis=0)
            feature_attention = np.mean(avg_attention, axis=1)
            
            # Find top attended features
            top_indices = np.argsort(feature_attention)[-3:]
            top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
            
            insights = f"The transformer model's attention mechanism for this {decision_type} decision "
            insights += f"primarily focuses on {', '.join(top_features)}. "
            
            # Analyze attention distribution
            entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-8))
            if entropy > 6.0:
                insights += "The attention pattern is highly distributed, suggesting the model considers "
                insights += "multiple factors with similar importance. "
            else:
                insights += "The attention pattern is focused, indicating strong conviction in specific "
                insights += "market indicators. "
            
            # Analyze temporal patterns
            temporal_attention = np.mean(avg_attention, axis=0)
            if len(temporal_attention) > 10:
                recent_focus = np.mean(temporal_attention[-len(temporal_attention)//4:])
                overall_focus = np.mean(temporal_attention)
                
                if recent_focus > overall_focus * 1.3:
                    insights += "There is a strong recency bias, with increased attention on recent market data. "
                elif recent_focus < overall_focus * 0.7:
                    insights += "The model shows distributed temporal attention, weighing historical patterns heavily. "
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating attention insights: {str(e)}")
            return "Attention analysis unavailable due to processing error."
    
    def _generate_market_event_analysis(
        self,
        market_events: List[Dict[str, Any]],
        attention_weights: torch.Tensor,
        timestamps: List[datetime]
    ) -> str:
        """Generate market event analysis."""
        if not market_events:
            return "No significant market events in analysis window."
        
        analysis = "Market event analysis reveals "
        
        # Simple analysis - would be enhanced with actual event-attention correlation
        event_names = [event['event'] for event in market_events]
        analysis += f"attention patterns potentially influenced by: {', '.join(event_names)}. "
        
        high_impact_events = [e for e in market_events if e.get('impact', 0) > 0.7]
        if high_impact_events:
            analysis += f"High-impact events include {high_impact_events[0]['event']}, which may have "
            analysis += "significantly influenced the model's decision-making process."
        
        return analysis
    
    def _load_compliance_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load regulatory compliance templates."""
        return {
            'MiFID_II': {
                'decision_rationale_required': True,
                'risk_disclosure_required': True,
                'model_transparency_required': True,
                'client_communication_required': True
            },
            'SEC_US': {
                'decision_rationale_required': True,
                'risk_disclosure_required': True,
                'model_transparency_required': False,
                'audit_trail_required': True
            },
            'FCA_UK': {
                'decision_rationale_required': True,
                'risk_disclosure_required': True,
                'model_transparency_required': True,
                'client_communication_required': True
            }
        }
    
    def _generate_compliance_rationale(self, explanation: TradingExplanation, framework: str) -> str:
        """Generate compliance-specific rationale."""
        rationale = f"This {explanation.decision_type} decision for {explanation.symbol} was generated "
        rationale += f"using a {explanation.model_type} model with {explanation.confidence:.1%} confidence. "
        
        if explanation.attention_data:
            rationale += "The decision is based on attention-weighted analysis of market features, "
            rationale += "providing transparency into the model's decision-making process. "
        
        rationale += f"The explanation methodology complies with {framework} requirements for "
        rationale += "algorithmic trading transparency and client communication."
        
        return rationale
    
    def _generate_risk_disclosure(self, explanation: TradingExplanation, framework: str) -> str:
        """Generate risk disclosure statement."""
        disclosure = "Risk Disclosure: This trading decision is generated by an AI model and carries "
        disclosure += "inherent risks including model uncertainty, market volatility, and potential losses. "
        
        if explanation.confidence < 0.7:
            disclosure += "The model confidence is below 70%, indicating higher uncertainty. "
        
        if 'risk_factors' in explanation.metadata:
            risk_factors = explanation.metadata['risk_factors']
            if risk_factors:
                disclosure += f"Identified risk factors include: {', '.join(risk_factors)}. "
        
        disclosure += "Past performance does not guarantee future results. Please consider your risk tolerance."
        
        return disclosure
    
    def _generate_model_transparency(self, explanation: TradingExplanation, framework: str) -> Dict[str, str]:
        """Generate model transparency information."""
        return {
            'attention_mechanism': 'Multi-head attention with temporal and cross-feature analysis',
            'feature_importance_method': 'Attention-weighted feature importance calculation',
            'uncertainty_estimation': 'Confidence scores based on attention pattern analysis',
            'model_architecture': f'{explanation.model_type} transformer-based architecture',
            'training_data': 'Historical market data with temporal sequences',
            'explainability_method': 'Attention pattern analysis and feature attribution'
        }
    
    def _generate_executive_summary(self, explanation: TradingExplanation) -> str:
        """Generate executive summary for reports."""
        summary = f"Trading Decision: {explanation.decision_type} signal for {explanation.symbol} "
        summary += f"generated on {explanation.timestamp} with {explanation.confidence:.1%} confidence. "
        
        if explanation.attention_data:
            summary += "Analysis based on transformer attention patterns. "
        
        # Add key insights from metadata
        if 'interpretable_signals' in explanation.metadata:
            signals = explanation.metadata['interpretable_signals']
            if 'feature_importance' in signals:
                top_features = signals['feature_importance'].get('top_features', [])[:3]
                summary += f"Primary factors: {', '.join(top_features)}."
        
        return summary
    
    def _generate_technical_analysis(self, explanation: TradingExplanation) -> str:
        """Generate technical analysis for reports."""
        analysis = f"Model Type: {explanation.model_type}\n"
        analysis += f"Confidence Level: {explanation.confidence:.1%}\n"
        
        if explanation.explanation_data:
            analysis += f"Feature Analysis: {len(explanation.explanation_data.feature_importance)} features analyzed\n"
            
            top_features = list(explanation.explanation_data.feature_importance.items())[:5]
            analysis += "Top Contributing Features:\n"
            for feature, importance in top_features:
                analysis += f"  - {feature}: {importance:.3f}\n"
        
        return analysis
    
    def _generate_attention_report(self, explanation: TradingExplanation) -> str:
        """Generate attention analysis for reports."""
        if not explanation.attention_data:
            return "Attention analysis not available."
        
        attention_data = explanation.attention_data
        report = "Attention Pattern Analysis:\n"
        
        if 'attention_entropy' in attention_data:
            report += f"Attention Entropy: {attention_data['attention_entropy']:.2f}\n"
        
        if 'attention_sparsity' in attention_data:
            report += f"Attention Sparsity: {attention_data['attention_sparsity']:.2f}\n"
        
        if 'head_analysis' in attention_data:
            report += f"Number of Attention Heads: {len(attention_data['head_analysis'])}\n"
        
        return report
    
    def _generate_risk_report(self, explanation: TradingExplanation) -> str:
        """Generate risk assessment for reports."""
        report = f"Risk Assessment for {explanation.decision_type} Decision:\n"
        report += f"Model Confidence: {explanation.confidence:.1%}\n"
        
        if explanation.confidence < 0.7:
            report += "⚠️ Lower confidence level indicates higher uncertainty\n"
        
        if 'risk_factors' in explanation.metadata:
            risk_factors = explanation.metadata['risk_factors']
            if risk_factors:
                report += "Identified Risk Factors:\n"
                for risk in risk_factors:
                    report += f"  - {risk}\n"
        
        if explanation.attention_data and 'attention_anomaly' in explanation.metadata:
            anomaly = explanation.metadata['attention_anomaly']
            if anomaly.get('is_anomalous', False):
                report += f"⚠️ Anomalous attention pattern detected (score: {anomaly.get('anomaly_score', 0):.2f})\n"
        
        return report