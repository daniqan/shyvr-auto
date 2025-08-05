"""
Trading Signal Dashboard for Interpretable Trading Decisions.

This module provides basic dashboard components for visualizing trading signal explanations,
confidence scores, feature importance, and regulatory compliance information.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np

from ..trading_integration import TradingExplanation
from ..transformers.interpretable_trading_signals import (
    TradingSignalExplanation,
    AttentionBasedFeatureImportance
)

logger = logging.getLogger(__name__)


class TradingSignalDashboard:
    """
    Basic trading signal dashboard for interpretable trading decisions.
    
    Provides foundational structure for creating trading signal visualizations,
    performance metrics, and compliance reporting dashboards.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading signal dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_recent_signals = self.config.get('max_recent_signals', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.risk_threshold = self.config.get('risk_threshold', 0.8)
        
        # Color schemes for different signal types
        self.signal_colors = {
            'BUY': '#4CAF50',   # Green
            'SELL': '#F44336',  # Red
            'HOLD': '#FF9800'   # Orange
        }
        
        logger.info("Initialized TradingSignalDashboard")
    
    def create_signal_summary_data(
        self,
        trading_explanations: List[TradingExplanation],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Create summary data for trading signals dashboard.
        
        Args:
            trading_explanations: List of trading explanations
            time_window_hours: Time window for analysis
            
        Returns:
            Signal summary data structure
        """
        try:
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_explanations = []
            
            for exp in trading_explanations:
                try:
                    exp_time = datetime.fromisoformat(exp.timestamp.replace('Z', '+00:00'))
                    if exp_time >= cutoff_time:
                        recent_explanations.append(exp)
                except Exception as e:
                    logger.debug(f"Error parsing timestamp {exp.timestamp}: {str(e)}")
                    continue
            
            if not recent_explanations:
                return self._create_empty_summary()
            
            # Calculate signal distribution
            signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            confidence_scores = []
            symbols = set()
            
            for exp in recent_explanations:
                signal_type = exp.decision_type.upper()
                if signal_type in signal_counts:
                    signal_counts[signal_type] += 1
                confidence_scores.append(exp.confidence)
                symbols.add(exp.symbol)
            
            # Calculate statistics
            total_signals = len(recent_explanations)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            high_confidence_signals = len([c for c in confidence_scores if c >= self.confidence_threshold])
            
            return {
                'time_window_hours': time_window_hours,
                'total_signals': total_signals,
                'signal_distribution': signal_counts,
                'signal_percentages': {
                    signal: count / total_signals * 100 if total_signals > 0 else 0
                    for signal, count in signal_counts.items()
                },
                'average_confidence': float(avg_confidence),
                'high_confidence_signals': high_confidence_signals,
                'high_confidence_percentage': high_confidence_signals / total_signals * 100 if total_signals > 0 else 0,
                'unique_symbols': len(symbols),
                'symbols_traded': sorted(list(symbols)),
                'confidence_distribution': self._calculate_confidence_distribution(confidence_scores),
                'signal_colors': self.signal_colors
            }
            
        except Exception as e:
            logger.error(f"Error creating signal summary data: {str(e)}")
            return self._create_empty_summary()
    
    def create_confidence_analysis_data(
        self,
        trading_explanations: List[TradingExplanation]
    ) -> Dict[str, Any]:
        """
        Create confidence analysis data for dashboard.
        
        Args:
            trading_explanations: List of trading explanations
            
        Returns:
            Confidence analysis data structure
        """
        try:
            if not trading_explanations:
                return {}
            
            # Extract confidence data
            confidence_data = []
            for exp in trading_explanations:
                confidence_data.append({
                    'decision_id': exp.decision_id,
                    'timestamp': exp.timestamp,
                    'symbol': exp.symbol,
                    'signal_type': exp.decision_type,
                    'confidence': exp.confidence,
                    'model_type': exp.model_type,
                    'has_attention': exp.attention_data is not None
                })
            
            # Calculate confidence statistics by signal type
            confidence_by_signal = {}
            for signal_type in ['BUY', 'SELL', 'HOLD']:
                signal_confidences = [
                    data['confidence'] for data in confidence_data 
                    if data['signal_type'].upper() == signal_type
                ]
                
                if signal_confidences:
                    confidence_by_signal[signal_type] = {
                        'mean': float(np.mean(signal_confidences)),
                        'std': float(np.std(signal_confidences)),
                        'min': float(np.min(signal_confidences)),
                        'max': float(np.max(signal_confidences)),
                        'count': len(signal_confidences)
                    }
                else:
                    confidence_by_signal[signal_type] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                    }
            
            # Calculate confidence distribution
            all_confidences = [data['confidence'] for data in confidence_data]
            confidence_histogram = self._create_confidence_histogram(all_confidences)
            
            # Identify low confidence signals
            low_confidence_signals = [
                data for data in confidence_data 
                if data['confidence'] < self.confidence_threshold
            ]
            
            return {
                'confidence_by_signal': confidence_by_signal,
                'confidence_histogram': confidence_histogram,
                'low_confidence_signals': low_confidence_signals,
                'low_confidence_count': len(low_confidence_signals),
                'average_confidence': float(np.mean(all_confidences)) if all_confidences else 0.0,
                'confidence_threshold': self.confidence_threshold,
                'total_signals': len(confidence_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating confidence analysis data: {str(e)}")
            return {}
    
    def create_feature_importance_dashboard_data(
        self,
        trading_explanations: List[TradingExplanation],
        top_k_features: int = 15
    ) -> Dict[str, Any]:
        """
        Create feature importance dashboard data.
        
        Args:
            trading_explanations: List of trading explanations
            top_k_features: Number of top features to display
            
        Returns:
            Feature importance dashboard data
        """
        try:
            if not trading_explanations:
                return {}
            
            # Aggregate feature importance across all explanations
            feature_importance_sum = {}
            feature_importance_count = {}
            
            for exp in trading_explanations:
                if exp.explanation_data and exp.explanation_data.feature_importance:
                    for feature, importance in exp.explanation_data.feature_importance.items():
                        feature_importance_sum[feature] = feature_importance_sum.get(feature, 0.0) + abs(importance)
                        feature_importance_count[feature] = feature_importance_count.get(feature, 0) + 1
            
            # Calculate average importance
            avg_feature_importance = {}
            for feature in feature_importance_sum:
                avg_feature_importance[feature] = feature_importance_sum[feature] / feature_importance_count[feature]
            
            # Sort and take top K
            sorted_features = sorted(
                avg_feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k_features]
            
            # Create feature importance by signal type
            importance_by_signal = {}
            for signal_type in ['BUY', 'SELL', 'HOLD']:
                signal_importance = {}
                signal_count = {}
                
                for exp in trading_explanations:
                    if (exp.decision_type.upper() == signal_type and 
                        exp.explanation_data and exp.explanation_data.feature_importance):
                        
                        for feature, importance in exp.explanation_data.feature_importance.items():
                            signal_importance[feature] = signal_importance.get(feature, 0.0) + abs(importance)
                            signal_count[feature] = signal_count.get(feature, 0) + 1
                
                # Calculate averages
                avg_signal_importance = {}
                for feature in signal_importance:
                    avg_signal_importance[feature] = signal_importance[feature] / signal_count[feature]
                
                # Take top features for this signal
                top_signal_features = sorted(
                    avg_signal_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 per signal
                
                importance_by_signal[signal_type] = {
                    'features': [name for name, _ in top_signal_features],
                    'importance_values': [importance for _, importance in top_signal_features]
                }
            
            return {
                'overall_top_features': {
                    'features': [name for name, _ in sorted_features],
                    'importance_values': [importance for _, importance in sorted_features]
                },
                'importance_by_signal': importance_by_signal,
                'total_features_analyzed': len(avg_feature_importance),
                'top_k_displayed': len(sorted_features),
                'feature_coverage': self._calculate_feature_coverage(trading_explanations)
            }
            
        except Exception as e:
            logger.error(f"Error creating feature importance dashboard data: {str(e)}")
            return {}
    
    def create_performance_metrics_data(
        self,
        trading_explanations: List[TradingExplanation],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Create performance metrics data for dashboard.
        
        Args:
            trading_explanations: List of trading explanations
            time_window_hours: Time window for analysis
            
        Returns:
            Performance metrics data structure
        """
        try:
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_explanations = []
            
            for exp in trading_explanations:
                try:
                    exp_time = datetime.fromisoformat(exp.timestamp.replace('Z', '+00:00'))
                    if exp_time >= cutoff_time:
                        recent_explanations.append(exp)
                except Exception:
                    continue
            
            if not recent_explanations:
                return {}
            
            # Calculate performance metrics
            total_explanations = len(recent_explanations)
            
            # Model type distribution
            model_types = {}
            for exp in recent_explanations:
                model_type = exp.model_type
                model_types[model_type] = model_types.get(model_type, 0) + 1
            
            # Attention analysis coverage
            with_attention = len([exp for exp in recent_explanations if exp.attention_data is not None])
            attention_coverage = with_attention / total_explanations if total_explanations > 0 else 0.0
            
            # Transformer-specific metrics
            transformer_explanations = [
                exp for exp in recent_explanations 
                if 'transformer' in exp.model_type.lower()
            ]
            transformer_coverage = len(transformer_explanations) / total_explanations if total_explanations > 0 else 0.0
            
            # Quality metrics
            high_quality_explanations = len([
                exp for exp in recent_explanations
                if (exp.confidence >= self.confidence_threshold and 
                    exp.explanation_data is not None and 
                    len(exp.explanation_data.feature_importance) >= 5)
            ])
            quality_score = high_quality_explanations / total_explanations if total_explanations > 0 else 0.0
            
            # Time-based metrics
            explanation_times = []
            for exp in recent_explanations:
                if 'generation_time_ms' in exp.metadata:
                    explanation_times.append(exp.metadata['generation_time_ms'])
            
            avg_generation_time = np.mean(explanation_times) if explanation_times else 0.0
            
            return {
                'time_window_hours': time_window_hours,
                'total_explanations': total_explanations,
                'model_type_distribution': model_types,
                'attention_coverage': float(attention_coverage),
                'transformer_coverage': float(transformer_coverage),
                'quality_score': float(quality_score),
                'high_quality_count': high_quality_explanations,
                'average_generation_time_ms': float(avg_generation_time),
                'performance_indicators': {
                    'explanation_completeness': quality_score,
                    'attention_analysis_availability': attention_coverage,
                    'transformer_utilization': transformer_coverage,
                    'generation_speed': 1000.0 / (avg_generation_time + 1.0)  # Explanations per second
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating performance metrics data: {str(e)}")
            return {}
    
    def create_risk_assessment_data(
        self,
        trading_explanations: List[TradingExplanation]
    ) -> Dict[str, Any]:
        """
        Create risk assessment data for dashboard.
        
        Args:
            trading_explanations: List of trading explanations
            
        Returns:
            Risk assessment data structure
        """
        try:
            if not trading_explanations:
                return {}
            
            # Risk factor analysis
            all_risk_factors = []
            risk_factor_counts = {}
            
            for exp in trading_explanations:
                risk_factors = exp.metadata.get('risk_factors', [])
                all_risk_factors.extend(risk_factors)
                
                for risk_factor in risk_factors:
                    risk_factor_counts[risk_factor] = risk_factor_counts.get(risk_factor, 0) + 1
            
            # Most common risk factors
            common_risk_factors = sorted(
                risk_factor_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Risk level distribution
            low_confidence_signals = len([
                exp for exp in trading_explanations
                if exp.confidence < 0.5
            ])
            medium_confidence_signals = len([
                exp for exp in trading_explanations
                if 0.5 <= exp.confidence < 0.75
            ])
            high_confidence_signals = len([
                exp for exp in trading_explanations
                if exp.confidence >= 0.75
            ])
            
            total_signals = len(trading_explanations)
            
            # Anomaly detection
            anomalous_signals = len([
                exp for exp in trading_explanations
                if exp.metadata.get('attention_anomaly', {}).get('is_anomalous', False)
            ])
            
            # Risk score calculation
            overall_risk_score = self._calculate_overall_risk_score(trading_explanations)
            
            return {
                'risk_factor_analysis': {
                    'total_risk_factors': len(all_risk_factors),
                    'unique_risk_factors': len(risk_factor_counts),
                    'common_risk_factors': common_risk_factors
                },
                'confidence_risk_distribution': {
                    'low_risk': high_confidence_signals,
                    'medium_risk': medium_confidence_signals,
                    'high_risk': low_confidence_signals,
                    'percentages': {
                        'low_risk': high_confidence_signals / total_signals * 100 if total_signals > 0 else 0,
                        'medium_risk': medium_confidence_signals / total_signals * 100 if total_signals > 0 else 0,
                        'high_risk': low_confidence_signals / total_signals * 100 if total_signals > 0 else 0
                    }
                },
                'anomaly_detection': {
                    'anomalous_signals': anomalous_signals,
                    'anomaly_rate': anomalous_signals / total_signals * 100 if total_signals > 0 else 0
                },
                'overall_risk_score': float(overall_risk_score),
                'risk_threshold': self.risk_threshold,
                'total_signals_analyzed': total_signals
            }
            
        except Exception as e:
            logger.error(f"Error creating risk assessment data: {str(e)}")
            return {}
    
    def create_compliance_dashboard_data(
        self,
        trading_explanations: List[TradingExplanation],
        regulatory_frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create regulatory compliance dashboard data.
        
        Args:
            trading_explanations: List of trading explanations
            regulatory_frameworks: Optional list of regulatory frameworks to assess
            
        Returns:
            Compliance dashboard data structure
        """
        try:
            if not trading_explanations:
                return {}
            
            frameworks = regulatory_frameworks or ['MiFID_II', 'SEC', 'FCA']
            compliance_data = {}
            
            for framework in frameworks:
                compliance_data[framework] = self._assess_framework_compliance(
                    trading_explanations, framework
                )
            
            # Overall compliance statistics
            total_signals = len(trading_explanations)
            
            # Explanation completeness
            complete_explanations = len([
                exp for exp in trading_explanations
                if (exp.explanation_data is not None and 
                    len(exp.explanation_data.feature_importance) > 0)
            ])
            
            # Regulatory metadata availability
            with_regulatory_metadata = len([
                exp for exp in trading_explanations
                if 'regulatory_compliance' in exp.metadata
            ])
            
            # Audit trail completeness
            complete_audit_trails = len([
                exp for exp in trading_explanations
                if (exp.decision_id and exp.timestamp and 
                    exp.explanation_data is not None)
            ])
            
            return {
                'framework_compliance': compliance_data,
                'overall_metrics': {
                    'explanation_completeness': complete_explanations / total_signals * 100 if total_signals > 0 else 0,
                    'regulatory_metadata_coverage': with_regulatory_metadata / total_signals * 100 if total_signals > 0 else 0,
                    'audit_trail_completeness': complete_audit_trails / total_signals * 100 if total_signals > 0 else 0
                },
                'compliance_summary': {
                    'total_signals': total_signals,
                    'compliant_signals': complete_explanations,
                    'compliance_rate': complete_explanations / total_signals * 100 if total_signals > 0 else 0
                },
                'frameworks_assessed': frameworks
            }
            
        except Exception as e:
            logger.error(f"Error creating compliance dashboard data: {str(e)}")
            return {}
    
    def export_dashboard_config(
        self,
        dashboard_data: Dict[str, Any],
        format: str = 'json'
    ) -> str:
        """
        Export dashboard configuration for external visualization tools.
        
        Args:
            dashboard_data: Dashboard data dictionary
            format: Export format ('json', 'plotly', 'grafana')
            
        Returns:
            Serialized dashboard configuration
        """
        try:
            if format == 'json':
                return json.dumps(dashboard_data, indent=2, default=str)
            elif format == 'plotly':
                return self._create_plotly_dashboard_config(dashboard_data)
            elif format == 'grafana':
                return self._create_grafana_dashboard_config(dashboard_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting dashboard config: {str(e)}")
            return "{}"
    
    # Helper methods
    
    def _create_empty_summary(self) -> Dict[str, Any]:
        """Create empty summary data structure."""
        return {
            'time_window_hours': 0,
            'total_signals': 0,
            'signal_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'signal_percentages': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'average_confidence': 0.0,
            'high_confidence_signals': 0,
            'high_confidence_percentage': 0.0,
            'unique_symbols': 0,
            'symbols_traded': [],
            'confidence_distribution': [],
            'signal_colors': self.signal_colors
        }
    
    def _calculate_confidence_distribution(self, confidence_scores: List[float]) -> List[Dict[str, Any]]:
        """Calculate confidence distribution for histogram."""
        if not confidence_scores:
            return []
        
        # Create bins
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        hist, bin_edges = np.histogram(confidence_scores, bins=bins)
        
        distribution = []
        for i in range(len(hist)):
            distribution.append({
                'bin_start': float(bin_edges[i]),
                'bin_end': float(bin_edges[i+1]),
                'count': int(hist[i]),
                'percentage': float(hist[i] / len(confidence_scores) * 100)
            })
        
        return distribution
    
    def _create_confidence_histogram(self, confidence_scores: List[float]) -> Dict[str, Any]:
        """Create confidence histogram data."""
        if not confidence_scores:
            return {}
        
        # Create bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, bin_edges = np.histogram(confidence_scores, bins=bins)
        
        bin_labels = [
            'Very Low (0-0.2)',
            'Low (0.2-0.4)',
            'Medium (0.4-0.6)',
            'High (0.6-0.8)',
            'Very High (0.8-1.0)'
        ]
        
        return {
            'bin_labels': bin_labels,
            'counts': hist.tolist(),
            'percentages': (hist / len(confidence_scores) * 100).tolist(),
            'total_samples': len(confidence_scores)
        }
    
    def _calculate_feature_coverage(self, trading_explanations: List[TradingExplanation]) -> Dict[str, Any]:
        """Calculate feature coverage statistics."""
        explanations_with_features = 0
        total_features = set()
        feature_counts = {}
        
        for exp in trading_explanations:
            if exp.explanation_data and exp.explanation_data.feature_importance:
                explanations_with_features += 1
                features = set(exp.explanation_data.feature_importance.keys())
                total_features.update(features)
                
                for feature in features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        total_explanations = len(trading_explanations)
        coverage_percentage = explanations_with_features / total_explanations * 100 if total_explanations > 0 else 0
        
        return {
            'explanations_with_features': explanations_with_features,
            'total_explanations': total_explanations,
            'coverage_percentage': coverage_percentage,
            'unique_features': len(total_features),
            'most_common_features': sorted(
                feature_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def _calculate_overall_risk_score(self, trading_explanations: List[TradingExplanation]) -> float:
        """Calculate overall risk score."""
        if not trading_explanations:
            return 0.0
        
        risk_factors = []
        
        for exp in trading_explanations:
            # Base risk from confidence
            confidence_risk = 1.0 - exp.confidence
            risk_factors.append(confidence_risk)
            
            # Additional risk from anomalies
            if exp.metadata.get('attention_anomaly', {}).get('is_anomalous', False):
                risk_factors.append(0.3)  # Additional 30% risk for anomalies
            
            # Risk from low feature coverage
            if exp.explanation_data:
                feature_count = len(exp.explanation_data.feature_importance)
                if feature_count < 5:
                    risk_factors.append(0.2)  # Additional 20% risk for low feature coverage
        
        # Calculate weighted average risk
        overall_risk = min(1.0, np.mean(risk_factors)) if risk_factors else 0.0
        return overall_risk
    
    def _assess_framework_compliance(
        self,
        trading_explanations: List[TradingExplanation],
        framework: str
    ) -> Dict[str, Any]:
        """Assess compliance with specific regulatory framework."""
        total_signals = len(trading_explanations)
        
        if framework == 'MiFID_II':
            # MiFID II requirements
            with_explanations = len([
                exp for exp in trading_explanations
                if exp.explanation_data is not None
            ])
            
            with_risk_disclosure = len([
                exp for exp in trading_explanations
                if 'risk_factors' in exp.metadata
            ])
            
            compliance_score = (with_explanations + with_risk_disclosure) / (2 * total_signals) if total_signals > 0 else 0
            
            return {
                'compliance_score': float(compliance_score * 100),
                'explanation_coverage': with_explanations / total_signals * 100 if total_signals > 0 else 0,
                'risk_disclosure_coverage': with_risk_disclosure / total_signals * 100 if total_signals > 0 else 0,
                'requirements_met': compliance_score >= 0.9
            }
        
        elif framework == 'SEC':
            # SEC requirements
            with_audit_trail = len([
                exp for exp in trading_explanations
                if exp.decision_id and exp.timestamp
            ])
            
            with_risk_controls = len([
                exp for exp in trading_explanations
                if 'risk_factors' in exp.metadata
            ])
            
            compliance_score = (with_audit_trail + with_risk_controls) / (2 * total_signals) if total_signals > 0 else 0
            
            return {
                'compliance_score': float(compliance_score * 100),
                'audit_trail_coverage': with_audit_trail / total_signals * 100 if total_signals > 0 else 0,
                'risk_control_coverage': with_risk_controls / total_signals * 100 if total_signals > 0 else 0,
                'requirements_met': compliance_score >= 0.95
            }
        
        elif framework == 'FCA':
            # FCA requirements
            with_governance = len([
                exp for exp in trading_explanations
                if exp.explanation_data is not None and exp.confidence >= 0.3
            ])
            
            with_transparency = len([
                exp for exp in trading_explanations
                if exp.attention_data is not None
            ])
            
            compliance_score = (with_governance + with_transparency) / (2 * total_signals) if total_signals > 0 else 0
            
            return {
                'compliance_score': float(compliance_score * 100),
                'governance_coverage': with_governance / total_signals * 100 if total_signals > 0 else 0,
                'transparency_coverage': with_transparency / total_signals * 100 if total_signals > 0 else 0,
                'requirements_met': compliance_score >= 0.85
            }
        
        else:
            return {
                'compliance_score': 0.0,
                'requirements_met': False,
                'error': f'Unsupported framework: {framework}'
            }
    
    def _create_plotly_dashboard_config(self, dashboard_data: Dict[str, Any]) -> str:
        """Create Plotly dashboard configuration."""
        config = {
            'data': [],
            'layout': {
                'title': 'Trading Signal Dashboard',
                'showlegend': True,
                'grid': {'rows': 2, 'columns': 2}
            }
        }
        
        # Add signal distribution pie chart
        if 'signal_distribution' in dashboard_data:
            signal_dist = dashboard_data['signal_distribution']
            config['data'].append({
                'type': 'pie',
                'labels': list(signal_dist.keys()),
                'values': list(signal_dist.values()),
                'name': 'Signal Distribution'
            })
        
        return json.dumps(config, indent=2)
    
    def _create_grafana_dashboard_config(self, dashboard_data: Dict[str, Any]) -> str:
        """Create Grafana dashboard configuration."""
        config = {
            'dashboard': {
                'title': 'Trading Signal Dashboard',
                'panels': [],
                'time': {'from': 'now-24h', 'to': 'now'},
                'refresh': '5m'
            }
        }
        
        # Add panels based on available data
        panel_id = 1
        
        if 'signal_distribution' in dashboard_data:
            config['dashboard']['panels'].append({
                'id': panel_id,
                'title': 'Signal Distribution',
                'type': 'piechart',
                'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}
            })
            panel_id += 1
        
        return json.dumps(config, indent=2)