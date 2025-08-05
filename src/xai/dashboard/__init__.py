"""
XAI Dashboard Components.

This module provides dashboard components for visualizing transformer attention patterns,
trading signal explanations, and regulatory compliance reports.
"""

from .attention_visualizer import AttentionVisualizer
from .trading_signal_dashboard import TradingSignalDashboard

__all__ = [
    'AttentionVisualizer',
    'TradingSignalDashboard'
]