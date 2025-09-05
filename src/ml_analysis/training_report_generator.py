"""
TrainingReportGenerator

Generates comprehensive HTML training reports with visualizations for ML model training.
Includes training metrics, plots, model architecture summaries, and feature importance.

Features:
- Training/validation loss and accuracy curves
- Learning rate schedules and model comparisons
- Resource usage and timing statistics
- Feature importance visualizations
- Model architecture summaries and hyperparameter tables
- Data distribution analysis
- Baseline model comparisons
- HTML generation with embedded charts
- Automatic GCS upload to trained-models/{model_type}/{timestamp}/reports/

Integration:
- Designed to integrate with UnifiedTrainingPipeline
- Automatic report generation after model training
- No manual execution required
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
from google.cloud import storage
import io
import tempfile

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart

logger = logging.getLogger(__name__)


class TrainingReportGeneratorError(Exception):
    """Base exception for TrainingReportGenerator errors"""
    pass


class ValidationError(TrainingReportGeneratorError):
    """Error validating training metrics"""
    pass


class VisualizationError(TrainingReportGeneratorError):
    """Error creating visualizations"""
    pass


class PDFGenerationError(TrainingReportGeneratorError):
    """Error generating PDF report"""
    pass


class TrainingReportGenerator:
    """
    Generates comprehensive training reports with visualizations and HTML output
    
    Creates visual HTML reports containing:
    - Training/validation loss and accuracy curves
    - Learning rate schedules
    - Model comparison charts
    - Resource usage statistics
    - Feature importance plots
    - Model architecture summaries
    - Hyperparameter tables
    - Data distribution analysis
    - Baseline comparisons
    
    Automatically uploads reports to GCS for model preservation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TrainingReportGenerator
        
        Args:
            config: Configuration dictionary with options:
                - output_dir: Local directory for report files
                - gcs_bucket: GCS bucket for upload
                - gcs_prefix: GCS path prefix
                - figure_size: Default figure size tuple
                - dpi: Image resolution
                - font_size: Default font size
                - color_palette: Matplotlib color palette
                - style: Matplotlib style
                - company_name: Company name for reports
                - include_sections: List of sections to include
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Setup output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client if bucket specified
        self.gcs_client = None
        self.gcs_bucket = None
        if self.config.get('gcs_bucket'):
            self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket(self.config['gcs_bucket'])
        
        # Setup matplotlib styling
        self._setup_plotting_style()
        
        logger.info(
            f"TrainingReportGenerator initialized - output_dir: {self.output_dir}, "
            f"gcs_bucket: {self.config.get('gcs_bucket')}, style: {self.config.get('style')}"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'output_dir': './reports',
            'gcs_bucket': 'shyvr-models-prod',
            'gcs_prefix': 'training-reports',
            'figure_size': (12, 8),
            'dpi': 300,
            'font_size': 10,
            'color_palette': 'Set1',
            'style': 'seaborn-v0_8-whitegrid',
            'company_name': 'ShyvR AI Trading',
            'logo_path': None,
            'report_author': 'ML Training System',
            'include_sections': [
                'title_page', 'executive_summary', 'model_comparison',
                'training_curves', 'hyperparameters', 'resource_usage',
                'feature_importance', 'data_distribution'
            ]
        }
    
    def _setup_plotting_style(self):
        """Setup matplotlib plotting style"""
        try:
            plt.style.use(self.config['style'])
        except OSError:
            # Fallback to default if style not available
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['figure.dpi'] = self.config['dpi']
        plt.rcParams['font.size'] = self.config['font_size']
        
        # Set color palette
        sns.set_palette(self.config['color_palette'])
    
    def validate_metrics(self, training_metrics: Dict[str, Dict[str, Any]]) -> bool:
        """
        Validate training metrics structure
        
        Args:
            training_metrics: Dictionary of training results per model
            
        Returns:
            True if valid, False if some models failed but structure is valid
            
        Raises:
            ValidationError: If metrics structure is invalid
        """
        if not training_metrics:
            raise ValueError("No training metrics provided")
        
        all_successful = True
        
        for model_name, metrics in training_metrics.items():
            if not isinstance(metrics, dict):
                raise ValueError(f"Invalid metrics structure for {model_name}")
            
            # Check if training was successful
            if metrics.get('success', False):
                # For successful training, require comprehensive fields
                required_fields = [
                    'model_type', 'training_duration_seconds',
                    'model_accuracy', 'feature_count', 'training_samples', 'config'
                ]
                for field in required_fields:
                    if field not in metrics:
                        logger.warning(f"Successful training missing field: {field} for {model_name}")
                        # Don't fail, just warn - we can still generate partial report
            else:
                # For failed training, be lenient
                all_successful = False
                if 'model_type' not in metrics:
                    metrics['model_type'] = model_name  # Infer from key
                if 'error' not in metrics and 'reason' not in metrics:
                    metrics['error'] = 'Unknown training failure'
                logger.info(f"Model {model_name} failed training: {metrics.get('error', 'Unknown')}")
        
        return True
    
    def extract_training_curves(self, model_metrics: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract training curves from model metrics
        
        Args:
            model_metrics: Metrics for a single model
            
        Returns:
            Dictionary with training curve data
        """
        curves = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        if 'training_history' in model_metrics:
            history = model_metrics['training_history']
            
            for key in curves.keys():
                if key == 'epochs':
                    # Generate epoch numbers based on data length
                    if 'train_loss' in history:
                        curves['epochs'] = list(range(1, len(history['train_loss']) + 1))
                elif key in history:
                    curves[key] = history[key]
        
        return curves
    
    def create_loss_curves_plot(self, model_metrics: Dict[str, Any], save_path: Path) -> Path:
        """
        Create training/validation loss curves plot
        
        Args:
            model_metrics: Model training metrics
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            curves = self.extract_training_curves(model_metrics)
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            if curves['train_loss'] and curves['val_loss']:
                epochs = curves['epochs'] or range(1, len(curves['train_loss']) + 1)
                
                ax.plot(epochs, curves['train_loss'], label='Training Loss', linewidth=2)
                ax.plot(epochs, curves['val_loss'], label='Validation Loss', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{model_metrics.get("model_type", "Model")} - Training Loss Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No training loss data available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create loss curves plot: {e}")
            raise VisualizationError(f"Failed to create loss curves plot: {e}")
    
    def create_accuracy_curves_plot(self, model_metrics: Dict[str, Any], save_path: Path) -> Path:
        """
        Create training/validation accuracy curves plot
        
        Args:
            model_metrics: Model training metrics
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            curves = self.extract_training_curves(model_metrics)
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            if curves['train_accuracy'] and curves['val_accuracy']:
                epochs = curves['epochs'] or range(1, len(curves['train_accuracy']) + 1)
                
                ax.plot(epochs, curves['train_accuracy'], label='Training Accuracy', linewidth=2)
                ax.plot(epochs, curves['val_accuracy'], label='Validation Accuracy', linewidth=2)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model_metrics.get("model_type", "Model")} - Training Accuracy Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0)
            else:
                ax.text(0.5, 0.5, 'No training accuracy data available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create accuracy curves plot: {e}")
            raise VisualizationError(f"Failed to create accuracy curves plot: {e}")
    
    def create_learning_rate_schedule_plot(self, model_metrics: Dict[str, Any], save_path: Path) -> Path:
        """
        Create learning rate schedule visualization
        
        Args:
            model_metrics: Model training metrics
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            curves = self.extract_training_curves(model_metrics)
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            if curves['learning_rate']:
                epochs = curves['epochs'] or range(1, len(curves['learning_rate']) + 1)
                
                ax.plot(epochs, curves['learning_rate'], linewidth=2, color='orange')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.set_title(f'{model_metrics.get("model_type", "Model")} - Learning Rate Schedule')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')  # Log scale for learning rate
            else:
                # Show constant learning rate if no schedule data
                lr = model_metrics.get('config', {}).get('learning_rate', 0.001)
                ax.axhline(y=lr, color='orange', linewidth=2, label=f'Constant LR: {lr}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.set_title(f'{model_metrics.get("model_type", "Model")} - Learning Rate Schedule')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create learning rate schedule plot: {e}")
            raise VisualizationError(f"Failed to create learning rate schedule plot: {e}")
    
    def create_model_comparison_plot(self, training_metrics: Dict[str, Dict[str, Any]], save_path: Path) -> Path:
        """
        Create model comparison visualization
        
        Args:
            training_metrics: All model training metrics
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            models = list(training_metrics.keys())
            accuracies = [metrics.get('model_accuracy', 0) for metrics in training_metrics.values()]
            training_times = [metrics.get('training_duration_seconds', 0) for metrics in training_metrics.values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Model accuracy comparison
            bars1 = ax1.bar(models, accuracies, alpha=0.7)
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylim(0, 1.0)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # Training time comparison
            bars2 = ax2.bar(models, [t/60 for t in training_times], alpha=0.7, color='orange')  # Convert to minutes
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Training Time (minutes)')
            ax2.set_title('Training Time Comparison')
            
            # Add value labels on bars
            for bar, time_sec in zip(bars2, training_times):
                time_min = time_sec / 60
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)/60 * 0.01,
                        f'{time_min:.1f}m', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create model comparison plot: {e}")
            raise VisualizationError(f"Failed to create model comparison plot: {e}")
    
    def create_resource_usage_plot(self, training_metrics: Dict[str, Dict[str, Any]], save_path: Path) -> Path:
        """
        Create resource usage and training statistics visualization
        
        Args:
            training_metrics: All model training metrics
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            models = list(training_metrics.keys())
            training_times = [metrics.get('training_duration_seconds', 0) for metrics in training_metrics.values()]
            sample_counts = [metrics.get('training_samples', 0) for metrics in training_metrics.values()]
            feature_counts = [metrics.get('feature_count', 0) for metrics in training_metrics.values()]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training time
            ax1.bar(models, [t/60 for t in training_times], alpha=0.7)
            ax1.set_ylabel('Training Time (minutes)')
            ax1.set_title('Training Duration by Model')
            ax1.tick_params(axis='x', rotation=45)
            
            # Training samples
            ax2.bar(models, sample_counts, alpha=0.7, color='green')
            ax2.set_ylabel('Training Samples')
            ax2.set_title('Training Data Size')
            ax2.tick_params(axis='x', rotation=45)
            
            # Feature count
            ax3.bar(models, feature_counts, alpha=0.7, color='purple')
            ax3.set_ylabel('Feature Count')
            ax3.set_title('Number of Input Features')
            ax3.tick_params(axis='x', rotation=45)
            
            # Throughput (samples per second)
            throughput = [samples / time_sec if time_sec > 0 else 0 
                         for samples, time_sec in zip(sample_counts, training_times)]
            ax4.bar(models, throughput, alpha=0.7, color='red')
            ax4.set_ylabel('Samples/Second')
            ax4.set_title('Training Throughput')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create resource usage plot: {e}")
            raise VisualizationError(f"Failed to create resource usage plot: {e}")
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, Any], save_path: Path) -> Path:
        """
        Create feature importance visualization
        
        Args:
            feature_importance: Dictionary with feature names and importance scores
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            feature_names = feature_importance['feature_names']
            importance_scores = feature_importance['importance_scores']
            method = feature_importance.get('method', 'unknown')
            
            # Sort by importance
            sorted_data = sorted(zip(feature_names, importance_scores), 
                               key=lambda x: x[1], reverse=True)
            sorted_names, sorted_scores = zip(*sorted_data)
            
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Create horizontal bar chart
            y_pos = np.arange(len(sorted_names))
            bars = ax.barh(y_pos, sorted_scores, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance ({method})')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                ax.text(bar.get_width() + max(sorted_scores) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create feature importance plot: {e}")
            raise VisualizationError(f"Failed to create feature importance plot: {e}")
    
    def calculate_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """
        Calculate feature importance from model
        
        Args:
            model: Trained model with feature importance
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance data
        """
        try:
            if hasattr(model, 'feature_importance_'):
                importance_scores = model.feature_importance_.tolist()
                method = 'model_intrinsic'
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_).flatten().tolist()
                method = 'linear_coefficients'
            else:
                # Generate mock importance for demo
                importance_scores = np.random.exponential(0.1, len(feature_names)).tolist()
                method = 'mock_random'
            
            return {
                'feature_names': feature_names,
                'importance_scores': importance_scores,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            # Return mock data on failure
            return {
                'feature_names': feature_names,
                'importance_scores': [0.1] * len(feature_names),
                'method': 'fallback'
            }
    
    def create_model_architecture_summary(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create model architecture summary
        
        Args:
            model_metrics: Model training metrics
            
        Returns:
            Dictionary with architecture summary
        """
        model_type = model_metrics.get('model_type', 'unknown')
        config = model_metrics.get('config', {})
        
        # Extract key architecture parameters based on model type
        if model_type == 'lstm':
            architecture = {
                'type': 'LSTM',
                'layers': config.get('num_layers', 'unknown'),
                'hidden_size': config.get('hidden_size', 'unknown'),
                'sequence_length': config.get('sequence_length', 'unknown'),
                'dropout': config.get('dropout', 'unknown')
            }
        elif model_type in ['transformer', 'itransformer']:
            architecture = {
                'type': 'Transformer',
                'layers': config.get('n_layers', 'unknown'),
                'model_dim': config.get('d_model', 'unknown'),
                'attention_heads': config.get('n_heads', 'unknown'),
                'sequence_length': config.get('sequence_length', 'unknown')
            }
        else:
            architecture = {
                'type': model_type,
                'config': str(config)
            }
        
        # Estimate parameter count (rough approximation)
        parameter_count = self._estimate_parameter_count(model_type, config)
        
        return {
            'model_type': model_type,
            'architecture': architecture,
            'parameters': parameter_count,
            'training_samples': model_metrics.get('training_samples', 0),
            'feature_count': model_metrics.get('feature_count', 0)
        }
    
    def _estimate_parameter_count(self, model_type: str, config: Dict[str, Any]) -> str:
        """Estimate model parameter count"""
        try:
            if model_type == 'lstm':
                hidden_size = config.get('hidden_size', 128)
                num_layers = config.get('num_layers', 2)
                # Rough LSTM parameter estimation
                params = num_layers * hidden_size * hidden_size * 4  # Approximate
                return f"~{params:,}"
            elif model_type in ['transformer', 'itransformer']:
                d_model = config.get('d_model', 256)
                n_layers = config.get('n_layers', 4)
                # Rough Transformer parameter estimation
                params = n_layers * d_model * d_model * 4  # Very rough approximation
                return f"~{params:,}"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def create_hyperparameter_table(self, training_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create hyperparameter comparison table
        
        Args:
            training_metrics: All model training metrics
            
        Returns:
            DataFrame with hyperparameter comparison
        """
        rows = []
        
        for model_name, metrics in training_metrics.items():
            config = metrics.get('config', {})
            row = {'Model': model_name}
            
            # Common hyperparameters
            common_params = ['learning_rate', 'batch_size', 'num_epochs']
            for param in common_params:
                row[param] = config.get(param, 'N/A')
            
            # Model-specific parameters
            if model_name == 'lstm':
                row['hidden_size'] = config.get('hidden_size', 'N/A')
                row['num_layers'] = config.get('num_layers', 'N/A')
                row['sequence_length'] = config.get('sequence_length', 'N/A')
            elif model_name in ['transformer', 'itransformer']:
                row['d_model'] = config.get('d_model', 'N/A')
                row['n_heads'] = config.get('n_heads', 'N/A')
                row['n_layers'] = config.get('n_layers', 'N/A')
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_data_distribution_plots(self, data: pd.DataFrame, save_path: Path) -> Path:
        """
        Create data distribution summary plots
        
        Args:
            data: Training data DataFrame
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            # Select numeric columns for distribution analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 6:
                numeric_cols = numeric_cols[:6]  # Limit to 6 for plotting
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Create histogram with density curve
                    data[col].hist(ax=ax, bins=30, alpha=0.7, density=True)
                    
                    # Add density curve
                    try:
                        data[col].plot.density(ax=ax, color='red', linewidth=2)
                    except Exception:
                        pass  # Skip density if it fails
                    
                    ax.set_title(f'{col} Distribution')
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create data distribution plots: {e}")
            raise VisualizationError(f"Failed to create data distribution plots: {e}")
    
    def calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data statistics
        
        Args:
            data: Training data DataFrame
            
        Returns:
            Dictionary with data statistics
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Basic summary statistics
            summary_stats = numeric_data.describe()
            
            # Missing values analysis
            missing_values = data.isnull().sum()
            missing_percentage = (missing_values / len(data)) * 100
            
            # Data quality score (simple metric)
            data_quality_score = 1.0 - (missing_values.sum() / (len(data) * len(data.columns)))
            
            return {
                'summary_stats': summary_stats,
                'missing_values': missing_values.to_dict(),
                'missing_percentage': missing_percentage.to_dict(),
                'data_quality_score': max(0.0, data_quality_score),
                'total_samples': len(data),
                'total_features': len(data.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate data statistics: {e}")
            return {
                'summary_stats': pd.DataFrame(),
                'missing_values': {},
                'missing_percentage': {},
                'data_quality_score': 0.0,
                'total_samples': 0,
                'total_features': 0
            }
    
    def create_baseline_comparison(self, training_metrics: Dict[str, Dict[str, Any]], 
                                 baseline_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create baseline model comparison
        
        Args:
            training_metrics: Trained model metrics
            baseline_metrics: Baseline model metrics
            
        Returns:
            Dictionary with comparison results
        """
        models = {}
        baselines = {}
        improvements = {}
        
        for name, metrics in training_metrics.items():
            models[name] = metrics.get('model_accuracy', 0.0)
        
        for name, metrics in baseline_metrics.items():
            baselines[name] = metrics.get('accuracy', 0.0)
        
        # Calculate improvements over best baseline
        best_baseline = max(baselines.values()) if baselines else 0.5
        
        for name, accuracy in models.items():
            improvement = ((accuracy - best_baseline) / best_baseline) * 100 if best_baseline > 0 else 0
            improvements[name] = improvement
        
        return {
            'models': models,
            'baselines': baselines,
            'improvement': improvements,
            'best_baseline': best_baseline
        }
    
    def generate_pdf_report(self, 
                          training_metrics: Dict[str, Dict[str, Any]], 
                          report_title: str, 
                          save_path: Path,
                          feature_importance: Optional[Dict[str, Any]] = None,
                          training_data: Optional[pd.DataFrame] = None) -> Path:
        """
        Generate comprehensive PDF training report
        
        Args:
            training_metrics: All model training metrics
            report_title: Title for the report
            save_path: Path to save the PDF
            feature_importance: Optional feature importance data
            training_data: Optional training data for distribution analysis
            
        Returns:
            Path to generated PDF
        """
        try:
            if not training_metrics:
                raise ValueError("No training metrics provided")
            
            self.validate_metrics(training_metrics)
            
            # Create temporary directory for plots
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate all plots
                plot_paths = {}
                
                # Model comparison plot
                plot_paths['comparison'] = self.create_model_comparison_plot(
                    training_metrics, temp_path / "model_comparison.png"
                )
                
                # Resource usage plot
                plot_paths['resources'] = self.create_resource_usage_plot(
                    training_metrics, temp_path / "resource_usage.png"
                )
                
                # Individual model plots
                for model_name, metrics in training_metrics.items():
                    if metrics.get('success'):
                        plot_paths[f'{model_name}_loss'] = self.create_loss_curves_plot(
                            metrics, temp_path / f"{model_name}_loss.png"
                        )
                        plot_paths[f'{model_name}_accuracy'] = self.create_accuracy_curves_plot(
                            metrics, temp_path / f"{model_name}_accuracy.png"
                        )
                        plot_paths[f'{model_name}_lr'] = self.create_learning_rate_schedule_plot(
                            metrics, temp_path / f"{model_name}_lr.png"
                        )
                
                # Feature importance plot if available
                if feature_importance:
                    plot_paths['feature_importance'] = self.create_feature_importance_plot(
                        feature_importance, temp_path / "feature_importance.png"
                    )
                
                # Data distribution plots if available
                if training_data is not None:
                    plot_paths['data_distribution'] = self.create_data_distribution_plots(
                        training_data, temp_path / "data_distribution.png"
                    )
                
                # Generate PDF with all sections
                pdf_path = self._create_pdf_document(
                    training_metrics, report_title, save_path, plot_paths, 
                    feature_importance, training_data
                )
                
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise PDFGenerationError(f"Failed to generate PDF report: {e}")
    
    def _create_pdf_document(self, 
                           training_metrics: Dict[str, Dict[str, Any]], 
                           report_title: str, 
                           save_path: Path,
                           plot_paths: Dict[str, Path],
                           feature_importance: Optional[Dict[str, Any]] = None,
                           training_data: Optional[pd.DataFrame] = None) -> Path:
        """Create the actual PDF document"""
        try:
            doc = SimpleDocTemplate(str(save_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=20,
                alignment=TA_CENTER,
                spaceAfter=30
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            )
            
            # Add sections based on configuration
            sections = self.config.get('include_sections', [])
            
            if 'title_page' in sections:
                self.add_title_page(story, report_title, training_metrics, title_style, styles)
            
            if 'executive_summary' in sections:
                self.add_executive_summary(story, training_metrics, heading_style, styles)
            
            if 'model_comparison' in sections and 'comparison' in plot_paths:
                self.add_model_comparison(story, plot_paths['comparison'], heading_style, styles)
            
            if 'training_curves' in sections:
                self.add_training_curves(story, training_metrics, plot_paths, heading_style, styles)
            
            if 'hyperparameters' in sections:
                self.add_hyperparameter_tables(story, training_metrics, heading_style, styles)
            
            if 'resource_usage' in sections and 'resources' in plot_paths:
                self.add_resource_usage(story, plot_paths['resources'], heading_style, styles)
            
            if 'feature_importance' in sections and 'feature_importance' in plot_paths:
                self.add_feature_importance(story, plot_paths['feature_importance'], 
                                          feature_importance, heading_style, styles)
            
            if 'data_distribution' in sections and training_data is not None:
                self.add_data_distribution(story, plot_paths.get('data_distribution'), 
                                         training_data, heading_style, styles)
            
            # Build PDF
            doc.build(story)
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create PDF document: {e}")
            raise PDFGenerationError(f"Failed to create PDF document: {e}")
    
    def add_title_page(self, story: List, title: str, training_metrics: Dict[str, Dict[str, Any]],
                      title_style: ParagraphStyle, styles: Dict):
        """Add title page to PDF"""
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        metadata = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Trained: {len(training_metrics)}",
            f"Company: {self.config.get('company_name', 'ShyvR AI Trading')}",
            f"Author: {self.config.get('report_author', 'ML Training System')}"
        ]
        
        for item in metadata:
            story.append(Paragraph(item, styles['Normal']))
        
        story.append(PageBreak())
    
    def add_executive_summary(self, story: List, training_metrics: Dict[str, Dict[str, Any]],
                            heading_style: ParagraphStyle, styles: Dict):
        """Add executive summary section"""
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Extract key metrics
        best_model = max(training_metrics.items(), 
                        key=lambda x: x[1].get('model_accuracy', 0))
        
        total_time = sum(m.get('training_duration_seconds', 0) for m in training_metrics.values())
        
        summary_text = f"""
        This report summarizes the training results for {len(training_metrics)} machine learning models.
        The best performing model was {best_model[0]} with an accuracy of {best_model[1].get('model_accuracy', 0):.3f}.
        Total training time was {total_time/60:.1f} minutes across all models.
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    def add_model_comparison(self, story: List, plot_path: Path, 
                           heading_style: ParagraphStyle, styles: Dict):
        """Add model comparison section"""
        story.append(Paragraph("Model Comparison", heading_style))
        story.append(Paragraph("Performance and training time comparison across all models:", styles['Normal']))
        
        if plot_path.exists():
            img = Image(str(plot_path), width=6*inch, height=3*inch)
            story.append(img)
        
        story.append(Spacer(1, 0.3*inch))
    
    def add_training_curves(self, story: List, training_metrics: Dict[str, Dict[str, Any]],
                          plot_paths: Dict[str, Path], heading_style: ParagraphStyle, styles: Dict):
        """Add training curves section"""
        story.append(Paragraph("Training Curves", heading_style))
        
        for model_name, metrics in training_metrics.items():
            if not metrics.get('success'):
                continue
                
            story.append(Paragraph(f"{model_name.upper()} Model", styles['Heading2']))
            
            # Add loss and accuracy plots
            loss_key = f'{model_name}_loss'
            acc_key = f'{model_name}_accuracy'
            
            if loss_key in plot_paths and plot_paths[loss_key].exists():
                story.append(Paragraph("Loss Curves:", styles['Normal']))
                img = Image(str(plot_paths[loss_key]), width=5*inch, height=3*inch)
                story.append(img)
            
            if acc_key in plot_paths and plot_paths[acc_key].exists():
                story.append(Paragraph("Accuracy Curves:", styles['Normal']))
                img = Image(str(plot_paths[acc_key]), width=5*inch, height=3*inch)
                story.append(img)
            
            story.append(Spacer(1, 0.2*inch))
    
    def add_hyperparameter_tables(self, story: List, training_metrics: Dict[str, Dict[str, Any]],
                                heading_style: ParagraphStyle, styles: Dict):
        """Add hyperparameter tables section"""
        story.append(Paragraph("Hyperparameters", heading_style))
        
        # Create hyperparameter table
        df = self.create_hyperparameter_table(training_metrics)
        
        # Convert DataFrame to reportlab table
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
    
    def add_resource_usage(self, story: List, plot_path: Path,
                         heading_style: ParagraphStyle, styles: Dict):
        """Add resource usage section"""
        story.append(Paragraph("Resource Usage", heading_style))
        story.append(Paragraph("Training time, data usage, and throughput statistics:", styles['Normal']))
        
        if plot_path.exists():
            img = Image(str(plot_path), width=6*inch, height=4*inch)
            story.append(img)
        
        story.append(Spacer(1, 0.3*inch))
    
    def add_feature_importance(self, story: List, plot_path: Optional[Path],
                             feature_importance: Optional[Dict[str, Any]],
                             heading_style: ParagraphStyle, styles: Dict):
        """Add feature importance section"""
        story.append(Paragraph("Feature Importance", heading_style))
        
        if feature_importance:
            method = feature_importance.get('method', 'unknown')
            story.append(Paragraph(f"Feature importance calculated using: {method}", styles['Normal']))
        
        if plot_path and plot_path.exists():
            img = Image(str(plot_path), width=6*inch, height=4*inch)
            story.append(img)
        
        story.append(Spacer(1, 0.3*inch))
    
    def add_data_distribution(self, story: List, plot_path: Optional[Path],
                            training_data: pd.DataFrame, heading_style: ParagraphStyle, styles: Dict):
        """Add data distribution section"""
        story.append(Paragraph("Data Distribution Analysis", heading_style))
        
        # Add data statistics
        stats = self.calculate_data_statistics(training_data)
        
        stats_text = f"""
        Training data contains {stats['total_samples']:,} samples with {stats['total_features']} features.
        Data quality score: {stats['data_quality_score']:.3f}
        """
        
        story.append(Paragraph(stats_text, styles['Normal']))
        
        if plot_path and plot_path.exists():
            img = Image(str(plot_path), width=6*inch, height=4*inch)
            story.append(img)
        
        story.append(Spacer(1, 0.3*inch))
    
    def generate_html_report(self,
                           training_metrics: Dict[str, Dict[str, Any]],
                           report_title: str,
                           save_path: Path,
                           feature_importance: Optional[Dict[str, Any]] = None,
                           training_data: Optional[pd.DataFrame] = None) -> Path:
        """
        Generate comprehensive HTML training report
        
        Args:
            training_metrics: All model training metrics
            report_title: Title for the report
            save_path: Path to save HTML file
            feature_importance: Optional feature importance data
            training_data: Optional training data for analysis
            
        Returns:
            Path to generated HTML report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate plots
            plots = {}
            for model_name, metrics in training_metrics.items():
                if metrics.get('success', False):
                    # Generate accuracy comparison plot
                    plot_path = self.create_model_accuracy_plot([{
                        'model': model_name,
                        'accuracy': metrics.get('model_accuracy', 0)
                    }])
                    if plot_path:
                        plots[model_name] = self._encode_image_base64(plot_path)
            
            # Create HTML content
            html_content = self._generate_html_content(
                report_title=report_title,
                training_metrics=training_metrics,
                plots=plots,
                timestamp=timestamp,
                feature_importance=feature_importance,
                training_data=training_data
            )
            
            # Save HTML file
            save_path = save_path.with_suffix('.html')
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise
    
    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode image as base64 for embedding in HTML"""
        import base64
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _generate_html_content(self,
                              report_title: str,
                              training_metrics: Dict[str, Dict[str, Any]],
                              plots: Dict[str, str],
                              timestamp: str,
                              feature_importance: Optional[Dict[str, Any]] = None,
                              training_data: Optional[pd.DataFrame] = None) -> str:
        """Generate HTML content with embedded CSS and charts"""
        
        # Calculate summary statistics
        successful_models = [m for m in training_metrics.values() if m.get('success', False)]
        total_training_time = sum(m.get('training_duration_seconds', 0) for m in successful_models)
        
        # Find best model
        best_model = None
        best_accuracy = 0
        for model_name, metrics in training_metrics.items():
            if metrics.get('model_accuracy', 0) > best_accuracy:
                best_accuracy = metrics.get('model_accuracy', 0)
                best_model = model_name
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #667eea;
                    padding-left: 10px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 1.2em;
                }}
                .summary-card .value {{
                    font-size: 2em;
                    font-weight: bold;
                }}
                .model-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .model-table th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                .model-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                .model-table tr:hover {{
                    background: #f5f5f5;
                }}
                .success {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .failed {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .plot-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .plot-container img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    text-align: center;
                    color: #7f8c8d;
                }}
                .timestamp {{
                    color: #95a5a6;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{report_title}</h1>
                <p class="timestamp">Generated: {timestamp}</p>
                
                <h2>📊 Training Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Total Models</h3>
                        <div class="value">{len(training_metrics)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Successful</h3>
                        <div class="value">{len(successful_models)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Best Accuracy</h3>
                        <div class="value">{best_accuracy:.2%}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Training Time</h3>
                        <div class="value">{total_training_time/60:.1f} min</div>
                    </div>
                </div>
                
                <h2>🤖 Model Performance</h2>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Accuracy (R²)</th>
                            <th>Training Time</th>
                            <th>Features</th>
                            <th>Samples</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add model rows
        for model_name, metrics in training_metrics.items():
            status = '<span class="success">✓ Success</span>' if metrics.get('success', False) else '<span class="failed">✗ Failed</span>'
            accuracy = f"{metrics.get('model_accuracy', 0):.2%}" if metrics.get('success', False) else "N/A"
            duration = f"{metrics.get('training_duration_seconds', 0):.1f}s" if metrics.get('training_duration_seconds') else "N/A"
            features = metrics.get('feature_count', 'N/A')
            samples = metrics.get('training_samples', 'N/A')
            
            # Highlight best model
            row_style = ' style="background: #f0f9ff;"' if model_name == best_model else ''
            
            html += f"""
                        <tr{row_style}>
                            <td><strong>{model_name}</strong></td>
                            <td>{status}</td>
                            <td>{accuracy}</td>
                            <td>{duration}</td>
                            <td>{features}</td>
                            <td>{samples}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
        """
        
        # Add plots if available
        if plots:
            html += "<h2>📈 Visualizations</h2>"
            for model_name, plot_base64 in plots.items():
                html += f"""
                <div class="plot-container">
                    <h3>{model_name} Performance</h3>
                    <img src="data:image/png;base64,{plot_base64}" alt="{model_name} plot">
                </div>
                """
        
        # Add configuration details
        html += """
                <h2>⚙️ Configuration Details</h2>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Key Parameters</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for model_name, metrics in training_metrics.items():
            if metrics.get('success', False) and metrics.get('config'):
                config = metrics['config']
                # Format key parameters
                params = []
                for key, value in config.items():
                    if key in ['sequence_length', 'd_model', 'n_heads', 'n_layers', 'num_epochs', 'batch_size', 'learning_rate']:
                        params.append(f"{key}: {value}")
                
                params_str = ", ".join(params[:5])  # Show first 5 params
                html += f"""
                        <tr>
                            <td><strong>{model_name}</strong></td>
                            <td>{params_str}</td>
                        </tr>
                """
        
        html += """
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>Generated by RLTE Training Pipeline</p>
                    <p>© 2025 Shyvr AI Trading System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def upload_to_gcs(self, local_path: Path, model_type: str, timestamp: str) -> str:
        """
        Upload report to GCS
        
        Args:
            local_path: Path to local file
            model_type: Type of model (for path organization) - DEPRECATED, kept for compatibility
            timestamp: Timestamp for unique path
            
        Returns:
            GCS path of uploaded file
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Report file not found: {local_path}")
        
        if not self.gcs_bucket:
            raise ValueError("GCS bucket not configured")
        
        try:
            # Create model-agnostic GCS path
            # Using training-reports prefix directly instead of the configured prefix
            gcs_path = f"training-reports/training_report_{timestamp}/{local_path.name}"
            
            # Upload file
            blob = self.gcs_bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            
            # Add metadata
            blob.metadata = {
                'report_type': 'training_report',
                'model_type': model_type,
                'generated_at': datetime.now().isoformat(),
                'generator': 'TrainingReportGenerator'
            }
            blob.patch()
            
            full_gcs_path = f"gs://{self.gcs_bucket.name}/{gcs_path}"
            logger.info(f"Report uploaded to GCS: {full_gcs_path}")
            
            return full_gcs_path
            
        except Exception as e:
            logger.error(f"Failed to upload report to GCS: {e}")
            raise
    
    async def generate_reports_for_training_session(self,
                                                  training_metrics: Dict[str, Dict[str, Any]],
                                                  pipeline: Any) -> Dict[str, str]:
        """
        Generate reports for a complete training session
        
        Args:
            training_metrics: All training results
            pipeline: UnifiedTrainingPipeline instance
            
        Returns:
            Dictionary with generated report paths
        """
        try:
            session_id = getattr(pipeline, 'session_id', 'unknown')
            session_timestamp = getattr(pipeline, 'session_timestamp', datetime.now())
            
            timestamp = session_timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Generate comprehensive report
            report_title = f"Training Report - Session {session_id[:8]}"
            html_path = self.output_dir / f"training_report_{timestamp}.html"
            
            # Generate HTML report
            generated_html = self.generate_html_report(
                training_metrics=training_metrics,
                report_title=report_title,
                save_path=html_path
            )
            
            # Upload to GCS if configured
            gcs_path = None
            gcs_image_paths = []
            if self.gcs_bucket:
                # Model type is deprecated but kept for backward compatibility
                # Pass 'session' as placeholder since path is now model-agnostic
                gcs_path = await self.upload_to_gcs(generated_html, 'session', timestamp)
                
                # Also upload any associated images from the report folder
                image_files = list(self.output_dir.glob(f"*_{timestamp}*.png"))
                image_files.extend(list(self.output_dir.glob(f"*_{timestamp}*.jpg")))
                
                for image_file in image_files:
                    image_gcs_path = await self.upload_to_gcs(image_file, 'session', timestamp)
                    gcs_image_paths.append(image_gcs_path)
            
            return {
                'html_report': str(generated_html),
                'pdf_report': str(generated_html),  # Keep for backward compatibility
                'gcs_upload_path': gcs_path,
                'gcs_image_paths': gcs_image_paths,
                'session_id': session_id,
                'timestamp': timestamp,
                'gcs_folder': f"gs://{self.gcs_bucket.name}/training-reports/training_report_{timestamp}/" if gcs_path else None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate reports for training session: {e}")
            raise
    
    def extract_training_session_metadata(self, training_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from training session
        
        Args:
            training_metrics: All training results
            
        Returns:
            Dictionary with session metadata
        """
        try:
            # Find best model
            best_model = max(training_metrics.items(), 
                           key=lambda x: x[1].get('model_accuracy', 0))
            
            # Calculate total training time
            total_training_time = sum(m.get('training_duration_seconds', 0) 
                                    for m in training_metrics.values())
            
            # Count successful models
            successful_models = sum(1 for m in training_metrics.values() 
                                  if m.get('success', False))
            
            # Session summary
            session_summary = {
                'total_models': len(training_metrics),
                'successful_models': successful_models,
                'failed_models': len(training_metrics) - successful_models,
                'total_training_time': total_training_time,
                'average_accuracy': np.mean([m.get('model_accuracy', 0) 
                                           for m in training_metrics.values()])
            }
            
            return {
                'session_summary': session_summary,
                'best_model': {
                    'name': best_model[0],
                    'accuracy': best_model[1].get('model_accuracy', 0),
                    'training_time': best_model[1].get('training_duration_seconds', 0)
                },
                'total_training_time': total_training_time,
                'models_trained': len(training_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract training session metadata: {e}")
            return {
                'session_summary': {},
                'best_model': {},
                'total_training_time': 0,
                'models_trained': 0
            }