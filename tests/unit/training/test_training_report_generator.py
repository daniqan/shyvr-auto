"""
Unit Tests for TrainingReportGenerator

Tests the training report generation functionality including:
- Metrics validation
- Visualization creation
- PDF generation
- GCS upload
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.ml_analysis.training_report_generator import (
    TrainingReportGenerator,
    TrainingReportGeneratorError,
    ValidationError
)


@pytest.fixture
def mock_training_metrics():
    """Mock training metrics for testing"""
    return {
        'lstm': {
            'success': True,
            'model_type': 'lstm',
            'training_duration_seconds': 120.5,
            'model_accuracy': 0.85,
            'feature_count': 65,
            'training_samples': 1000,
            'config': {
                'sequence_length': 50,
                'hidden_size': 128,
                'num_layers': 2,
                'num_epochs': 10
            },
            'trained_at': datetime.now().isoformat(),
            'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.55, 0.45, 0.35, 0.3, 0.28],
            'train_accuracy': [0.6, 0.7, 0.75, 0.8, 0.85],
            'val_accuracy': [0.58, 0.68, 0.73, 0.78, 0.82]
        },
        'transformer': {
            'success': True,
            'model_type': 'transformer',
            'training_duration_seconds': 180.2,
            'model_accuracy': 0.88,
            'feature_count': 65,
            'training_samples': 1000,
            'config': {
                'sequence_length': 192,
                'd_model': 256,
                'n_heads': 8,
                'num_epochs': 10
            },
            'trained_at': datetime.now().isoformat(),
            'train_loss': [0.6, 0.45, 0.35, 0.28, 0.22],
            'val_loss': [0.65, 0.5, 0.4, 0.35, 0.32]
        }
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def report_generator(temp_output_dir):
    """Create TrainingReportGenerator instance"""
    config = {
        'output_dir': str(temp_output_dir),
        'gcs_bucket': 'test-bucket',
        'gcs_prefix': 'test-models'
    }
    return TrainingReportGenerator(config)


class TestTrainingReportGenerator:
    """Test suite for TrainingReportGenerator"""
    
    def test_initialization(self, report_generator):
        """Test report generator initialization"""
        assert report_generator.output_dir.exists()
        assert report_generator.gcs_bucket.name == 'test-bucket'
        assert report_generator.config['gcs_prefix'] == 'test-models'
        assert 'seaborn' in report_generator.config['style']  # Style may include version
    
    def test_validate_metrics_valid(self, report_generator, mock_training_metrics):
        """Test validation of valid training metrics"""
        result = report_generator.validate_metrics(mock_training_metrics)
        assert result is True
    
    def test_validate_metrics_failed_training(self, report_generator):
        """Test validation handles failed training gracefully"""
        failed_metrics = {
            'lstm': {
                'success': False,  # Failed training
                'error': 'Out of memory during training'
            },
            'transformer': {
                'success': True,
                'model_type': 'transformer',
                'training_duration_seconds': 180.2,
                'model_accuracy': 0.88,
                'feature_count': 65,
                'training_samples': 1000,
                'config': {'d_model': 256}
            }
        }
        result = report_generator.validate_metrics(failed_metrics)
        assert result is True  # Should still be valid even with failed models
        assert failed_metrics['lstm']['model_type'] == 'lstm'  # Should infer model_type
    
    def test_extract_training_curves(self, report_generator, mock_training_metrics):
        """Test extraction of training curves from metrics"""
        lstm_metrics = mock_training_metrics['lstm']
        curves = report_generator.extract_training_curves(lstm_metrics)
        
        assert 'train_loss' in curves
        assert 'val_loss' in curves
        assert len(curves['train_loss']) == 5
        assert len(curves['val_loss']) == 5
    
    def test_create_loss_curves_plot(self, report_generator, mock_training_metrics, temp_output_dir):
        """Test creation of loss curves plot"""
        save_path = temp_output_dir / 'loss_curves.png'
        lstm_metrics = mock_training_metrics['lstm']
        
        # Add loss history to metrics
        lstm_metrics['loss_history'] = {
            'train': lstm_metrics['train_loss'],
            'val': lstm_metrics['val_loss']
        }
        
        result_path = report_generator.create_loss_curves_plot(lstm_metrics, save_path)
        assert result_path.exists()
        assert result_path.suffix == '.png'
    
    def test_create_model_comparison_plot(self, report_generator, mock_training_metrics, temp_output_dir):
        """Test creation of model comparison plot"""
        save_path = temp_output_dir / 'model_comparison.png'
        result_path = report_generator.create_model_comparison_plot(mock_training_metrics, save_path)
        assert result_path.exists()
    
    def test_create_resource_usage_plot(self, report_generator, mock_training_metrics, temp_output_dir):
        """Test creation of resource usage plot"""
        save_path = temp_output_dir / 'resource_usage.png'
        result_path = report_generator.create_resource_usage_plot(mock_training_metrics, save_path)
        assert result_path.exists()
    
    def test_calculate_data_statistics(self, report_generator):
        """Test calculation of data statistics"""
        # Create sample data
        data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100) * 1000,
            'rsi_14': np.random.uniform(0, 100, 100)
        })
        
        stats = report_generator.calculate_data_statistics(data)
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert stats['n_samples'] == 100
        assert stats['n_features'] == 3
    
    def test_create_hyperparameter_table(self, report_generator, mock_training_metrics):
        """Test creation of hyperparameter table"""
        df = report_generator.create_hyperparameter_table(mock_training_metrics)
        
        assert isinstance(df, pd.DataFrame)
        assert 'lstm' in df.index
        assert 'transformer' in df.index
        assert 'sequence_length' in df.columns
    
    def test_generate_pdf_report(self, report_generator, mock_training_metrics, temp_output_dir):
        """Test PDF report generation"""
        # Create minimal plots directory
        plots_dir = temp_output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Create a dummy plot file
        dummy_plot = plots_dir / 'model_comparison.png'
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.savefig(dummy_plot)
        plt.close()
        
        pdf_path = report_generator.generate_pdf_report(
            training_metrics=mock_training_metrics,
            plots_dir=plots_dir,
            output_path=temp_output_dir / 'report.pdf'
        )
        
        assert pdf_path.exists()
        assert pdf_path.suffix == '.pdf'
        assert pdf_path.stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_upload_to_gcs_mock(self, report_generator, temp_output_dir):
        """Test GCS upload functionality with mocks"""
        # Create test file
        test_file = temp_output_dir / 'test_report.pdf'
        test_file.write_text('Test report content')
        
        with patch('src.ml_analysis.training_report_generator.storage.Client') as mock_client:
            # Setup mock GCS client
            mock_bucket = Mock()
            mock_blob = Mock()
            mock_client.return_value.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            
            result = await report_generator.upload_to_gcs(
                local_path=test_file,
                model_type='lstm',
                timestamp='20250825_150000'
            )
            
            # Verify upload was called
            mock_blob.upload_from_filename.assert_called_once_with(str(test_file))
            assert 'lstm/20250825_150000/reports' in result
    
    @pytest.mark.asyncio
    async def test_generate_reports_for_training_session(self, report_generator, mock_training_metrics):
        """Test complete report generation for training session"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.session_id = 'test-session-123'
        mock_pipeline.session_timestamp = datetime.now()
        
        with patch.object(report_generator, 'generate_pdf_report') as mock_pdf:
            with patch.object(report_generator, 'upload_to_gcs') as mock_upload:
                mock_pdf.return_value = Path('/tmp/test_report.pdf')
                mock_upload.return_value = 'gs://test-bucket/reports/test_report.pdf'
                
                results = await report_generator.generate_reports_for_training_session(
                    training_metrics=mock_training_metrics,
                    pipeline=mock_pipeline
                )
                
                assert 'pdf_report' in results
                assert 'gcs_upload_path' in results
                assert results['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])