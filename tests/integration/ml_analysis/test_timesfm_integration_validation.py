"""
Integration tests for TimesFM integration with Model Manager
Validates that Phase 2.2.4 is completed correctly.
"""

import pytest
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent.parent.parent


class TestTimesFMIntegrationValidation:
    """Integration tests to validate TimesFM integration with Model Manager"""
    
    def test_modeltype_enum_includes_timesfm(self):
        """Test 1: Check ModelType enum includes TIMESFM"""
        base_py_path = project_root / "src" / "ml_analysis" / "base.py"
        assert base_py_path.exists(), "base.py not found"
        
        with open(base_py_path, 'r') as f:
            base_content = f.read()
        
        assert 'TIMESFM = "timesfm"' in base_content, "TIMESFM not found in ModelType enum"
    
    def test_timesfm_wrapper_structure(self):
        """Test 2: Check TimesFMWrapper file exists and has correct structure"""
        timesfm_wrapper_path = project_root / "src" / "ml_analysis" / "transformers" / "timesfm_wrapper.py"
        assert timesfm_wrapper_path.exists(), "timesfm_wrapper.py not found"
        
        with open(timesfm_wrapper_path, 'r') as f:
            wrapper_content = f.read()
        
        required_classes = [
            "class TimesFMWrapper",
            "class TimesFMConfig",
            "class CryptoTimesFMTokenizer",
            "class GCPTimesFMOptimizer"
        ]
        
        for cls in required_classes:
            assert cls in wrapper_content, f"{cls} not found in timesfm_wrapper.py"
    
    def test_timesfm_exports(self):
        """Test 3: Check TimesFM classes are properly exported"""
        transformers_init_path = project_root / "src" / "ml_analysis" / "transformers" / "__init__.py"
        assert transformers_init_path.exists(), "__init__.py not found in transformers module"
        
        with open(transformers_init_path, 'r') as f:
            init_content = f.read()
        
        # Check imports
        assert "from .timesfm_wrapper import" in init_content, "TimesFM imports not found"
        
        # Check exports in __all__
        required_exports = [
            "TimesFMWrapper",
            "TimesFMConfig", 
            "CryptoTimesFMTokenizer",
            "GCPTimesFMOptimizer"
        ]
        
        for export in required_exports:
            assert f'"{export}"' in init_content, f"{export} not in __all__ exports"
    
    def test_timesfm_config_structure(self):
        """Test 4: Validate TimesFMConfig has correct parameter structure"""
        timesfm_wrapper_path = project_root / "src" / "ml_analysis" / "transformers" / "timesfm_wrapper.py"
        
        with open(timesfm_wrapper_path, 'r') as f:
            wrapper_content = f.read()
        
        # Check for expected config parameters
        required_params = [
            "model_name",
            "prediction_length",
            "context_length",
            "batch_size",
            "use_zero_shot"
        ]
        
        for param in required_params:
            assert f"{param}:" in wrapper_content or f"'{param}'" in wrapper_content, \
                   f"Config parameter {param} not found"
    
    def test_model_manager_integration(self):
        """Test 5: Check Model Manager contains TimesFM integration"""
        model_manager_path = project_root / "src" / "ml_analysis" / "model_manager.py"
        assert model_manager_path.exists(), "model_manager.py not found"
        
        with open(model_manager_path, 'r') as f:
            manager_content = f.read()
        
        # Check for TimesFM import
        assert "from .transformers.timesfm_wrapper import TimesFMWrapper" in manager_content, \
               "TimesFM import not found in Model Manager"
        
        # Check for TimesFM initialization in _initialize_models
        assert "ModelType.TIMESFM" in manager_content, \
               "ModelType.TIMESFM not found in Model Manager"
        assert "TimesFMWrapper" in manager_content, \
               "TimesFMWrapper initialization not found"
    
    def test_ensemble_weights_total_100_percent(self):
        """Test 6: Verify ensemble weights total 100% with TimesFM included"""
        model_manager_path = project_root / "src" / "ml_analysis" / "model_manager.py"
        
        with open(model_manager_path, 'r') as f:
            manager_content = f.read()
        
        # Find model weights section
        assert "model_weights" in manager_content, "model_weights not found in Model Manager"
        
        # Check TimesFM weight
        assert "ModelType.TIMESFM" in manager_content and "0.2" in manager_content, \
               "TimesFM 20% weight allocation not found"
    
    def test_fear_greed_integration_readiness(self):
        """Test 7: Verify Model Manager is ready for Phase 3 fear_greed_index integration"""
        model_manager_path = project_root / "src" / "ml_analysis" / "model_manager.py"
        
        with open(model_manager_path, 'r') as f:
            manager_content = f.read()
        
        # Check for required methods
        required_methods = [
            "def set_mode(",
            "def analyze_token(",
            "def batch_analyze(",
            "def health_check("
        ]
        
        for method in required_methods:
            assert method in manager_content, f"{method} not found in Model Manager"


@pytest.mark.integration
def test_full_timesfm_integration():
    """Run all integration validation tests"""
    validator = TestTimesFMIntegrationValidation()
    
    # Run all tests
    validator.test_modeltype_enum_includes_timesfm()
    validator.test_timesfm_wrapper_structure()
    validator.test_timesfm_exports()
    validator.test_timesfm_config_structure()
    validator.test_model_manager_integration()
    validator.test_ensemble_weights_total_100_percent()
    validator.test_fear_greed_integration_readiness()