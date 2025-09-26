"""
Unit tests for S2OptimizedConverter.
"""

import pytest
import xarray as xr
from xarray import DataTree

from eopf_geozarr.s2_optimization.s2_converter import S2OptimizedConverter

@pytest.fixture
def mock_input_data():
    """Create mock input DataTree for testing."""
    # Placeholder for creating mock DataTree
    return DataTree()

def test_conversion_pipeline(mock_input_data, tmp_path):
    """Test the full conversion pipeline."""
    output_path = tmp_path / "optimized_output"
    converter = S2OptimizedConverter(enable_sharding=True, spatial_chunk=1024)
    
    result = converter.convert_s2(mock_input_data, str(output_path))
    
    # Validate multiscale data
    assert "multiscale_data" in result
    assert isinstance(result["multiscale_data"], dict)
    
    # Validate output path
    assert output_path.exists()
    
    # Validate validation results
    assert "validation_results" in result
    assert result["validation_results"]["is_valid"]
