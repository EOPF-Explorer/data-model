"""
Main S2 optimization converter.
"""

class S2OptimizedConverter:
    """Optimized Sentinel-2 to GeoZarr converter."""
    
    def __init__(self, enable_sharding=True, spatial_chunk=1024):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        
    def convert_s2(self, dt_input, output_path, **kwargs):
        """Main conversion entry point."""
        from .s2_data_consolidator import S2DataConsolidator
        from .s2_multiscale import S2MultiscalePyramid
        from .s2_validation import S2OptimizationValidator

        # Consolidate data
        consolidator = S2DataConsolidator(dt_input)
        measurements, geometry, meteorology = consolidator.consolidate_all_data()

        # Create multiscale pyramids
        pyramid = S2MultiscalePyramid(
            enable_sharding=self.enable_sharding,
            spatial_chunk=self.spatial_chunk
        )
        multiscale_data = pyramid.create_multiscale_measurements(measurements, output_path)

        # Validate the output
        validator = S2OptimizationValidator()
        validation_results = validator.validate_optimized_dataset(output_path)

        return {
            "multiscale_data": multiscale_data,
            "validation_results": validation_results
        }
