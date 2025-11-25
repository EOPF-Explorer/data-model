"""
Downsampling operations for Sentinel-2 data (no upsampling).
"""

import numpy as np
import xarray as xr


class S2ResamplingEngine:
    """Handles downsampling operations for S2 multiscale creation."""

    def __init__(self) -> None:
        self.resampling_methods = {
            "reflectance": self._downsample_reflectance,
            "classification": self._downsample_classification,
            "quality_mask": self._downsample_quality_mask,
            "probability": self._downsample_probability,
            "detector_footprint": self._downsample_quality_mask,  # Same as quality mask
        }

    def downsample_variable(
        self, data: xr.DataArray, target_height: int, target_width: int, var_type: str
    ) -> xr.DataArray:
        """
        Downsample a variable to target dimensions.

        Args:
            data: Input data array
            target_height: Target height in pixels
            target_width: Target width in pixels
            var_type: Type of variable ('reflectance', 'classification', etc.)

        Returns:
            Downsampled data array
        """
        if var_type not in self.resampling_methods:
            raise ValueError(f"Unknown variable type: {var_type}")

        method = self.resampling_methods[var_type]
        return method(data, target_height, target_width)

    def _downsample_reflectance(
        self, data: xr.DataArray, target_height: int, target_width: int
    ) -> xr.DataArray:
        """Block averaging for reflectance bands."""
        # Calculate block sizes
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width

        # Ensure exact divisibility
        if current_height % target_height != 0 or current_width % target_width != 0:
            # Crop to make it divisible
            new_height = (current_height // block_h) * block_h
            new_width = (current_width // block_w) * block_w
            data = data[..., :new_height, :new_width]

        # Perform block averaging
        if data.ndim == 3:  # (time, y, x) or similar
            reshaped = data.values.reshape(
                data.shape[0], target_height, block_h, target_width, block_w
            )
            downsampled = reshaped.mean(axis=(2, 4))
        else:  # (y, x)
            reshaped = data.values.reshape(
                target_height, block_h, target_width, block_w
            )
            downsampled = reshaped.mean(axis=(1, 3))

        # Create new coordinates
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]

        # Create new DataArray
        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords,
            }
        else:
            coords = {data.dims[-2]: y_coords, data.dims[-1]: x_coords}

        return xr.DataArray(
            downsampled, dims=data.dims, coords=coords, attrs=data.attrs.copy()
        )

    def _downsample_classification(
        self, data: xr.DataArray, target_height: int, target_width: int
    ) -> xr.DataArray:
        """Fast nearest neighbor downsampling for classification data."""
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width

        # Crop to make divisible
        new_height = (current_height // block_h) * block_h
        new_width = (current_width // block_w) * block_w
        data = data[..., :new_height, :new_width]

        # Use simple nearest neighbor sampling (much faster than mode)
        # Take the center pixel of each block as representative
        center_h = block_h // 2
        center_w = block_w // 2
        if data.ndim == 3:
            # Sample every block_h and block_w pixels, starting from center
            downsampled = data.values[:, center_h::block_h, center_w::block_w]
            # Ensure we get exactly the target dimensions
            downsampled = downsampled[:, :target_height, :target_width]
        else:
            # Sample every block_h and block_w pixels, starting from center
            downsampled = data.values[center_h::block_h, center_w::block_w]
            # Ensure we get exactly the target dimensions
            downsampled = downsampled[:target_height, :target_width]

        # Create coordinates (use same sampling as other methods for consistency)
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]

        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords,
            }
        else:
            coords = {data.dims[-2]: y_coords, data.dims[-1]: x_coords}

        return xr.DataArray(
            downsampled, dims=data.dims, coords=coords, attrs=data.attrs.copy()
        )

    def _downsample_quality_mask(
        self, data: xr.DataArray, target_height: int, target_width: int
    ) -> xr.DataArray:
        """Logical OR downsampling for quality masks (any bad pixel = bad block)."""
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width

        # Crop to make divisible
        new_height = (current_height // block_h) * block_h
        new_width = (current_width // block_w) * block_w
        data = data[..., :new_height, :new_width]

        if data.ndim == 3:
            reshaped = data.values.reshape(
                data.shape[0], target_height, block_h, target_width, block_w
            )
            # Any non-zero value in block makes the downsampled pixel non-zero
            downsampled = (reshaped.sum(axis=(2, 4)) > 0).astype(data.dtype)
        else:
            reshaped = data.values.reshape(
                target_height, block_h, target_width, block_w
            )
            downsampled = (reshaped.sum(axis=(1, 3)) > 0).astype(data.dtype)

        # Create coordinates
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]

        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords,
            }
        else:
            coords = {data.dims[-2]: y_coords, data.dims[-1]: x_coords}

        return xr.DataArray(
            downsampled, dims=data.dims, coords=coords, attrs=data.attrs.copy()
        )

    def _downsample_probability(
        self, data: xr.DataArray, target_height: int, target_width: int
    ) -> xr.DataArray:
        """Average downsampling for probability data."""
        # Use same method as reflectance but ensure values stay in [0,1] or [0,100] range
        result = self._downsample_reflectance(data, target_height, target_width)

        # Clamp values to valid probability range
        if result.max() <= 1.0:  # [0,1] probabilities
            result.values = np.clip(result.values, 0, 1)
        else:  # [0,100] percentages
            result.values = np.clip(result.values, 0, 100)

        return result


def determine_variable_type(var_name: str, var_data: xr.DataArray) -> str:
    """
    Determine the type of a variable for appropriate resampling.

    Args:
        var_name: Name of the variable
        var_data: The data array

    Returns:
        Variable type string
    """
    # Spectral bands
    if var_name.startswith("b") and (var_name[1:].isdigit() or var_name == "b8a"):
        return "reflectance"

    # Quality data
    if var_name in ["scl"]:  # Scene Classification Layer
        return "classification"

    if var_name in ["cld", "snw"]:  # Probability data
        return "probability"

    if var_name in ["aot", "wvp"]:  # Atmosphere quality - treat as reflectance
        return "reflectance"

    if var_name.startswith("detector_footprint_") or var_name.startswith("quality_"):
        return "quality_mask"

    # Default to reflectance for unknown variables
    return "reflectance"
