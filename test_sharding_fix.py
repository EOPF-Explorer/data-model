#!/usr/bin/env python3
"""
Test script to verify the sharding fix for GeoZarr conversion.
This script tests the _calculate_shard_dimension function and validates
that shard dimensions are properly divisible by chunk dimensions.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_calculate_shard_dimension():
    """Test the _calculate_shard_dimension function."""
    from eopf_geozarr.conversion.geozarr import _calculate_shard_dimension
    
    print("🧪 Testing _calculate_shard_dimension function...")
    
    # Test cases: (data_dim, chunk_dim, description)
    test_cases = [
        (10980, 4096, "Sentinel-2 10m resolution typical case"),
        (5490, 2048, "Half resolution case"),
        (8192, 4096, "Perfect 2x multiple"),
        (12288, 4096, "Perfect 3x multiple"),
        (16384, 4096, "Perfect 4x multiple"),
        (20480, 4096, "Perfect 5x multiple"),
        (24576, 4096, "Perfect 6x multiple"),
        (1000, 512, "Small dimension case"),
        (256, 512, "Chunk larger than data"),
        (1024, 256, "4x multiple case"),
    ]
    
    print("\nTest Results:")
    print("=" * 80)
    print(f"{'Data Dim':<10} {'Chunk Dim':<10} {'Shard Dim':<10} {'Divisible?':<12} {'Description'}")
    print("-" * 80)
    
    all_passed = True
    for data_dim, chunk_dim, description in test_cases:
        shard_dim = _calculate_shard_dimension(data_dim, chunk_dim)
        
        # When chunk_dim >= data_dim, the effective chunk size is data_dim
        effective_chunk_dim = min(chunk_dim, data_dim)
        is_divisible = shard_dim % effective_chunk_dim == 0
        status = "✅ YES" if is_divisible else "❌ NO"
        
        print(f"{data_dim:<10} {chunk_dim:<10} {shard_dim:<10} {status:<12} {description}")
        
        if not is_divisible:
            all_passed = False
            print(f"  ⚠️  ERROR: {shard_dim} % {effective_chunk_dim} = {shard_dim % effective_chunk_dim}")
    
    print("-" * 80)
    if all_passed:
        print("✅ All tests passed! Shard dimensions are properly divisible by chunk dimensions.")
    else:
        print("❌ Some tests failed! Check the implementation.")
    
    return all_passed


def test_encoding_creation():
    """Test the encoding creation with sharding enabled."""
    import numpy as np
    import xarray as xr
    from zarr.codecs import BloscCodec
    from eopf_geozarr.conversion.geozarr import _create_geozarr_encoding
    
    print("\n🧪 Testing encoding creation with sharding...")
    
    # Create a test dataset
    data = np.random.rand(1, 10980, 10980).astype(np.float32)
    ds = xr.Dataset({
        'b02': (['time', 'y', 'x'], data),
    }, coords={
        'time': [np.datetime64('2023-01-01')],
        'y': np.arange(10980),
        'x': np.arange(10980),
    })
    
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
    spatial_chunk = 4096
    
    # Test with sharding enabled
    print("\nTesting with sharding enabled:")
    encoding = _create_geozarr_encoding(ds, compressor, spatial_chunk, enable_sharding=True)
    
    for var, enc in encoding.items():
        if 'shards' in enc and enc['shards'] is not None:
            chunks = enc['chunks']
            shards = enc['shards']
            print(f"Variable: {var}")
            print(f"  Data shape: {ds[var].shape}")
            print(f"  Chunks: {chunks}")
            print(f"  Shards: {shards}")
            
            # Validate divisibility
            valid = True
            for i, (shard_dim, chunk_dim) in enumerate(zip(shards, chunks)):
                if shard_dim % chunk_dim != 0:
                    print(f"  ❌ Axis {i}: {shard_dim} % {chunk_dim} = {shard_dim % chunk_dim}")
                    valid = False
                else:
                    print(f"  ✅ Axis {i}: {shard_dim} % {chunk_dim} = 0")
            
            if valid:
                print("  ✅ All shard dimensions are divisible by chunk dimensions")
            else:
                print("  ❌ Some shard dimensions are not divisible by chunk dimensions")
    
    print("\nTesting with sharding disabled:")
    encoding_no_shard = _create_geozarr_encoding(ds, compressor, spatial_chunk, enable_sharding=False)
    
    for var, enc in encoding_no_shard.items():
        if 'shards' in enc:
            print(f"Variable: {var}, Shards: {enc['shards']}")


def main():
    """Run all tests."""
    print("🔧 Testing Zarr v3 Sharding Fix for GeoZarr")
    print("=" * 50)
    
    # Test the shard dimension calculation
    test1_passed = test_calculate_shard_dimension()
    
    # Test the encoding creation
    test_encoding_creation()
    
    print("\n" + "=" * 50)
    if test1_passed:
        print("✅ All critical tests passed!")
        print("🎉 The sharding fix should resolve the checksum mismatch issues.")
        print("\nKey improvements:")
        print("- Shard dimensions are now evenly divisible by chunk dimensions")
        print("- Added debugging output to show sharding configuration")
        print("- Enhanced shard calculation with preference for larger multipliers")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return test1_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
