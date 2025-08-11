#!/usr/bin/env python3
"""
Test zarr consolidation behavior with xarray's to_zarr() function.
The issue might be related to how xarray's to_zarr() handles consolidated metadata,
especially with DataTree structures.
"""

import xarray as xr
import zarr
import numpy as np
import tempfile
from pathlib import Path


def create_sample_datatree():
    """
    Create a sample DataTree structure similar to EOPF Sentinel data.
    """
    print("Creating sample DataTree structure...")
    
    # Create root dataset
    root_data = xr.Dataset({
        'metadata': xr.DataArray(
            'EOPF Sentinel Data',
            attrs={'description': 'Root level metadata'}
        )
    })
    
    # Create child datasets for different resolution groups
    datasets = {}
    
    for resolution in ['r10m', 'r20m', 'r60m']:
        size = {'r10m': 100, 'r20m': 50, 'r60m': 25}[resolution]
        
        # Create data for different overview levels
        level_data = {}
        for level in [0, 1, 2]:
            level_size = size // (2**level)
            
            # Create sample data
            data = np.random.random((level_size, level_size)).astype(np.float32)
            
            level_data[f'level_{level}'] = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={
                    'y': np.linspace(0, 1000, level_size),
                    'x': np.linspace(0, 1000, level_size)
                },
                attrs={
                    'long_name': f'Reflectance data level {level}',
                    'resolution': resolution,
                    'overview_level': level
                }
            )
        
        # Create dataset for this resolution
        datasets[f'measurements/reflectance/{resolution}'] = xr.Dataset(
            level_data,
            attrs={
                'resolution': resolution,
                'description': f'Data at {resolution} resolution'
            }
        )
    
    # Create DataTree
    try:
        from datatree import DataTree
        
        # Create root node
        dt = DataTree(data=root_data, name='root')
        
        # Add child nodes
        for path, dataset in datasets.items():
            dt[path] = DataTree(data=dataset)
        
        print("✓ DataTree structure created successfully")
        return dt
        
    except ImportError:
        print("⚠ DataTree not available, creating manual structure")
        return None, datasets


def test_to_zarr_consolidation_behavior():
    """
    Test how xarray's to_zarr() behaves with consolidation.
    """
    print("=" * 80)
    print("TESTING XARRAY to_zarr() CONSOLIDATION BEHAVIOR")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store.zarr"
        print(f"Using temporary store path: {store_path}")
        
        # Create sample data structure
        dt_or_datasets = create_sample_datatree()
        
        if isinstance(dt_or_datasets, tuple):
            # DataTree not available, use manual approach
            dt, datasets = dt_or_datasets
            
            print("\n1. Writing datasets manually to zarr...")
            
            # Write each dataset to its group
            for group_path, dataset in datasets.items():
                print(f"   Writing {group_path}...")
                dataset.to_zarr(store_path, group=group_path, mode='a')
                
        else:
            # Use DataTree
            dt = dt_or_datasets
            
            print("\n1. Writing DataTree to zarr...")
            try:
                dt.to_zarr(store_path)
                print("✓ DataTree written to zarr successfully")
            except Exception as e:
                print(f"✗ DataTree to_zarr failed: {e}")
                return
        
        # Step 2: Check initial zarr structure
        print("\n2. Checking initial zarr structure...")
        try:
            root_group = zarr.open_group(store_path, mode='r')
            print(f"✓ Root group accessible: {list(root_group.keys())}")
            
            # Check if child groups exist
            for resolution in ['r10m', 'r20m', 'r60m']:
                group_path = f'measurements/reflectance/{resolution}'
                try:
                    child_group = root_group[group_path]
                    print(f"✓ {resolution} group accessible: {list(child_group.keys())}")
                except Exception as e:
                    print(f"✗ {resolution} group not accessible: {e}")
                    
        except Exception as e:
            print(f"✗ Initial zarr structure check failed: {e}")
            return
        
        # Step 3: Test consolidation at child level first
        print("\n3. Consolidating metadata at child levels...")
        resolution_paths = [
            "measurements/reflectance/r10m",
            "measurements/reflectance/r20m", 
            "measurements/reflectance/r60m"
        ]
        
        for path in resolution_paths:
            try:
                print(f"   Consolidating: {path}")
                zarr.consolidate_metadata(store_path, path=path)
                print(f"   ✓ {path} consolidated successfully")
            except Exception as e:
                print(f"   ✗ {path} consolidation failed: {e}")
        
        # Step 4: Test xarray access to consolidated children
        print("\n4. Testing xarray access to consolidated children...")
        for resolution in ['r10m', 'r20m', 'r60m']:
            group_path = f'measurements/reflectance/{resolution}'
            try:
                ds = xr.open_zarr(
                    store_path,
                    group=group_path,
                    consolidated=True
                )
                print(f"✓ {resolution} accessible via xarray: {list(ds.data_vars.keys())}")
            except Exception as e:
                print(f"✗ {resolution} not accessible via xarray: {e}")
        
        # Step 5: Consolidate at root level - THIS IS THE CRITICAL TEST
        print("\n5. Consolidating metadata at ROOT level...")
        try:
            zarr.consolidate_metadata(store_path, path="")
            print("✓ Root consolidation successful")
        except Exception as e:
            print(f"✗ Root consolidation failed: {e}")
            return
        
        # Step 6: Test xarray access to children AFTER root consolidation
        print("\n6. CRITICAL TEST: xarray access to children AFTER root consolidation...")
        
        for resolution in ['r10m', 'r20m', 'r60m']:
            group_path = f'measurements/reflectance/{resolution}'
            
            # Test with consolidated=True
            try:
                ds_consolidated = xr.open_zarr(
                    store_path,
                    group=group_path,
                    consolidated=True
                )
                print(f"✓ {resolution} still accessible (consolidated=True): {list(ds_consolidated.data_vars.keys())}")
                consolidated_works = True
            except Exception as e:
                print(f"✗ {resolution} NOT accessible (consolidated=True): {e}")
                consolidated_works = False
            
            # Test with consolidated=False as fallback
            try:
                ds_non_consolidated = xr.open_zarr(
                    store_path,
                    group=group_path,
                    consolidated=False
                )
                print(f"✓ {resolution} accessible (consolidated=False): {list(ds_non_consolidated.data_vars.keys())}")
                non_consolidated_works = True
            except Exception as e:
                print(f"✗ {resolution} NOT accessible (consolidated=False): {e}")
                non_consolidated_works = False
            
            # Analyze the issue
            if not consolidated_works and non_consolidated_works:
                print(f"🚨 ISSUE DETECTED for {resolution}: consolidated=True fails after root consolidation!")
            elif not consolidated_works and not non_consolidated_works:
                print(f"🚨 SEVERE ISSUE for {resolution}: Both consolidated and non-consolidated access fail!")


def test_to_zarr_with_existing_consolidated():
    """
    Test what happens when using to_zarr() on a store that already has consolidated metadata.
    """
    print("\n" + "=" * 80)
    print("TESTING to_zarr() WITH EXISTING CONSOLIDATED METADATA")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store.zarr"
        print(f"Using temporary store path: {store_path}")
        
        print("1. Creating initial zarr store with consolidated metadata...")
        
        # Create initial structure
        root = zarr.open_group(store_path, mode='w')
        child = root.create_group("child")
        child.create_array("data", shape=(10, 10), dtype="float32")
        child.attrs['description'] = 'Initial data'
        
        # Consolidate
        zarr.consolidate_metadata(store_path, path="child")
        print("✓ Initial store with consolidated metadata created")
        
        # Verify consolidation works
        try:
            ds_initial = xr.open_zarr(
                store_path,
                group="child",
                consolidated=True
            )
            print(f"✓ Initial consolidated access works: {list(ds_initial.data_vars.keys())}")
        except Exception as e:
            print(f"✗ Initial consolidated access failed: {e}")
        
        print("\n2. Adding new data using to_zarr()...")
        
        # Create new data to add
        new_data = xr.Dataset({
            'temperature': xr.DataArray(
                np.random.random((5, 5)),
                dims=['y', 'x'],
                attrs={'units': 'K'}
            )
        })
        
        # Add to existing store
        try:
            new_data.to_zarr(store_path, group="child", mode='a')
            print("✓ New data added via to_zarr()")
        except Exception as e:
            print(f"✗ to_zarr() failed: {e}")
            return
        
        print("\n3. Testing consolidated access after to_zarr()...")
        
        # Test if consolidated access still works
        try:
            ds_after = xr.open_zarr(
                store_path,
                group="child",
                consolidated=True
            )
            print(f"✓ Consolidated access still works: {list(ds_after.data_vars.keys())}")
        except Exception as e:
            print(f"✗ Consolidated access broken after to_zarr(): {e}")
            
            # Try non-consolidated access
            try:
                ds_non_consolidated = xr.open_zarr(
                    store_path,
                    group="child",
                    consolidated=False
                )
                print(f"✓ Non-consolidated access works: {list(ds_non_consolidated.data_vars.keys())}")
                print("🚨 ISSUE DETECTED: to_zarr() breaks consolidated metadata!")
            except Exception as e2:
                print(f"✗ Even non-consolidated access failed: {e2}")


def test_version_info():
    """Print version information for debugging."""
    print("=" * 80)
    print("ENVIRONMENT INFO")
    print("=" * 80)
    
    print(f"Xarray version: {xr.__version__}")
    print(f"Zarr version: {zarr.__version__}")
    
    try:
        from datatree import DataTree
        import datatree
        print(f"DataTree version: {datatree.__version__}")
        print("✓ DataTree available")
    except ImportError:
        print("✗ DataTree not available")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")


if __name__ == "__main__":
    test_version_info()
    test_to_zarr_consolidation_behavior()
    test_to_zarr_with_existing_consolidated()
