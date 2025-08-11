#!/usr/bin/env python3
"""
Additional tests for zarr consolidation behavior to understand the issue better.
This script tests various scenarios mentioned in the GitHub issue.
"""

import zarr
import tempfile
import shutil
from pathlib import Path
import json


def test_scenario_from_issue():
    """
    Test the exact scenario described in the GitHub issue:
    1. Consolidate at resolution group level (works fine)
    2. Consolidate at root level (should remove child consolidated metadata?)
    """
    print("=" * 80)
    print("TESTING EXACT SCENARIO FROM GITHUB ISSUE")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "store.zarr"
        
        # Create the exact structure mentioned in the issue
        print("Creating EOPF Sentinel-like structure...")
        root = zarr.open_group(store_path, mode='w')
        
        # Create measurements/reflectance structure
        measurements = root.create_group("measurements")
        reflectance = measurements.create_group("reflectance")
        
        # Create resolution groups with overview levels
        for resolution in ["r10m", "r20m", "r60m"]:
            res_group = reflectance.create_group(resolution)
            # Create overview levels 0, 1, 2
            for level in [0, 1, 2]:
                level_array = res_group.create_array(
                    str(level), 
                    shape=(100 // (2**level), 100 // (2**level)), 
                    dtype="float32", 
                    chunks=(50 // (2**level), 50 // (2**level))
                )
                level_array.attrs["overview_level"] = level
        
        # Create quality structure
        quality = root.create_group("quality")
        l1c_quicklook = quality.create_group("l1c_quicklook")
        r10m_quality = l1c_quicklook.create_group("r10m")
        for level in [0, 1, 2]:
            r10m_quality.create_array(
                str(level), 
                shape=(100 // (2**level), 100 // (2**level)), 
                dtype="uint8", 
                chunks=(50 // (2**level), 50 // (2**level))
            )
        
        print("✓ Structure created")
        
        # Step 1: Consolidate at resolution group level (as mentioned in issue)
        print("\nStep 1: Consolidate at resolution group level...")
        zarr.consolidate_metadata(store_path, path="measurements/reflectance/r10m")
        zarr.consolidate_metadata(store_path, path="measurements/reflectance/r20m") 
        zarr.consolidate_metadata(store_path, path="measurements/reflectance/r60m")
        
        # Verify consolidation worked
        print("Verifying resolution group consolidation...")
        for resolution in ["r10m", "r20m", "r60m"]:
            path = f"measurements/reflectance/{resolution}"
            try:
                group = zarr.open_consolidated(store_path, path=path)
                print(f"✓ {resolution}: {len(group.metadata.consolidated_metadata.metadata)} items consolidated")
            except Exception as e:
                print(f"✗ {resolution}: Failed to open consolidated - {e}")
        
        # Step 2: Consolidate at root level (this is where issue should occur)
        print("\nStep 2: Consolidate at root level...")
        zarr.consolidate_metadata(store_path, path="")
        
        # Check if resolution groups still have consolidated metadata
        print("\nChecking if resolution groups still have consolidated metadata...")
        for resolution in ["r10m", "r20m", "r60m"]:
            path = f"measurements/reflectance/{resolution}"
            try:
                group = zarr.open_consolidated(store_path, path=path)
                if group.metadata.consolidated_metadata is not None:
                    print(f"✓ {resolution}: STILL has consolidated metadata ({len(group.metadata.consolidated_metadata.metadata)} items)")
                else:
                    print(f"✗ {resolution}: NO LONGER has consolidated metadata")
            except Exception as e:
                print(f"✗ {resolution}: Cannot open as consolidated - {e}")
        
        # Check root level access
        print("\nChecking root level access...")
        try:
            root_group = zarr.open_consolidated(store_path, path="")
            print(f"✓ Root has consolidated metadata with {len(root_group.metadata.consolidated_metadata.metadata)} items")
            
            # Try to access resolution groups through root
            try:
                r10m_via_root = root_group["measurements/reflectance/r10m"]
                print(f"✓ Can access r10m through root: {list(r10m_via_root.keys())}")
            except Exception as e:
                print(f"✗ Cannot access r10m through root: {e}")
                
        except Exception as e:
            print(f"✗ Root consolidation failed: {e}")


def test_disk_vs_memory_metadata():
    """
    Test whether the issue is about disk storage vs in-memory representation.
    """
    print("\n" + "=" * 80)
    print("TESTING DISK VS MEMORY METADATA")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "store.zarr"
        
        # Create simple structure
        root = zarr.open_group(store_path, mode='w')
        child = root.create_group("child")
        child.create_array("data", shape=(10, 10), dtype="float32")
        
        print("1. Consolidate child first...")
        zarr.consolidate_metadata(store_path, path="child")
        
        # Check disk files
        child_zarr_json = store_path / "child" / "zarr.json"
        print(f"2. Child zarr.json exists: {child_zarr_json.exists()}")
        
        if child_zarr_json.exists():
            with open(child_zarr_json, 'r') as f:
                child_data = json.load(f)
            has_consolidated = "consolidated_metadata" in child_data
            print(f"   Child zarr.json has consolidated_metadata: {has_consolidated}")
            if has_consolidated:
                print(f"   Items in child consolidated metadata: {len(child_data['consolidated_metadata']['metadata'])}")
        
        print("3. Consolidate root...")
        zarr.consolidate_metadata(store_path, path="")
        
        # Check disk files after root consolidation
        print("4. After root consolidation:")
        print(f"   Child zarr.json still exists: {child_zarr_json.exists()}")
        
        if child_zarr_json.exists():
            with open(child_zarr_json, 'r') as f:
                child_data_after = json.load(f)
            has_consolidated_after = "consolidated_metadata" in child_data_after
            print(f"   Child zarr.json still has consolidated_metadata: {has_consolidated_after}")
            if has_consolidated_after:
                print(f"   Items in child consolidated metadata: {len(child_data_after['consolidated_metadata']['metadata'])}")
        
        # Check root zarr.json
        root_zarr_json = store_path / "zarr.json"
        if root_zarr_json.exists():
            with open(root_zarr_json, 'r') as f:
                root_data = json.load(f)
            has_root_consolidated = "consolidated_metadata" in root_data
            print(f"   Root zarr.json has consolidated_metadata: {has_root_consolidated}")
            if has_root_consolidated:
                print(f"   Items in root consolidated metadata: {len(root_data['consolidated_metadata']['metadata'])}")


def test_zarr_version_info():
    """Print zarr version and other relevant info."""
    print("\n" + "=" * 80)
    print("ENVIRONMENT INFO")
    print("=" * 80)
    
    import zarr
    print(f"Zarr version: {zarr.__version__}")
    
    # Try to get more detailed version info
    try:
        import zarr._version
        print(f"Zarr detailed version: {zarr._version.version}")
    except:
        pass
    
    # Check if we're using zarr v3
    try:
        # This is a zarr v3 specific import
        from zarr.core.metadata import ArrayV3Metadata
        print("✓ Zarr v3 features available")
    except ImportError:
        print("✗ Zarr v3 features not available")


if __name__ == "__main__":
    test_zarr_version_info()
    test_scenario_from_issue()
    test_disk_vs_memory_metadata()
