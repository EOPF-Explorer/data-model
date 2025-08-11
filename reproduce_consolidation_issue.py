#!/usr/bin/env python3
"""
Reproduction script for zarr-python issue #3289:
"Option to preserve child metadata during consolidation at parent level"

This script demonstrates the current behavior of zarr.consolidate_metadata()
when consolidating at different hierarchical levels.
"""

import zarr
import tempfile
import shutil
from pathlib import Path
import json


def create_hierarchical_zarr_structure(store_path: Path):
    """
    Create a hierarchical zarr structure similar to EOPF Sentinel data.
    
    Structure:
    store.zarr/
    ├── measurements/
    │   └── reflectance/
    │       ├── r10m/           # 10m resolution group
    │       │   ├── l0/         # Native resolution array
    │       │   └── l1/         # Overview level 1 array
    │       ├── r20m/           # 20m resolution group
    │       │   ├── l0/
    │       │   └── l1/
    │       └── r60m/           # 60m resolution group
    │           ├── l0/
    │           └── l1/
    └── quality/
        └── l1c_quicklook/
            └── r10m/
                ├── l0/
                └── l1/
    """
    print(f"Creating hierarchical zarr structure at: {store_path}")
    
    # Create root group
    root = zarr.open_group(store_path, mode='w')
    
    # Create measurements hierarchy
    measurements = root.create_group("measurements")
    reflectance = measurements.create_group("reflectance")
    
    # Create resolution groups with arrays
    for resolution in ["r10m", "r20m", "r60m"]:
        res_group = reflectance.create_group(resolution)
        # Add some arrays to each resolution group
        res_group.create_array("l0", shape=(100, 100), dtype="float32", chunks=(50, 50))
        res_group.create_array("l1", shape=(50, 50), dtype="float32", chunks=(25, 25))
        
        # Add some attributes to make it more realistic
        res_group.attrs["resolution"] = resolution
        res_group.attrs["description"] = f"Data at {resolution} resolution"
    
    # Create quality hierarchy
    quality = root.create_group("quality")
    l1c = quality.create_group("l1c_quicklook")
    r10m_quality = l1c.create_group("r10m")
    r10m_quality.create_array("l0", shape=(100, 100), dtype="uint8", chunks=(50, 50))
    r10m_quality.create_array("l1", shape=(50, 50), dtype="uint8", chunks=(25, 25))
    
    print("✓ Hierarchical zarr structure created successfully")
    return root


def check_consolidated_metadata(store_path: Path, path: str = ""):
    """Check if consolidated metadata exists at a given path."""
    try:
        group = zarr.open_consolidated(store_path, path=path)
        has_consolidated = group.metadata.consolidated_metadata is not None
        if has_consolidated:
            num_items = len(group.metadata.consolidated_metadata.metadata)
            print(f"✓ Path '{path}' has consolidated metadata ({num_items} items)")
            return True
        else:
            print(f"✗ Path '{path}' has NO consolidated metadata")
            return False
    except Exception as e:
        print(f"✗ Path '{path}' cannot be opened as consolidated: {e}")
        return False


def check_zarr_json_files(store_path: Path):
    """Check which zarr.json files exist and if they contain consolidated metadata."""
    print("\n=== Checking zarr.json files on disk ===")
    
    zarr_json_files = list(store_path.rglob("zarr.json"))
    zarr_json_files.sort()
    
    for json_file in zarr_json_files:
        rel_path = json_file.relative_to(store_path)
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            has_consolidated = "consolidated_metadata" in data
            group_path = str(rel_path.parent) if rel_path.parent != Path('.') else "root"
            
            if has_consolidated:
                consolidated_items = len(data["consolidated_metadata"]["metadata"])
                print(f"✓ {group_path}: zarr.json contains consolidated_metadata ({consolidated_items} items)")
            else:
                print(f"✗ {group_path}: zarr.json has NO consolidated_metadata")
                
        except Exception as e:
            print(f"✗ Error reading {rel_path}: {e}")


def demonstrate_issue():
    """Demonstrate the consolidation issue described in zarr-python #3289."""
    
    print("=" * 80)
    print("ZARR CONSOLIDATION ISSUE REPRODUCTION")
    print("Issue: https://github.com/zarr-developers/zarr-python/issues/3289")
    print("=" * 80)
    
    # Create temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store.zarr"
        
        # Step 1: Create the hierarchical structure
        print("\n1. Creating hierarchical zarr structure...")
        root = create_hierarchical_zarr_structure(store_path)
        
        # Step 2: Consolidate metadata at resolution group levels first
        print("\n2. Consolidating metadata at resolution group levels...")
        resolution_paths = [
            "measurements/reflectance/r10m",
            "measurements/reflectance/r20m", 
            "measurements/reflectance/r60m"
        ]
        
        for path in resolution_paths:
            print(f"   Consolidating: {path}")
            zarr.consolidate_metadata(store_path, path=path)
        
        # Check that child consolidation worked
        print("\n3. Verifying child-level consolidated metadata...")
        for path in resolution_paths:
            check_consolidated_metadata(store_path, path)
        
        # Check zarr.json files after child consolidation
        check_zarr_json_files(store_path)
        
        # Step 3: Now consolidate at root level - this is where the issue occurs
        print("\n4. Consolidating metadata at ROOT level...")
        print("   This should preserve child consolidated metadata, but does it?")
        zarr.consolidate_metadata(store_path, path="")
        
        # Step 4: Check if child consolidated metadata still exists
        print("\n5. Checking if child consolidated metadata is preserved...")
        
        print("\n--- Checking in-memory consolidated metadata ---")
        for path in resolution_paths:
            check_consolidated_metadata(store_path, path)
        
        # Also check root level
        check_consolidated_metadata(store_path, "")
        
        # Check zarr.json files after root consolidation
        check_zarr_json_files(store_path)
        
        # Step 5: Demonstrate the specific issue from the GitHub issue
        print("\n6. Demonstrating the specific issue scenario...")
        print("   Scenario: Consolidate children first, then parent")
        
        # Re-create fresh structure
        shutil.rmtree(store_path)
        create_hierarchical_zarr_structure(store_path)
        
        # Consolidate children first
        print("   Step A: Consolidate resolution groups...")
        for path in resolution_paths:
            zarr.consolidate_metadata(store_path, path=path)
            
        print("   Step B: Verify children have consolidated metadata...")
        children_have_metadata = []
        for path in resolution_paths:
            has_metadata = check_consolidated_metadata(store_path, path)
            children_have_metadata.append(has_metadata)
        
        # Now consolidate parent
        print("   Step C: Consolidate at root level...")
        zarr.consolidate_metadata(store_path, path="")
        
        print("   Step D: Check if children STILL have consolidated metadata...")
        children_still_have_metadata = []
        for path in resolution_paths:
            has_metadata = check_consolidated_metadata(store_path, path)
            children_still_have_metadata.append(has_metadata)
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if all(children_have_metadata) and all(children_still_have_metadata):
            print("✓ SUCCESS: Child consolidated metadata is PRESERVED after parent consolidation")
            print("  This means the issue may be resolved or doesn't exist as described.")
        elif all(children_have_metadata) and not all(children_still_have_metadata):
            print("✗ ISSUE CONFIRMED: Child consolidated metadata is CLEARED after parent consolidation")
            print("  This confirms the issue described in zarr-python #3289")
        else:
            print("? UNCLEAR: Unexpected behavior detected")
            
        print(f"\nChildren had metadata before root consolidation: {children_have_metadata}")
        print(f"Children have metadata after root consolidation: {children_still_have_metadata}")
        
        # Additional analysis
        print("\n--- Additional Analysis ---")
        try:
            root_group = zarr.open_consolidated(store_path, path="")
            print(f"Root consolidated metadata contains {len(root_group.metadata.consolidated_metadata.metadata)} items")
            
            # Check if child groups are accessible through root consolidated metadata
            try:
                r10m_through_root = root_group["measurements/reflectance/r10m"]
                print("✓ Can access r10m group through root consolidated metadata")
                print(f"  r10m arrays accessible: {list(r10m_through_root.keys())}")
            except Exception as e:
                print(f"✗ Cannot access r10m through root: {e}")
                
        except Exception as e:
            print(f"✗ Error analyzing root consolidated metadata: {e}")


if __name__ == "__main__":
    demonstrate_issue()
