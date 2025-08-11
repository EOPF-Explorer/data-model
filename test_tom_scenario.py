#!/usr/bin/env python3
"""
Test the exact scenario provided by TomAugspurger in the GitHub issue.
This replicates his test case to see if we can reproduce any issues.
"""

import zarr
import tempfile
from pathlib import Path


def test_tom_augspurger_scenario():
    """
    Replicate the exact test case provided by TomAugspurger in the GitHub issue.
    """
    print("=" * 80)
    print("TESTING TOM AUGSPURGER'S SCENARIO FROM GITHUB ISSUE")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store.zarr"
        
        # Replicate Tom's exact code - use the store path directly
        root = zarr.create_group(store_path, path="", overwrite=True)
        measurements = root.create_group("measurements", overwrite=True)
        reflectance = measurements.create_group("reflectance", overwrite=True)
        r10 = reflectance.create_group("r10", overwrite=True)
        r10.create_array("l0", shape=(10, 10), dtype="float32", overwrite=True)
        r10.create_array("l1", shape=(10, 10), dtype="float32", overwrite=True)
        r20 = reflectance.create_group("r20", overwrite=True)
        r20.create_array("l0", shape=(10, 10), dtype="float32", overwrite=True)
        r20.create_array("l1", shape=(10, 10), dtype="float32", overwrite=True)
        r60 = reflectance.create_group("r60", overwrite=True)
        r60.create_array("l0", shape=(10, 10), dtype="float32", overwrite=True)
        r60.create_array("l1", shape=(10, 10), dtype="float32", overwrite=True)
        quality = root.create_group("quality")
        l1c = quality.create_group("l1c", overwrite=True)
        r10m = l1c.create_group("r10")
        r10m.create_array("l0", shape=(10, 10), dtype="float32", overwrite=True)
        r10m.create_array("l1", shape=(10, 10), dtype="float32", overwrite=True)
        
        print("✓ Structure created (Tom's exact structure)")
        
        # Consolidate at all the paths Tom mentioned
        paths_to_consolidate = [
            "",
            "measurements",
            "measurements/reflectance",
            "measurements/reflectance/r10",
            "measurements/reflectance/r20",
            "measurements/reflectance/r60",
        ]
        
        print("\nConsolidating metadata at all paths...")
        for path in paths_to_consolidate:
            print(f"   Consolidating: '{path}'")
            zarr.consolidate_metadata(store_path, path=path)
        
        print("\nChecking consolidated metadata results...")
        
        # Check root
        try:
            root_consolidated = zarr.open_consolidated(store_path, path="")
            print(f"✓ Root: {root_consolidated.metadata.consolidated_metadata}")
            print(f"   Root consolidated items: {len(root_consolidated.metadata.consolidated_metadata.metadata) if root_consolidated.metadata.consolidated_metadata else 0}")
        except Exception as e:
            print(f"✗ Root: {e}")
        
        # Check measurements
        try:
            measurements_consolidated = zarr.open_consolidated(store_path, path="measurements")
            print(f"✓ Measurements: {measurements_consolidated.metadata.consolidated_metadata}")
            print(f"   Measurements consolidated items: {len(measurements_consolidated.metadata.consolidated_metadata.metadata) if measurements_consolidated.metadata.consolidated_metadata else 0}")
        except Exception as e:
            print(f"✗ Measurements: {e}")
        
        # Check measurements/reflectance/r10
        try:
            r10_consolidated = zarr.open_consolidated(store_path, path="measurements/reflectance/r10")
            print(f"✓ R10: {r10_consolidated.metadata.consolidated_metadata}")
            print(f"   R10 consolidated items: {len(r10_consolidated.metadata.consolidated_metadata.metadata) if r10_consolidated.metadata.consolidated_metadata else 0}")
        except Exception as e:
            print(f"✗ R10: {e}")
        
        print("\n" + "=" * 60)
        print("DETAILED ANALYSIS")
        print("=" * 60)
        
        # Show the detailed output like Tom's example
        print("\nroot")
        try:
            root_group = zarr.open_consolidated(store_path, path="")
            print(root_group.metadata.consolidated_metadata)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\nmeasurements")
        try:
            measurements_group = zarr.open_consolidated(store_path, path="measurements")
            print(measurements_group.metadata.consolidated_metadata)
        except Exception as e:
            print(f"Error: {e}")
        
        print("\nmeasurements/reflectance/r10")
        try:
            r10_group = zarr.open_consolidated(store_path, path="measurements/reflectance/r10")
            print(r10_group.metadata.consolidated_metadata)
        except Exception as e:
            print(f"Error: {e}")


def test_issue_reproduction_attempt():
    """
    Try to reproduce the issue by testing different consolidation orders.
    """
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT CONSOLIDATION ORDERS")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test_store.zarr"
        
        print("Scenario 1: Child first, then parent")
        print("-" * 40)
        
        # Create structure
        root = zarr.open_group(store_path, mode='w')
        child = root.create_group("child")
        child.create_array("data", shape=(10, 10), dtype="float32")
        
        # Consolidate child first
        zarr.consolidate_metadata(store_path, path="child")
        print("✓ Child consolidated first")
        
        # Check child has consolidated metadata
        child_group = zarr.open_consolidated(store_path, path="child")
        child_has_metadata = child_group.metadata.consolidated_metadata is not None
        print(f"   Child has consolidated metadata: {child_has_metadata}")
        
        # Now consolidate parent
        zarr.consolidate_metadata(store_path, path="")
        print("✓ Parent consolidated")
        
        # Check if child still has consolidated metadata
        try:
            child_group_after = zarr.open_consolidated(store_path, path="child")
            child_still_has_metadata = child_group_after.metadata.consolidated_metadata is not None
            print(f"   Child still has consolidated metadata: {child_still_has_metadata}")
            
            if child_has_metadata and not child_still_has_metadata:
                print("🚨 ISSUE DETECTED: Child metadata was cleared!")
            elif child_has_metadata and child_still_has_metadata:
                print("✅ Child metadata preserved")
            else:
                print("❓ Unexpected state")
                
        except Exception as e:
            print(f"✗ Error checking child after parent consolidation: {e}")


if __name__ == "__main__":
    test_tom_augspurger_scenario()
    test_issue_reproduction_attempt()
