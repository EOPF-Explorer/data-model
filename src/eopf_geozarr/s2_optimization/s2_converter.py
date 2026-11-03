"""
Main S2 optimization converter.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict

import structlog
import zarr
from pydantic import TypeAdapter
from pydantic_zarr.v3 import is_array_spec, is_group_spec
from pyproj import CRS

from eopf_geozarr.conversion import utils
from eopf_geozarr.conversion.geozarr import get_zarr_group
from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.data_api.s2_plan import load_sentinel2
from eopf_geozarr.data_api.s2_write import write_geozarr

if TYPE_CHECKING:
    from collections.abc import Sequence

    import xarray as xr
    from pydantic_zarr.v3 import AnyGroupSpec

log = structlog.get_logger()


def initialize_crs_from_dataset(dt_input: xr.DataTree) -> CRS | None:
    """
    Initialize CRS from dataset by checking data variables.

    Args:
        dt_input: Input DataTree

    Returns:
        CRS object if found, None otherwise
    """
    # For CPM >= 2.6.0, the EPSG code is stored in root attributes
    epsg_cpm_260 = dt_input.attrs.get("other_metadata", {}).get(
        "horizontal_CRS_code",
        dt_input.attrs.get("other_metadata", {}).get("horizontal_crs_code", None),
    )
    if epsg_cpm_260 is not None:
        try:
            # Handle both integer (32632) and string ("EPSG:32632" or "32632") formats
            if isinstance(epsg_cpm_260, str):
                # Extract numeric part from string like "EPSG:32632" or "32632"
                epsg_code = int(epsg_cpm_260.split(":")[-1])
            else:
                # Already an integer
                epsg_code = int(epsg_cpm_260)
            crs = CRS.from_epsg(epsg_code)
            log.info("Initialized CRS from CPM 2.6.0+ metadata", epsg=epsg_code)
        except Exception as e:
            log.warning(
                "Failed to initialize CRS from CPM 2.6.0+ metadata",
                epsg=epsg_cpm_260,
                error=str(e),
            )
        else:
            return crs

    for group_path in dt_input.groups:
        if group_path == ".":
            continue
        group_node = dt_input[group_path]
        if not hasattr(group_node, "ds") or group_node.ds is None:
            continue
        dataset = group_node.ds

        # Check if dataset has rio accessor with CRS. rioxarray returns a
        # rasterio CRS; convert it to a pyproj CRS (the declared return type),
        # which also validates the value at runtime.
        if hasattr(dataset, "rio"):
            try:
                rio_crs = dataset.rio.crs
                if rio_crs is not None:
                    ds_crs = CRS.from_user_input(rio_crs)
                    log.info("Initialized CRS from dataset", crs=str(ds_crs))
                    return ds_crs
            except Exception:
                log.debug("Failed to get CRS from dataset rio accessor")

        # Check data variables for CRS information
        for var in dataset.data_vars.values():
            if hasattr(var, "rio"):
                try:
                    rio_crs = var.rio.crs
                    if rio_crs is not None:
                        var_crs = CRS.from_user_input(rio_crs)
                        log.info("Initialized CRS from variable", crs=str(var_crs))
                        return var_crs
                except Exception:
                    log.debug("Failed to get CRS from variable rio accessor")

            # Check for proj:epsg attribute
            if "proj:epsg" in var.attrs:
                try:
                    epsg = var.attrs["proj:epsg"]
                    crs = CRS.from_epsg(epsg)
                    log.info("Initialized CRS from EPSG code", epsg=epsg)
                except Exception:
                    log.debug("Failed to initialize CRS from proj:epsg attribute")
                else:
                    return crs

    log.warning("Could not initialize CRS from dataset")
    return None


def _validate_s2_input(dt_input: xr.DataTree) -> None:
    """
    Validate that the input DataTree is a Sentinel-2 product.

    Validation runs against the DataTree's backing zarr store. Trees built in
    memory (e.g. by the CPM SAFE reader) have no backing group; callers on
    that path are responsible for routing only Sentinel-2 products here.
    """
    try:
        backing_group = get_zarr_group(dt_input)
    except TypeError:
        log.info("Input DataTree has no zarr backend; skipping input store validation")
        return
    try:
        is_s2 = is_sentinel2_dataset(backing_group)
    except TypeError:
        # is_sentinel2_dataset can still raise on backing stores it can't
        # introspect at all (neither the Zarr V2 model nor the V3 structural
        # check apply); treat that as "not checkable" rather than "not S2".
        log.info("Backing zarr store could not be validated; skipping input store validation")
        return
    if not is_s2:
        raise ValueError("Input dataset is not a Sentinel-2 product")


#: The write policy the spec-level plan fixes; `convert_s2_optimized` refuses
#: option values that deviate from it.
_PLAN_POLICY = (
    "sharding enabled, 1024-pixel inner chunks, blosc-zstd level 3, "
    "and the ESA packed-integer layout (no scale_offset codec)"
)


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = True,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    validate_output: bool = True,
    scale_offset_codec: bool = False,
) -> xr.DataTree:
    """
    Convert a Sentinel-2 product to the optimized GeoZarr layout.

    The conversion is plan-driven: the input store is validated into the
    `Sentinel2Root` model, `plan_geozarr` derives the output hierarchy from
    the model's own metadata, and `write_geozarr` writes the planned
    documents first and then fills in the chunk data. The write policy is
    the plan's fixed policy, so the layout options must hold their default
    values; a deviating value is refused.

    Args:
        dt_input: Input Sentinel-2 DataTree. A tree opened from a Zarr V2
            store is read through its backing store; any other tree (built in
            memory by the CPM SAFE reader, or backed by a Zarr V3 store) is
            read through the tree itself, with its variables taken as
            xarray's zarr writer would store them.
        output_path: Output path
        enable_sharding: Must be True (the plan's policy).
        spatial_chunk: Must be 1024 (the plan's policy).
        compression_level: Must be 3 (the plan's policy).
        validate_output: Whether to validate the output
        scale_offset_codec: Must be False: the output is written as in the
            ESA product (packed integers with CF `scale_factor` /
            `add_offset` / `_FillValue`).

    Returns:
        Optimized DataTree
    """

    start_time = time.time()

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )

    deviations = [
        name
        for name, held in (
            ("enable_sharding=False", not enable_sharding),
            (f"spatial_chunk={spatial_chunk}", spatial_chunk != 1024),
            (f"compression_level={compression_level}", compression_level != 3),
            ("scale_offset_codec=True", scale_offset_codec),
        )
        if held
    ]
    if deviations:
        msg = (
            f"The plan-driven S2 writer has a fixed write policy ({_PLAN_POLICY}); "
            f"unsupported option(s): {', '.join(deviations)}"
        )
        raise ValueError(msg)

    _validate_s2_input(dt_input)

    # Initialize CRS from dataset
    crs = initialize_crs_from_dataset(dt_input)

    try:
        backing: zarr.Group | None = get_zarr_group(dt_input)
    except TypeError:
        # A tree built in memory (the CPM SAFE reader) has no backing store.
        backing = None
    # A Zarr V2 store is the input model's native domain and is read directly;
    # everything else (an in-memory tree, a V3-backed tree) is read through
    # the tree's GroupSpec-shaped view — no scratch copy.
    source: zarr.Group | xr.DataTree = (
        backing if backing is not None and backing.metadata.zarr_format == 2 else dt_input
    )

    log.info("Step 1: Validating the input against the Sentinel-2 model")
    root = load_sentinel2(source)

    log.info("Step 2: Writing the planned GeoZarr hierarchy")
    plan = write_geozarr(root, source, output_path)

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    datasets = _planned_dataset_index(plan)
    add_s2_stac_assets(
        dt_input, datasets, crs=crs, reflectance_shape=_planned_reflectance_shape(plan)
    )
    utils.updated_root_consolidation(
        dt_input=dt_input,
        output_path=output_path,
        datasets=datasets,
        # the paths the plan maps to None are the multiscale pyramid roots
        # (reflectance and the written aux pyramids alike), so the consolidated
        # set follows the plan instead of a second policy table
        additional_consolidate_paths=[
            path for path, variable_names in datasets.items() if variable_names is None
        ],
    )

    # Step 4: Validation
    if validate_output:
        log.info("Step 4: Validating optimized dataset")
        validation_results = validate_optimized_dataset(output_path)
        if not validation_results["is_valid"]:
            log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return result_dt


def _planned_dataset_index(plan: AnyGroupSpec) -> dict[str, tuple[str, ...] | None]:
    """
    The planned dataset paths, each mapped to its data variable names (an
    array that is not a coordinate or the grid mapping), with multiscale
    roots mapped to None — the bookkeeping the root consolidation reads,
    matching what the DataTree writer's `processed_groups` carried.
    """
    index: dict[str, tuple[str, ...] | None] = {}

    def walk(spec: AnyGroupSpec, path: str) -> None:
        attributes = spec.attributes
        if path and isinstance(attributes, Mapping) and "multiscales" in attributes:
            index[path] = None
        names = tuple(
            name
            for name, member in spec.members.items()
            if is_array_spec(member)
            and name != "spatial_ref"
            and (member.dimension_names or ()) != (name,)
        )
        if names:
            index[path] = names
        for name, member in spec.members.items():
            if is_group_spec(member):
                walk(member, f"{path}/{name}")

    walk(plan, "")
    return index


def _planned_reflectance_shape(plan: AnyGroupSpec) -> tuple[int, ...] | None:
    """The (rows, cols) of the finest planned reflectance level, if present."""
    level = plan.get_path("measurements/reflectance/r10m")
    if not is_group_spec(level):
        return None
    for member in level.members.values():
        if is_array_spec(member) and len(member.shape) == 2:
            return tuple(member.shape)
    return None


def add_s2_stac_assets(
    dt_input: xr.DataTree,
    datasets: Mapping[str, Sequence[str] | None],
    crs: CRS | None = None,
    reflectance_shape: tuple[int, ...] | None = None,
) -> None:
    """Add the S2 pyramid levels and the reflectance pyramid as STAC assets.

    `datasets` is the planned dataset index: every dataset group path mapped
    to its data variable names, with multiscale roots mapped to None.
    `reflectance_shape` is the finest reflectance level's (rows, cols) for the
    asset's proj:shape. Updates `dt_input.attrs["stac_discovery"]` in place,
    so call it before `updated_root_consolidation` writes the root metadata
    from those attributes. This is a temporary fix that should move to
    eopf-stac.
    """
    if not dt_input.attrs:
        return

    # this can be used to add multiscale paths to the stac attributes
    updated_stac_attrs = utils.add_multiscale_pyramids_to_stac_metadata(datasets, dt_input.attrs)

    # Reference the pyramid root group, not the individual levels. That
    # group carries the `multiscales` attribute, and the
    # `profile=multiscales` media-type parameter tells a consumer to look
    # for it there and resolve the levels from the convention itself.
    stac = updated_stac_attrs.get("stac_discovery")
    if stac is None:
        return

    reflectance_asset: dict[str, Any] = {
        "href": "/measurements/reflectance",
        "type": "application/vnd.zarr; version=3; profile=multiscales",
        "title": "Surface Reflectance",
        "roles": ["data", "reflectance"],
        "gsd": 10,
    }

    if crs is not None and crs.to_epsg() is not None:
        reflectance_asset.update(utils.proj_attrs_for_crs(crs))

    if reflectance_shape is not None:
        reflectance_asset["proj:shape"] = list(reflectance_shape)

    stac.setdefault("assets", {})["reflectance"] = reflectance_asset


def is_sentinel2_dataset(group: zarr.Group) -> bool:
    if group.metadata.zarr_format == 3:
        return _is_sentinel2_dataset_v3(group)

    from eopf_geozarr.pyz.v2 import GroupSpec

    adapter = TypeAdapter(Sentinel1Root | Sentinel2Root)
    try:
        model = adapter.validate_python(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.warning("Could not validate Sentinel-2 dataset", error=str(e))
        return False

    return isinstance(model, Sentinel2Root)


def _is_sentinel2_dataset_v3(group: zarr.Group) -> bool:
    """Lightweight structural check for Zarr V3 Sentinel-2 stores.

    ``Sentinel2Root`` (and its full-fidelity round-trip validation) is defined
    against the Zarr V2 pydantic model only; there is no V3 counterpart yet.
    Rather than guess at one, this checks the handful of root-level members
    that already distinguish an S2 product from anything else we route on:
    S2 has ``measurements``/``quality``/``conditions`` directly at the root,
    with an S2-specific ``measurements/reflectance`` group underneath.
    Sentinel-1, by contrast, nests those same three groups one level down
    under arbitrary per-polarization product-id groups.
    """
    if set(group.keys()) != {"measurements", "quality", "conditions"}:
        return False

    measurements = group.get("measurements")
    if not isinstance(measurements, zarr.Group):
        return False

    reflectance = measurements.get("reflectance")
    return isinstance(reflectance, zarr.Group) and any(
        isinstance(sub, zarr.Group) for _, sub in reflectance.groups()
    )


class ValidationResult(TypedDict):
    """Result of validating an optimized Sentinel-2 dataset."""

    is_valid: bool
    issues: list[str]
    warnings: list[str]
    summary: dict[str, object]


def validate_optimized_dataset(dataset_path: str) -> ValidationResult:
    """
    Validate an optimized Sentinel-2 dataset.

    Args:
        dataset_path: Path to the optimized dataset

    Returns:
        Validation results dictionary
    """
    return {"is_valid": True, "issues": [], "warnings": [], "summary": {}}

    # Placeholder for validation logic
