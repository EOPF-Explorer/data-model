"""Pipeline routing for DataTree products handed to the CPM writer plugin.

CPM products carry their identity in the ``stac_discovery`` root attributes
(see ``StacProductConvention`` in eopf-cpm), so routing prefers the declared
``product:type`` and only falls back to structural inspection when the
attribute is absent. Routing must work on any ``xarray.DataTree`` regardless
of backend: trees produced by the CPM SAFE reader are built in memory and
have no backing zarr store to introspect.

This module deliberately has no dependency on the ``eopf`` package so it can
be imported and tested without eopf-cpm installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import xarray as xr

PipelineName = Literal["s2-optimized", "s3-olci-optimized", "generic"]

#: CPM product types routed to the Sentinel-2 optimized pipeline. Deliberately
#: only the L1C/L2A imagery products: other MSI product types (S02MSIL0_,
#: S02MSIRAW, ...) do not have the reflectance pyramid structure the optimized
#: pipeline expects.
S2_PRODUCT_TYPE_PREFIXES = ("S02MSIL1C", "S02MSIL2A")

#: CPM product types routed to the Sentinel-3 OLCI optimized pipeline.
#: Deliberately only the L1 EFR/ERR radiance products: the L2 products
#: (LRR, LFR, ...) do not have the flat oa01..oa21 radiance layout the
#: optimized pipeline expects.
S3_OLCI_PRODUCT_TYPE_PREFIXES = ("S03OLCEFR", "S03OLCERR")

#: Native Sentinel-2 resolution groups expected under measurements/reflectance.
_S2_NATIVE_RESOLUTIONS = frozenset({"r10m", "r20m", "r60m"})

#: First OLCI radiance band, used as a cheap structural marker for the
#: fallback detector. Matches ``OLCI_BANDS[0]`` in
#: ``s3_olci_optimization.olci_band_mapping``, duplicated here (rather than
#: imported) to keep this module import-light and independently testable.
_OLCI_FIRST_BAND = "oa01_radiance"


def product_type_of(dtree: xr.DataTree) -> str | None:
    """
    Return the CPM product type declared in the root STAC attributes, if any.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.

    Returns
    -------
    str | None
        The value of ``stac_discovery.properties["product:type"]`` (e.g.
        ``"S02MSIL1C"``), or None when the attribute chain is absent.
    """
    stac = dtree.attrs.get("stac_discovery")
    if not isinstance(stac, Mapping):
        return None
    properties = stac.get("properties")
    if not isinstance(properties, Mapping):
        return None
    product_type = properties.get("product:type")
    if isinstance(product_type, str):
        return product_type
    return None


def looks_like_sentinel2(dtree: xr.DataTree) -> bool:
    """
    Report whether a DataTree looks like a Sentinel-2 L1C/L2A product.

    The declared CPM product type wins when present; otherwise the tree is
    inspected for the native ``measurements/reflectance/r{10,20,60}m`` groups.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.

    Returns
    -------
    bool
    """
    product_type = product_type_of(dtree)
    if product_type is not None:
        return product_type.startswith(S2_PRODUCT_TYPE_PREFIXES)

    measurements = dtree.children.get("measurements")
    if measurements is None:
        return False
    reflectance = measurements.children.get("reflectance")
    if reflectance is None:
        return False
    return set(reflectance.children) >= _S2_NATIVE_RESOLUTIONS


def looks_like_sentinel3_olci(dtree: xr.DataTree) -> bool:
    """
    Report whether a DataTree looks like a Sentinel-3 OLCI L1 EFR/ERR product.

    The declared CPM product type wins when present; otherwise the tree is
    inspected for the native flat ``measurements/oa01_radiance`` band, the
    same structural marker :func:`eopf_geozarr.s3_olci_optimization.olci_converter.is_sentinel3_olci_dataset`
    uses, checked directly on the DataTree so this works on in-memory trees
    with no backing zarr store. Unlike the Sentinel-2 pyramid groups,
    ``oa01_radiance`` is a *data variable* of the ``measurements`` node
    (a zarr array member of that group), not a child node.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.

    Returns
    -------
    bool
    """
    product_type = product_type_of(dtree)
    if product_type is not None:
        return product_type.startswith(S3_OLCI_PRODUCT_TYPE_PREFIXES)

    measurements = dtree.children.get("measurements")
    if measurements is None:
        return False
    return _OLCI_FIRST_BAND in measurements.data_vars


def select_pipeline(
    dtree: xr.DataTree,
    *,
    force: PipelineName | None = None,
) -> PipelineName:
    """
    Select the conversion pipeline for a product.

    Parameters
    ----------
    dtree : xr.DataTree
        Product root node.
    force : PipelineName, optional
        When given, short-circuits auto-detection and returns this pipeline
        unconditionally. None (default) auto-detects: Sentinel-2 first,
        then Sentinel-3 OLCI, falling back to the generic pipeline.

    Returns
    -------
    PipelineName
        ``"s2-optimized"``, ``"s3-olci-optimized"``, or ``"generic"``.
    """
    if force is not None:
        return force
    if looks_like_sentinel2(dtree):
        return "s2-optimized"
    if looks_like_sentinel3_olci(dtree):
        return "s3-olci-optimized"
    return "generic"
