"""
Realize the Sentinel-2 GeoZarr plan: the data plane.

`write_geozarr` writes the hierarchy that `eopf_geozarr.data_api.s2_plan`
describes, without xarray's DataTree writer. The planned metadata documents
are written first, with `eopf_geozarr.zarr_io.write_hierarchy`, so every
stored document is exactly the plan; the chunk data is then filled in with zarr-python alone.
The input is read through a small `SourceReader` protocol — a Zarr V2 store
directly, or an xarray DataTree whose variables are read as xarray's zarr
writer would store them (CF-encoded), so a tree built in memory or backed by
a Zarr V3 store feeds the writer without a scratch copy.
Native arrays are copied from the input, and the derived arrays are
block-resampled with the converter's arithmetic: injected bands come from the
finest input level that holds them, generated pyramid levels are chained from
the level before them, measurements and probabilities are reduced by the
nodata-masked block mean (rounded for integer data), classification masks
take the window-centre pixel, and coordinates take the window mean.

The write order is metadata first, per array: a reader that opens the store
mid-write sees complete, valid documents (absent chunks read as the fill
value). Product-level publication — root attributes, consolidated metadata,
the STAC item — is not written here and stays the converter pipeline's final,
separate step, so consumers that discover the product through the root never
see a partially filled store.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import numpy as np
import xarray as xr
import zarr
from pydantic_zarr.v3 import GroupSpec, is_group_spec
from zarr.errors import ContainsArrayAndGroupError, ContainsArrayError, ContainsGroupError
from zarr.storage import FsspecStore, LocalStore

from eopf_geozarr.conversion.encoding_utils import find_fill_value
from eopf_geozarr.data_api.datatree import encoded_variable
from eopf_geozarr.data_api.s2_plan import level_meters, plan_geozarr
from eopf_geozarr.s2_optimization.s2_resampling import determine_variable_type
from eopf_geozarr.zarr_io import read_documents, write_hierarchy

if TYPE_CHECKING:
    from pydantic_zarr.v3 import AnyGroupSpec
    from zarr.abc.store import Store

    from eopf_geozarr.data_api.s2 import Sentinel2Root


class SourceReader(Protocol):
    """Read access to the conversion input, by output-plan path."""

    def values(self, path: str) -> np.ndarray | None:
        """The stored values of the array at `path`, or None if there is none."""
        ...

    def nodata(self, path: str) -> int | None:
        """
        The integer nodata value the input declares for the array at `path`, in
        stored units, or None when it declares none: the CF `_FillValue` (a Zarr
        V2 array's `fill_value`, or the attribute) or the EOPF `fill_value`
        attribute. A resampling mean skips this value and no other: an array
        that declares no nodata is averaged whole, zeros included.
        """
        ...

    def is_group(self, path: str) -> bool:
        """Whether the input holds a group at `path`."""
        ...

    def group_names(self, path: str) -> tuple[str, ...]:
        """The names of the subgroups of the group at `path` (empty if none)."""
        ...


def _integer_nodata(candidates: tuple[object, ...]) -> int | None:
    """The first declared candidate, when it is a whole number (else None)."""
    declared = next((value for value in candidates if value is not None), None)
    if isinstance(declared, (bool, np.bool_)) or not isinstance(
        declared, (int, float, np.integer, np.floating)
    ):
        return None
    return int(declared) if float(declared).is_integer() else None


class ZarrSourceReader:
    """`SourceReader` over the Zarr V2 store a product was read from."""

    def __init__(self, group: zarr.Group) -> None:
        self._group = group

    def _member(self, path: str) -> zarr.Group | zarr.Array | None:
        try:
            return self._group[path]
        except KeyError:
            return None

    def values(self, path: str) -> np.ndarray | None:
        member = self._member(path)
        return np.asarray(member[...]) if isinstance(member, zarr.Array) else None

    def nodata(self, path: str) -> int | None:
        member = self._member(path)
        if not isinstance(member, zarr.Array):
            return None
        # a Zarr V2 array's `fill_value` is what xarray reads as `_FillValue`
        attrs = member.attrs
        return _integer_nodata(
            (member.metadata.fill_value, attrs.get("_FillValue"), attrs.get("fill_value"))
        )

    def is_group(self, path: str) -> bool:
        return isinstance(self._member(path), zarr.Group)

    def group_names(self, path: str) -> tuple[str, ...]:
        member = self._member(path)
        return tuple(member.group_keys()) if isinstance(member, zarr.Group) else ()


class DataTreeSourceReader:
    """
    `SourceReader` over an xarray DataTree (a tree built in memory, or one
    backed by a store in any zarr format). Values are read as xarray's zarr
    writer would store them: CF-encoded, with `_FillValue` applied — the same
    representation `group_spec_from_datatree` documents.
    """

    def __init__(self, tree: xr.DataTree) -> None:
        self._tree = tree

    def _node(self, path: str) -> xr.DataTree | None:
        if not path:
            return self._tree
        try:
            member = self._tree[path]
        except KeyError:
            return None
        return member if isinstance(member, xr.DataTree) else None

    def _variable(self, path: str) -> tuple[str, xr.Variable] | None:
        node_path, _, name = path.rpartition("/")
        node = self._node(node_path)
        if node is None:
            return None
        variables = node.to_dataset(inherit=False).variables
        return (name, variables[name]) if name in variables else None

    def values(self, path: str) -> np.ndarray | None:
        found = self._variable(path)
        return None if found is None else np.asarray(encoded_variable(*found).values)

    def nodata(self, path: str) -> int | None:
        found = self._variable(path)
        if found is None:
            return None
        # `find_fill_value` takes anything with `.encoding` and `.attrs`
        return _integer_nodata((find_fill_value(found[1]),))  # type: ignore[arg-type]

    def is_group(self, path: str) -> bool:
        return self._node(path) is not None

    def group_names(self, path: str) -> tuple[str, ...]:
        node = self._node(path)
        return tuple(node.children) if node is not None else ()


#: Rows per stripe for the streamed block mean (a multiple of the coarsening
#: factor is taken per stripe, so windows never straddle a stripe boundary).
_MEAN_STRIPE_ROWS = 2048


def write_geozarr(
    root: Sentinel2Root, source: zarr.Group | xr.DataTree, store: Store | str
) -> AnyGroupSpec:
    """
    Write the GeoZarr hierarchy planned from `root` to `store`, copying and
    resampling chunk data from `source` — the Zarr V2 input store or the
    xarray DataTree `root` was read from. Returns the plan that was written.
    """
    reader: SourceReader = (
        DataTreeSourceReader(source)
        if isinstance(source, xr.DataTree)
        else ZarrSourceReader(source)
    )
    plan = plan_geozarr(root, coordinate_values=reader.values)
    # the metadata plane: every group and array document, exactly as planned
    if isinstance(store, str):
        # a URL goes through fsspec, anything else is a local path, as zarr
        # resolves a store given as a string
        store = FsspecStore.from_url(store) if "://" in store else LocalStore(store)
    try:
        output = write_hierarchy(plan, store)
    except (ContainsArrayError, ContainsGroupError, ContainsArrayAndGroupError) as error:
        # an output that already holds exactly the plan (a rerun) is reused and
        # its chunk data rewritten; any other hierarchy is refused
        existing = _existing_hierarchy(store)
        if existing is None or existing != plan.model_dump(mode="json"):
            msg = (
                f"The output {store!r} already holds a Zarr hierarchy that differs from "
                "the planned one; remove it before converting"
            )
            raise FileExistsError(msg) from error
        output = zarr.open_group(store, mode="r+", use_consolidated=False)
    _fill_group(plan, reader, output, "")
    return plan


def _existing_hierarchy(store: Store) -> dict[str, object] | None:
    """
    The Zarr V3 group hierarchy at the root of `store`, dumped as JSON, or
    None when the root holds something else (an array, a Zarr V2 node, or
    documents that do not read as a group).
    """
    try:
        group = zarr.open_group(store, mode="r", zarr_format=3, use_consolidated=False)
        spec = GroupSpec.from_stored_metadata_json(read_documents(group))
    except (FileNotFoundError, ValueError, TypeError, ContainsArrayError):
        return None
    return spec.model_dump(mode="json")


def _store_values(output: zarr.Group, path: str, values: np.ndarray) -> None:
    """
    Store `values` in the planned array at `path`. Refuses values whose data
    type does not cast safely to the planned one: zarr casts on assignment,
    which would turn, say, float reflectance into the planned uint16 silently.
    """
    out_array = _output_array(output, path)
    # fixed-length unicode labels fit the variable-length `string` data type
    # they are planned as (#356) without loss, though numpy does not call that
    # cast safe
    text_to_string = values.dtype.kind == "U" and isinstance(out_array.dtype, np.dtypes.StringDType)
    if not text_to_string and not np.can_cast(values.dtype, out_array.dtype, casting="safe"):
        msg = (
            f"The input values for {path!r} have data type {values.dtype}, which does "
            f"not cast safely to the planned data type {out_array.dtype}"
        )
        raise ValueError(msg)
    out_array[...] = values


def _fill_group(spec: AnyGroupSpec, source: SourceReader, output: zarr.Group, path: str) -> None:
    """Fill the data of every array under one planned group (recursively)."""
    for name, member in spec.members.items():
        member_path = f"{path}/{name}".lstrip("/")
        if is_group_spec(member):
            attributes = member.attributes
            if isinstance(attributes, Mapping) and "multiscales" in attributes:
                _fill_pyramid(member, source, output, member_path)
            else:
                _fill_group(member, source, output, member_path)
        else:
            _fill_carried_array(source, output, member_path)


def _fill_carried_array(source: SourceReader, output: zarr.Group, path: str) -> None:
    """Copy one carried array's data from the input."""
    values = source.values(path)
    if values is None:
        msg = f"The input source holds no array at {path!r}"
        raise ValueError(msg)
    if values.dtype.kind == "S":
        # bytes labels are planned as the `string` data type (#356): decode
        # them as ASCII rather than leave the conversion to zarr
        values = np.char.decode(values, "ascii")
    _store_values(output, path, values)


def _fill_pyramid(spec: AnyGroupSpec, source: SourceReader, output: zarr.Group, path: str) -> None:
    """
    Fill one multiscale subtree, finest level first: native levels are copied
    from the input (with their injected bands resampled from the finest
    input level that holds them), and generated levels are chained from the
    data of the level before them, as the converter chains them.
    """
    levels = sorted(spec.members, key=_level_meters)
    previous_data: dict[str, np.ndarray] = {}
    # the nodata each variable's input declares, carried down the chain
    nodata: dict[str, int | None] = {}
    previous_coords: dict[str, np.ndarray] = {}
    previous_meters: int | None = None
    for level_name in levels:
        level_spec = spec.members[level_name]
        if not is_group_spec(level_spec):
            msg = f"Planned pyramid member {path}/{level_name} is not a group"
            raise TypeError(msg)
        meters = _level_meters(level_name)
        level_path = f"{path}/{level_name}"
        native = source.is_group(level_path)
        data: dict[str, np.ndarray] = {}
        coords: dict[str, np.ndarray] = {}
        for var in level_spec.members:
            if var in ("x", "y"):
                coords[var] = _level_coordinate(
                    source,
                    f"{level_path}/{var}",
                    previous_coords.get(var),
                    meters,
                    previous_meters,
                    native=native,
                )
            else:
                data[var] = _level_variable(
                    nodata,
                    source,
                    var,
                    path,
                    level_path,
                    meters,
                    previous_meters,
                    previous_data,
                    native=native,
                )
        for var, values in {**data, **coords}.items():
            _store_values(output, f"{level_path}/{var}", values)
        previous_data, previous_coords, previous_meters = data, coords, meters


def _level_variable(
    nodata: dict[str, int | None],
    source: SourceReader,
    var: str,
    base_path: str,
    level_path: str,
    meters: int,
    previous_meters: int | None,
    previous_data: dict[str, np.ndarray],
    *,
    native: bool,
) -> np.ndarray:
    """
    The data of one pyramid-level variable: copied, injected, or chained.
    `nodata` records, per variable, the nodata its input declares (from the
    level it was last read from), which the resampling mean skips.
    """
    if native:
        values = source.values(f"{level_path}/{var}")
        if values is not None:
            nodata[var] = source.nodata(f"{level_path}/{var}")
            return values
        # an injected band: resample from the finest input level that holds it
        injected = _finest_input_source(source, base_path, var, meters)
        if injected is None:
            msg = f"No input level finer than {meters} m holds the injected band {var!r}"
            raise ValueError(msg)
        src_values, src_meters, src_path = injected
        nodata[var] = source.nodata(src_path)
        return _coarsen(src_values, meters // src_meters, var, nodata[var])
    if previous_meters is None:
        msg = f"The finest planned level {level_path} is missing from the input source"
        raise ValueError(msg)
    if var not in previous_data:
        msg = f"Variable {var!r} of generated level {level_path} is missing from the previous level"
        raise ValueError(msg)
    return _coarsen(previous_data[var], meters // previous_meters, var, nodata.get(var))


def _level_coordinate(
    source: SourceReader,
    coord_path: str,
    previous: np.ndarray | None,
    meters: int,
    previous_meters: int | None,
    *,
    native: bool,
) -> np.ndarray:
    """A level's x/y values: the input's at native levels, window means after."""
    if native:
        values = source.values(coord_path)
        if values is None:
            msg = f"The input source holds no coordinate array at {coord_path!r}"
            raise ValueError(msg)
        return values
    if previous is None or previous_meters is None:
        msg = f"No previous level to derive the coordinate at {coord_path!r} from"
        raise ValueError(msg)
    factor = meters // previous_meters
    size = previous.shape[0] // factor
    return previous[: size * factor].reshape(size, factor).mean(axis=1)


def _coarsen(values: np.ndarray, factor: int, var: str, fill: int | None) -> np.ndarray:
    """
    Downsample one variable by `factor`, with the converter's arithmetic:
    classification masks take the window-centre pixel; everything else takes
    the block mean — NaN-skipping for floats, nodata-masked and rounded for
    integers (an all-nodata window keeps the fill value). `fill` is the
    nodata the input declares; with None, integers are averaged whole.
    """
    kind = determine_variable_type(var)
    rows = values.shape[0] // factor * factor
    cols = values.shape[1] // factor * factor
    trimmed = values[:rows, :cols]
    if kind == "classification":
        offset = factor // 2
        return trimmed[offset::factor, offset::factor]
    if kind not in ("reflectance", "probability"):
        msg = f"Unknown/Unapplicable variable type {kind} for {var!r}"
        raise ValueError(msg)
    out = np.empty((rows // factor, cols // factor), dtype=values.dtype)
    stripe = factor * max(1, _MEAN_STRIPE_ROWS // factor)
    for row in range(0, rows, stripe):
        block = trimmed[row : row + stripe]
        windows = block.reshape(block.shape[0] // factor, factor, cols // factor, factor)
        out[row // factor : (row + block.shape[0]) // factor] = _window_mean(windows, fill)
    return out


def _window_mean(windows: np.ndarray, fill: int | None) -> np.ndarray:
    """The converter's block-mean reduction over one (i, factor, j, factor) view."""
    if np.issubdtype(windows.dtype, np.floating):
        # float values: nodata is NaN, the NaN-skipping mean excludes it, and
        # an all-NaN window stays NaN (as xarray's skipna mean leaves it)
        total = np.nansum(windows.astype("float64"), axis=(1, 3))
        count = np.sum(~np.isnan(windows), axis=(1, 3))
        mean = total / np.maximum(count, 1)
        return np.where(count > 0, mean, np.nan).astype(windows.dtype)
    as_float = windows.astype("float64")
    if fill is None:
        return np.round(as_float.mean(axis=(1, 3))).astype(windows.dtype)
    # raw integers: mask nodata, average, round (a plain cast truncates), and
    # keep the fill where a window holds nothing but nodata
    mask = windows != fill
    total = np.where(mask, as_float, 0.0).sum(axis=(1, 3))
    count = mask.sum(axis=(1, 3))
    mean = np.round(total / np.maximum(count, 1))
    return np.where(count > 0, mean, fill).astype(windows.dtype)


def _finest_input_source(
    source: SourceReader, base_path: str, var: str, meters: int
) -> tuple[np.ndarray, int, str] | None:
    """The finest input level of this subtree holding `var`, with its metres and path."""
    candidates = sorted(
        (name for name in source.group_names(base_path) if _is_level_name(name)),
        key=_level_meters,
    )
    for name in candidates:
        src_meters = _level_meters(name)
        if src_meters >= meters:
            return None
        values = source.values(f"{base_path}/{name}/{var}")
        if values is not None:
            return values, src_meters, f"{base_path}/{name}/{var}"
    return None


def _level_meters(name: str) -> int:
    """The resolution in metres of a level group the plan names (`level_meters`)."""
    meters = level_meters(name)
    if meters is None:
        msg = f"{name!r} is not a resolution level name"
        raise ValueError(msg)
    return meters


def _is_level_name(name: str) -> bool:
    return level_meters(name) is not None


def _output_array(output: zarr.Group, path: str) -> zarr.Array:
    member = output[path]
    if not isinstance(member, zarr.Array):
        msg = f"The planned store holds no array at {path!r}"
        raise TypeError(msg)
    return member
