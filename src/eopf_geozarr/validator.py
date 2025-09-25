from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import xarray as xr

from .conversion.fs_utils import get_storage_options


@dataclass(slots=True)
class ValidationIssue:
    group: str
    variable: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    total_variables: int
    compliant_variables: int
    issues: Tuple[ValidationIssue, ...]

    @property
    def is_compliant(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        if self.is_compliant:
            return "GeoZarr store is compliant"
        return f"Non-compliant variables: {len(self.issues)}"

    def detailed(self) -> Iterable[str]:
        for issue in self.issues:
            yield f"[{issue.group}] {issue.variable}: {issue.message}"


def validate_geozarr_store(path: str) -> ValidationReport:
    storage_options = get_storage_options(path)
    dt = xr.open_datatree(
        path, engine="zarr", chunks="auto", storage_options=storage_options
    )
    try:
        issues: List[ValidationIssue] = []
        total = 0
        compliant = 0
        for group_name, group in dt.children.items():
            if not hasattr(group, "data_vars"):
                continue
            for var_name, var in group.data_vars.items():
                total += 1
                missing = []
                attrs = getattr(var, "attrs", {}) or {}
                if "_ARRAY_DIMENSIONS" not in attrs:
                    missing.append("_ARRAY_DIMENSIONS")
                if "standard_name" not in attrs:
                    missing.append("standard_name")
                if "grid_mapping" not in attrs and "grid_mapping_name" not in attrs:
                    missing.append("grid_mapping")
                if missing:
                    issues.append(
                        ValidationIssue(
                            group=group_name,
                            variable=var_name,
                            message="Missing attributes: " + ", ".join(missing),
                        )
                    )
                else:
                    compliant += 1
        return ValidationReport(
            total_variables=total,
            compliant_variables=compliant,
            issues=tuple(issues),
        )
    finally:
        try:
            dt.close()
        except AttributeError:
            pass
