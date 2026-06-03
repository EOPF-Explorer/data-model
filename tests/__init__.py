"""Tests for the eopf-geozarr package.

Verification helpers live in :mod:`tests.conftest` and are re-exported here so
integration tests can ``from tests import _verify_basic_structure`` etc.
"""

from .conftest import (
    _verify_basic_structure,
    _verify_geozarr_spec_compliance,
    _verify_multiscale_structure,
    _verify_rgb_data_access,
)

__all__ = [
    "_verify_basic_structure",
    "_verify_geozarr_spec_compliance",
    "_verify_multiscale_structure",
    "_verify_rgb_data_access",
]
