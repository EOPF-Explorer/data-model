"""Data API for accessing GeoZarr compliant EOPF datasets."""

from .transactions import TransactionsClient, ensure_collection

__all__ = ["TransactionsClient", "ensure_collection"]
