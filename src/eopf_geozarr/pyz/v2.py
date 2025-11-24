"""
Zarr V2 models

The generic type parameter TMembers allows the use of typeddicts to more explicitly model
the key structure of the members of a zarr group:

```python
from typing import TypedDict

class MyMembers(TypedDict, closed=True):
    a: GroupSpec[Any, Any]
    b: GroupSpec[Any, Any]

# this class will enforce that the members have keys "a" and "b"
class MyGroup(GroupSpec[Any, MyMembers])
   ...
```
"""

from __future__ import annotations

from typing import Any, Generic, Mapping, TypeAlias, TypeVar, Union

from pydantic_zarr.v2 import ArraySpec as ArraySpecV2
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2

from eopf_geozarr.pyz.common import (
    TBaseAttr,
    _format_array_html,
    format_html_repr,
    format_text_repr,
    get_member_names,
)

TBaseMember: TypeAlias = Mapping[str, Union["GroupSpec[Any, Any]", "ArraySpec[Any]"]]

TAttr = TypeVar("TAttr", bound=TBaseAttr)
TMembers = TypeVar("TMembers", bound=TBaseMember)
TArraySpecType = TypeVar("TArraySpecType")


class GroupSpec(GroupSpecV2[TAttr, TMembers]):
    attributes: TAttr
    members: TMembers

    def __repr__(self) -> str:
        """Return a condensed text representation of the GroupSpec."""
        class_name = self.__class__.__name__
        member_names = get_member_names(self.members)
        return format_text_repr(class_name, member_names)

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter/IPython."""
        class_name = self.__class__.__name__
        member_names = get_member_names(self.members)
        members_dict = {k: v for k, v in self.members.items()} if self.members else None
        return format_html_repr(
            class_name,
            member_names,
            members_dict=members_dict,
            attributes=self.attributes,
        )


class ArraySpec(ArraySpecV2[Any], Generic[TArraySpecType]):
    """Zarr V2 ArraySpec with enhanced HTML representation for Jupyter/IPython."""

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter/IPython."""
        return _format_array_html(self)
