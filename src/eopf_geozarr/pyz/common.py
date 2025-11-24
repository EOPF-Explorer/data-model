from collections.abc import Mapping
from html import escape
from typing import Any, TypeAlias

from pydantic import BaseModel

TBaseAttr: TypeAlias = Mapping[str, object] | BaseModel


def get_member_names(members: Any) -> list[str]:
    """Get list of member names from members dict/mapping."""
    if members is None:
        return []
    if isinstance(members, Mapping):
        return list(members.keys())
    return []


def format_text_repr(
    class_name: str, member_names: list[str], max_members: int = 5
) -> str:
    """Format a condensed text representation of a GroupSpec.

    Args:
        class_name: Name of the GroupSpec class
        member_names: List of member names
        max_members: Maximum number of members to display before truncating

    Returns:
        Formatted text representation
    """
    if not member_names:
        members_str = "(empty)"
    elif len(member_names) <= max_members:
        members_str = ", ".join(member_names)
    else:
        displayed = ", ".join(member_names[:max_members])
        remaining = len(member_names) - max_members
        members_str = f"{displayed}, +{remaining} more"

    return f"{class_name}(members={{{members_str}}})"


def _get_array_info(arr: Any) -> str:
    """Get concise info about an array or ArraySpec."""
    # ArraySpec objects (from pydantic-zarr) have shape and either dtype or data_type
    if hasattr(arr, "shape"):
        shape_tuple = str(arr.shape)  # e.g., "(100, 200)"

        # Try to get dtype first (V2 or real arrays)
        dtype = getattr(arr, "dtype", None)
        if dtype is None:
            # Try data_type (V3 ArraySpec)
            dtype = getattr(arr, "data_type", None)

        if dtype is None:
            return f"Array{shape_tuple}"
        return f"Array{shape_tuple} dtype={dtype}"

    return type(arr).__name__


def _format_array_html(arr: Any) -> str:
    """Format an array/ArraySpec as HTML with expandable metadata and attributes.

    Shows all array properties in key-value format with attributes as an expandable section.
    """
    if not hasattr(arr, "shape"):
        return f"<div><i>{type(arr).__name__}</i></div>"

    class_name = type(arr).__name__
    html_parts = [
        "<div>",
        f"<div style='font-weight: bold; margin-bottom: 8px;'>{class_name}</div>",
    ]

    # Build metadata dict for display - include all available properties
    metadata = {}

    # Define the order of properties to display (most important first)
    property_names = [
        "zarr_format",
        "shape",
        "dtype",
        "data_type",
        "chunks",
        "order",
        "dimension_separator",
        "fill_value",
        "compressor",
        "filters",
    ]

    for prop_name in property_names:
        value = getattr(arr, prop_name, None)

        # Skip None/empty values
        if value is None:
            continue

        # Handle dtype specially - may come from dtype or data_type
        if prop_name == "dtype":
            if value is None:
                value = getattr(arr, "data_type", None)
            if value is None:
                continue
            dtype_str = str(value).strip()
            value_str = dtype_str if dtype_str else "(not set)"
        # Skip data_type if we already handled it via dtype
        elif prop_name == "data_type":
            if getattr(arr, "dtype", None) is not None:
                continue
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
        # Skip empty/None filters and compressor
        elif prop_name in ("filters", "compressor"):
            if value is None:
                continue
            value_str = str(value)
        else:
            # For all other properties, show them
            value_str = repr(value) if prop_name == "fill_value" else str(value)

        if "value_str" in locals():
            metadata[prop_name] = value_str
            del value_str

    # Display metadata in key-value format
    for key, value in metadata.items():
        escaped_value = escape(str(value))
        html_parts.append(
            f"<div>• {key}: <code style='background: none; padding: 0;'>{escaped_value}</code></div>"
        )

    # Get attributes if available
    attributes = getattr(arr, "attributes", None)

    # Show attributes if present and non-empty
    # Handle both dict and Pydantic BaseModel
    if attributes:
        is_dict_attrs = isinstance(attributes, dict) and len(attributes) > 0
        is_model_attrs = isinstance(attributes, BaseModel)

        if is_dict_attrs or is_model_attrs:
            html_parts.append(
                "<details style='margin-top: 8px;'>"
                "<summary style='cursor: pointer;'>attributes</summary>"
                "<div style='margin-left: 20px; margin-top: 5px;'>"
            )

            # Get items based on type
            if is_dict_attrs:
                attrs_dict = attributes
                items = list(attrs_dict.items())
            else:  # is_model_attrs
                attrs_dict = attributes.model_dump()
                items = list(attrs_dict.items())

            for key, value in items:
                if isinstance(value, dict):
                    # Render dicts as expandable
                    value_summary = escape(_format_value_repr(value))
                    html_parts.append(
                        f"<details style='margin-bottom: 5px;'>"
                        f"<summary style='cursor: pointer;'>{key}: <code style='background: none; padding: 0;'>{value_summary}</code></summary>"
                    )
                    html_parts.append(_render_dict_html(value))
                    html_parts.append("</details>")
                elif isinstance(value, list):
                    # Render lists as expandable
                    list_summary = f"list[{len(value)}]"
                    html_parts.append(
                        f"<details style='margin-bottom: 5px;'>"
                        f"<summary style='cursor: pointer;'>{key}: <code style='background: none; padding: 0;'>{list_summary}</code></summary>"
                    )
                    html_parts.append(_render_list_html(value))
                    html_parts.append("</details>")
                else:
                    value_repr = escape(repr(value))
                    html_parts.append(
                        f"<div>• {key}: <code style='background: none; padding: 0;'>{value_repr}</code></div>"
                    )

            html_parts.extend(
                [
                    "</div>",
                    "</details>",
                ]
            )

    html_parts.append("</div>")

    return "".join(html_parts)


def _get_attributes_repr(attributes: Any) -> str:
    """Get a useful repr for attributes object."""
    if attributes is None:
        return "(no attributes)"

    if isinstance(attributes, BaseModel):
        # For Pydantic models, show the model dump
        fields = (
            attributes.model_fields_set
            if hasattr(attributes, "model_fields_set")
            else set()
        )
        if fields:
            return f"{type(attributes).__name__}({', '.join(sorted(fields))})"
        else:
            return f"{type(attributes).__name__}()"

    if isinstance(attributes, Mapping):
        keys = list(attributes.keys())[:5]
        if len(keys) < len(attributes):
            keys_str = ", ".join(keys) + f", +{len(attributes) - len(keys)} more"
        else:
            keys_str = ", ".join(keys)
        return f"{{{keys_str}}}"

    return type(attributes).__name__


def _format_value_repr(value: Any) -> str:
    """Format a value for display in HTML repr (summary only).

    Returns a brief representation - dict keys, list length, or scalar value.
    """
    if isinstance(value, dict):
        keys = list(value.keys())[:3]
        if len(keys) < len(value):
            keys_str = ", ".join(keys) + f", +{len(value) - len(keys)} more"
        else:
            keys_str = ", ".join(keys)
        return f"{{{keys_str}}}"
    elif isinstance(value, list):
        return f"list[{len(value)}]"
    else:
        return repr(value)


def _render_list_html(lst: list[Any]) -> str:
    """Render a list as HTML with expandable items."""
    if not lst:
        return ""

    html_parts = ["<div style='margin-left: 20px; margin-top: 5px;'>"]

    items = lst[:15]  # Show up to 15 items
    for i, item in enumerate(items):
        if isinstance(item, dict):
            # Render nested dicts as expandable
            dict_summary = _format_value_repr(item)
            escaped_summary = escape(dict_summary)
            html_parts.append(
                f"<details style='margin-bottom: 5px;'>"
                f"<summary style='cursor: pointer;'>[{i}]: <code style='background: none; padding: 0;'>{escaped_summary}</code></summary>"
            )
            html_parts.append(_render_dict_html(item))
            html_parts.append("</details>")
        elif isinstance(item, list):
            # Render nested lists as expandable
            list_summary = f"list[{len(item)}]"
            html_parts.append(
                f"<details style='margin-bottom: 5px;'>"
                f"<summary style='cursor: pointer;'>[{i}]: <code style='background: none; padding: 0;'>{list_summary}</code></summary>"
            )
            html_parts.append(_render_list_html(item))
            html_parts.append("</details>")
        else:
            item_repr = escape(repr(item))
            html_parts.append(
                f"<div>• [{i}]: <code style='background: none; padding: 0;'>{item_repr}</code></div>"
            )

    if len(lst) > 15:
        remaining = len(lst) - 15
        html_parts.append(f"<div><i>+{remaining} more items</i></div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def _render_dict_html(d: dict[str, Any], indent: int = 0) -> str:
    """Render a dictionary as HTML with expandable key-value pairs."""
    if not d:
        return ""

    html_parts = ["<div style='margin-left: 20px; margin-top: 5px;'>"]

    items = list(d.items())[:15]  # Show up to 15 items
    for key, value in items:
        if isinstance(value, dict):
            # Render nested dicts as expandable
            value_summary = _format_value_repr(value)
            escaped_summary = escape(value_summary)
            html_parts.append(
                f"<details style='margin-bottom: 5px;'>"
                f"<summary style='cursor: pointer;'>{key}: <code style='background: none; padding: 0;'>{escaped_summary}</code></summary>"
            )
            html_parts.append(_render_dict_html(value, indent + 1))
            html_parts.append("</details>")
        elif isinstance(value, list):
            # Render lists as expandable
            list_summary = f"list[{len(value)}]"
            html_parts.append(
                f"<details style='margin-bottom: 5px;'>"
                f"<summary style='cursor: pointer;'>{key}: <code style='background: none; padding: 0;'>{list_summary}</code></summary>"
            )
            html_parts.append(_render_list_html(value))
            html_parts.append("</details>")
        else:
            value_repr = escape(repr(value))
            html_parts.append(
                f"<div>• {key}: <code style='background: none; padding: 0;'>{value_repr}</code></div>"
            )

    if len(d) > 15:
        remaining = len(d) - 15
        html_parts.append(f"<div><i>+{remaining} more items</i></div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def format_html_repr(
    class_name: str,
    member_names: list[str],
    members_dict: dict[str, Any] | None = None,
    attributes: Any = None,
    indent_level: int = 0,
) -> str:
    """Format an HTML tree representation of a GroupSpec for Jupyter/IPython.

    Args:
        class_name: Name of the GroupSpec class
        member_names: List of member names
        members_dict: Optional dict of actual member objects for recursive repr
        attributes: Optional attributes object to display
        indent_level: Current indentation level for recursion

    Returns:
        HTML string with collapsible tree structure
    """
    indent = indent_level * 20

    html_parts = [
        f"<div style='font-family: monospace; margin: 5px 0; margin-left: {indent}px;'>",
        "<details open>",
        f"<summary style='cursor: pointer; font-weight: bold; user-select: none;'>{class_name}</summary>",
        "<div style='margin-left: 20px; margin-top: 8px;'>",
    ]

    # Show attributes section if present
    if attributes is not None:
        attrs_repr = _get_attributes_repr(attributes)
        html_parts.append(
            "<details style='margin-bottom: 10px;'>"
            "<summary style='cursor: pointer; font-style: italic;'>attributes: "
            + attrs_repr
            + "</summary>"
            "<div style='margin-left: 20px; margin-top: 5px;'>"
        )

        if isinstance(attributes, Mapping):
            attrs_list = list(attributes.keys())[:10]
            for attr_name in attrs_list:
                attr_value = attributes[attr_name]
                value_repr = escape(_format_value_repr(attr_value))
                html_parts.append(
                    f"<div>• {attr_name}: <code style='background: none; padding: 0;'>{value_repr}</code></div>"
                )
            if len(attributes) > 10:
                remaining = len(attributes) - 10
                html_parts.append(f"<div><i>+{remaining} more attributes</i></div>")
        elif isinstance(attributes, BaseModel):
            # Show Pydantic model fields with values using model_dump
            attrs_dict = attributes.model_dump()
            attrs_items = list(attrs_dict.items())[:10]
            for field_name, field_value in attrs_items:
                if isinstance(field_value, dict):
                    # Render dicts as expandable sections
                    value_summary = escape(_format_value_repr(field_value))
                    html_parts.append(
                        f"<details style='margin-bottom: 8px;'>"
                        f"<summary style='cursor: pointer;'>{field_name}: <code style='background: none; padding: 0;'>{value_summary}</code></summary>"
                    )
                    html_parts.append(_render_dict_html(field_value))
                    html_parts.append("</details>")
                elif isinstance(field_value, list):
                    # Render lists as expandable sections
                    list_summary = f"list[{len(field_value)}]"
                    html_parts.append(
                        f"<details style='margin-bottom: 8px;'>"
                        f"<summary style='cursor: pointer;'>{field_name}: <code style='background: none; padding: 0;'>{list_summary}</code></summary>"
                    )
                    html_parts.append(_render_list_html(field_value))
                    html_parts.append("</details>")
                else:
                    value_repr = escape(_format_value_repr(field_value))
                    html_parts.append(
                        f"<div>• {field_name}: <code style='background: none; padding: 0;'>{value_repr}</code></div>"
                    )
            if len(attrs_dict) > 10:
                remaining = len(attrs_dict) - 10
                html_parts.append(f"<div><i>+{remaining} more fields</i></div>")
        else:
            html_parts.append(f"<div>{type(attributes).__name__}</div>")

        html_parts.extend(
            [
                "</div>",
                "</details>",
            ]
        )

    # Show members section
    if not member_names:
        html_parts.append("<div><i>(no members)</i></div>")
    else:
        html_parts.append(
            "<details style='margin-bottom: 10px;'>"
            "<summary style='cursor: pointer; font-style: italic;'>members</summary>"
            "<div style='margin-left: 20px; margin-top: 5px;'>"
        )

        for member_name in member_names:
            member_obj = None
            if members_dict and member_name in members_dict:
                member_obj = members_dict[member_name]

            # Check if member is a GroupSpec (has _repr_html_ method)
            if member_obj is not None and hasattr(member_obj, "_repr_html_"):
                # Recursively render nested group
                html_parts.append(
                    f"<div style='margin: 8px 0;'><b>{member_name}</b>:</div>"
                )
                # Add indented nested repr
                nested_html = member_obj._repr_html_()
                # Wrap nested content with proper indentation
                html_parts.append(
                    f"<div style='margin-left: 20px;'>{nested_html}</div>"
                )
            else:
                # Render as simple item - arrays get detailed html, others get simple text
                if member_obj is not None:
                    if hasattr(member_obj, "shape"):
                        # It's an array or ArraySpec - use detailed HTML repr
                        html_parts.append(
                            f"<div style='margin: 5px 0;'><b>{member_name}</b>:</div>"
                        )
                        array_html = _format_array_html(member_obj)
                        html_parts.append(
                            f"<div style='margin-left: 20px;'>{array_html}</div>"
                        )
                    else:
                        # Unknown object - show type name
                        member_info = _get_array_info(member_obj)
                        html_parts.append(
                            f"<div>• {member_name}: <i>{member_info}</i></div>"
                        )
                else:
                    html_parts.append(f"<div>• {member_name}: <i>Unknown</i></div>")

        html_parts.extend(
            [
                "</div>",
                "</details>",
            ]
        )

    html_parts.extend(
        [
            "</div>",
            "</details>",
            "</div>",
        ]
    )

    return "".join(html_parts)
