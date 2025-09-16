import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import xarray as xr


def _optimized_tree_html(dt: xr.DataTree) -> str:  # condensed from original
    def has_content(node: Any) -> bool:
        ds: xr.Dataset | None = getattr(node, "ds", None)
        if ds is not None:
            if getattr(ds, "data_vars", None) and len(ds.data_vars) > 0:
                return True
            if getattr(ds, "attrs", None):
                return True
        children: MutableMapping[str, Any] | None = getattr(node, "children", None)
        if children:
            return any(has_content(c) for c in children.values())
        return False

    def format_data_vars(data_vars: Mapping[str, Any]) -> str:
        if not data_vars:
            return ""
        try:
            html_repr = xr.Dataset(data_vars)._repr_html_()
            return f'<div class="xarray-variables">{html_repr}</div>'
        except Exception:
            return "".join(
                f"<div class='tree-variable'><span class='var-name'>{name}</span></div>"
                for name in data_vars
            )

    def format_attrs(attrs: Mapping[str, Any]) -> str:
        if not attrs:
            return ""
        items: Sequence[tuple[str, Any]] = list(attrs.items())[:5]
        parts: list[str] = []
        for k, v in items:
            vstr = str(v)
            if len(vstr) > 50:
                vstr = vstr[:47] + "..."
            parts.append(
                f"<div class='tree-attribute'><span class='attr-key'>{k}:</span><span class='attr-value'>{vstr}</span></div>"
            )
        if len(attrs) > 5:
            parts.append(f"<div class='tree-attribute-more'>... and {len(attrs) - 5} more</div>")
        return "".join(parts)

    def render(node: Any, path: str = "", level: int = 0) -> str:
        if not has_content(node):
            return ""
        node_name = path.split("/")[-1] if path else "root"
        has_data = getattr(node, "ds", None) is not None and getattr(node.ds, "data_vars", None)
        data_vars_count = len(node.ds.data_vars) if has_data else 0
        attrs_count = len(node.ds.attrs) if has_data and node.ds.attrs else 0
        children: Iterable[Any] = [
            c for c in getattr(node, "children", {}).values() if has_content(c)
        ]
        summary: list[str] = []
        if data_vars_count:
            summary.append(f"{data_vars_count} variables")
        if attrs_count:
            summary.append(f"{attrs_count} attributes")
        if children:
            child_list = list(children)
            summary.append(f"{len(child_list)} subgroups")
        summary_txt = " • ".join(summary) if summary else "empty group"
        html: list[str] = [
            f"<div class='tree-node' style='margin-left:{level * 20}px;'>",
            f"<details class='tree-details' {'open' if level < 2 else ''}><summary class='tree-summary'><span class='tree-icon'>{'📁' if list(children) else '📄'}</span><span class='tree-name'>{node_name}</span><span class='tree-info'>({summary_txt})</span></summary><div class='tree-content'>",
        ]
        if has_data:
            html.append(
                "<div class='tree-section'><h4 class='section-title'>Variables</h4><div class='tree-variables'>"
                + format_data_vars(getattr(node.ds, "data_vars"))
                + "</div></div>"
            )
            if getattr(node.ds, "attrs"):
                html.append(
                    "<div class='tree-section'><h4 class='section-title'>Attributes</h4><div class='tree-attributes'>"
                    + format_attrs(getattr(node.ds, "attrs"))
                    + "</div></div>"
                )
        if children:
            html.append(
                "<div class='tree-section'><h4 class='section-title'>Subgroups</h4><div class='tree-children'>"
            )
            for cname, cnode in getattr(node, "children", {}).items():
                cpath = f"{path}/{cname}" if path else cname
                ch = render(cnode, cpath, level + 1)
                if ch:
                    html.append(ch)
            html.append("</div></div>")
        html.append("</div></details></div>")
        return "".join(html)

    tree_content = render(dt)
    return f"<div class='optimized-tree'>{tree_content}</div>"


def generate_html_output(
    dt: xr.DataTree, output_path: str, input_path: str, verbose: bool = False
) -> None:
    try:
        tree_html = _optimized_tree_html(dt)
        html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>DataTree - {Path(input_path).name}</title></head><body>{tree_html}</body></html>"
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")
        print(f"HTML visualization generated: {out}")
        if verbose:
            print(f"   File size: {out.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"Error generating HTML output: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
