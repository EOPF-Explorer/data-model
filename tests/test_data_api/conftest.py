from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from zarr import open_group
from zarr.core.buffer import default_buffer_prototype

if TYPE_CHECKING:
    from typing import Any

    from zarr import Group


@pytest.fixture
def example_group() -> Group:
    meta = _load_geozarr_file("sentinel_2.json")
    example_group = open_group(
        store={
            "zarr.json": default_buffer_prototype().buffer.from_bytes(
                json.dumps(meta).encode("utf-8")
            )
        },
        mode="r",
    )
    return example_group


def _load_geozarr_file(filename: str) -> dict[str, Any]:
    """Load an example Geozarr group metadata file from the geozarr_examples directory."""
    examples_dir = Path(__file__).parent / "geozarr_examples"
    file_path = examples_dir / filename
    with open(file_path, "r") as f:
        return json.load(f)


def _load_projjson_file(filename: str) -> dict[str, Any]:
    """Load a PROJ JSON file from the projjson_examples directory."""
    examples_dir = Path(__file__).parent / "projjson_examples"
    file_path = examples_dir / filename
    with open(file_path, "r") as f:
        return json.load(f)


def _load_sentinel1_examples() -> tuple[dict[str, object], ...]:
    examples_dir = Path(__file__).parent / "s1_examples"
    return tuple(
        json.loads((examples_dir / filename).read_text())
        for filename in sorted(examples_dir.glob("*.json"))
    )


def _load_sentinel2_examples() -> tuple[dict[str, object], ...]:
    examples_dir = Path(__file__).parent / "s2_examples"
    return tuple(
        json.loads((examples_dir / filename).read_text())
        for filename in sorted(examples_dir.glob("*.json"))
    )


S1_EXAMPLES: Final[tuple[dict[str, object], ...]] = _load_sentinel1_examples()
S2_EXAMPLES: Final[tuple[dict[str, object], ...]] = _load_sentinel2_examples()


@pytest.fixture
def projected_crs_json() -> dict[str, Any]:
    """Load projected CRS example."""
    return _load_projjson_file("projected_crs.json")


@pytest.fixture
def bound_crs_json() -> dict[str, Any]:
    """Load bound CRS example."""
    return _load_projjson_file("bound_crs.json")


@pytest.fixture
def compound_crs_json() -> dict[str, Any]:
    """Load compound CRS example."""
    return _load_projjson_file("compound_crs.json")


@pytest.fixture
def transformation_json() -> dict[str, Any]:
    """Load transformation example."""
    return _load_projjson_file("transformation.json")


@pytest.fixture
def datum_ensemble_json() -> dict[str, Any]:
    """Load datum ensemble example."""
    return _load_projjson_file("datum_ensemble.json")


@pytest.fixture
def explicit_prime_meridian_json() -> dict[str, Any]:
    """Load explicit prime meridian example."""
    return _load_projjson_file("explicit_prime_meridian.json")


@pytest.fixture
def implicit_prime_meridian_json() -> dict[str, Any]:
    """Load implicit prime meridian example."""
    return _load_projjson_file("implicit_prime_meridian.json")


@pytest.fixture
def all_projjson_examples() -> list[dict[str, Any]]:
    """Load all PROJ JSON examples."""
    examples_dir = Path(__file__).parent / "projjson_examples"
    examples = []
    for json_file in examples_dir.glob("*.json"):
        with open(json_file, "r") as f:
            examples.append(json.load(f))
    return examples


def extract_json_code_blocks(
    markdown_content: str,
) -> dict[tuple[int, int], dict[str, object]]:
    """
    Extract all JSON code blocks from a markdown document.

    Each extracted code block includes the JSON content and its position in the document.

    Args:
        markdown_content: The markdown document content as a string.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'json': The parsed JSON object (or None if parsing failed)
        - 'raw': The raw JSON string
        - 'start_line': The line number where the code block starts (1-indexed)
        - 'end_line': The line number where the code block ends (1-indexed)
        - 'error': Error message if JSON parsing failed (optional)
    """
    lines: list[str] = markdown_content.split("\n")
    code_blocks: dict[tuple[int, int], dict[str, object]] = {}
    i: int = 0

    while i < len(lines):
        line: str = lines[i]

        # Check for JSON code block start
        if re.match(r"^\s*```json\s*$", line):
            start_line: int = i + 1  # Line after ```json
            json_lines: list[str] = []
            i += 1

            # Collect lines until we find the closing ```
            while i < len(lines) and not re.match(r"^\s*```\s*$", lines[i]):
                json_lines.append(lines[i])
                i += 1

            end_line: int = i  # Line with closing ```
            raw_json: str = "\n".join(json_lines)

            parsed_json: Any = json.loads(raw_json)
            code_blocks[(start_line + 1, end_line)] = parsed_json

        i += 1

    return code_blocks


@pytest.fixture()
def proj_attrs_examples() -> dict[tuple[int, int], dict[str, object]]:
    """
    Extract JSON code blocks from the proj extension README.md.
    """
    spec_md = (
        Path(__file__).parent.parent.parent
        / "attributes"
        / "geo"
        / "proj"
        / "README.md"
    )
    content = spec_md.read_text(encoding="utf-8")
    return extract_json_code_blocks(content)


@pytest.fixture()
def json_code_blocks_from_readme(
    request: pytest.FixtureRequest,
) -> dict[tuple[int, int], dict[str, object]]:
    """Extract JSON code blocks from a markdown file."""
    readme_path = Path(request.param)
    content = readme_path.read_text(encoding="utf-8")
    return extract_json_code_blocks(content)


def _load_multiscales_examples() -> dict[str, dict[str, object]]:
    """Load a multiscale example."""
    examples_dir = Path(__file__).parent / "multiscales_examples"
    return {
        path.name: json.loads(path.read_text()) for path in examples_dir.glob("*.json")
    }


MULTISCALES_EXAMPLES = _load_multiscales_examples()


def _load_json_examples(
    *, prefix: Path, glob_str: str = "*.json"
) -> dict[str, dict[str, object]]:
    """
    Loads JSON examples from a prefix / directory. By default all files ending with .json are collected.
    """
    return {path.name: json.loads(path.read_text()) for path in prefix.glob(glob_str)}


GEOPROJ_EXAMPLES = _load_json_examples(
    prefix=Path(__file__).parent / "geoproj_examples"
)


def view_json_diff(
    a: dict[str, object],
    b: dict[str, object],
    *,
    sort_keys: bool = True,
    indent: int = 2,
) -> str:
    """
    Generate a human-readable diff between two JSON objects
    """
    a_str = json.dumps(a, indent=indent, sort_keys=sort_keys)
    b_str = json.dumps(b, indent=indent, sort_keys=sort_keys)

    # difflib.unified_diff returns an iterable of lines
    diff = difflib.unified_diff(
        a_str.splitlines(keepends=True),
        b_str.splitlines(keepends=True),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "".join(diff)
