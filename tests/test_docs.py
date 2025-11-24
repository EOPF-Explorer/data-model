"""
Tests for Python code examples embedded in documentation.

This module uses pytest-examples to extract all Python code blocks
from the documentation markdown files and executes them to ensure they
remain correct and up-to-date.

Output assertions can be specified using #> style comments:
    print("hello")
    #> hello
"""

from io import StringIO
from pathlib import Path

import pytest
from pytest_examples import CodeExample, find_examples


def _should_skip_example(example: CodeExample) -> bool:
    """Determine if an example should be skipped."""
    # Check for explicit test: skip marker
    if "# test: skip" in example.source or "test: skip" in example.source:
        return True
    skip_patterns: list[str] = []
    return any(pattern in example.source for pattern in skip_patterns)


def _make_group_id(group_examples: list[CodeExample]) -> str:
    """Generate a descriptive ID for a group of code examples.

    Uses the filename and line range from the first and last examples
    in the group to create an informative test ID.

    Example: "sentinel2.md[12-26]"
    """
    if not group_examples:
        return "empty-group"
    first = group_examples[0]
    last = group_examples[-1]
    # Extract filename from path
    filename = Path(first.path).name
    return f"{filename}[{first.start_line}-{last.end_line}]"


def _parse_output_assertions(source: str) -> list[str]:
    """Parse expected output assertions from code using #> style comments.

    Examples:
        print("hello")
        #> hello

    Returns a list of expected output lines.
    """
    expected_lines = []
    for line in source.split("\n"):
        if line.strip().startswith("#>"):
            # Extract the expected output after '#>'
            expected = line.split("#>", 1)[1].strip()
            expected_lines.append(expected)
    return expected_lines


def _validate_output(example: CodeExample, captured_output: str) -> None:
    """Validate that captured output matches expected output assertions.

    Raises AssertionError if output doesn't match expected assertions.
    """
    expected_lines = _parse_output_assertions(example.source)
    if not expected_lines:
        # No output assertions to validate
        return

    actual_lines = [line.rstrip() for line in captured_output.rstrip().split("\n")]

    if len(actual_lines) != len(expected_lines):
        raise AssertionError(
            f"Output line count mismatch in {example.path} (lines {example.start_line}-{example.end_line}):\n"
            f"Expected {len(expected_lines)} lines but got {len(actual_lines)} lines\n"
            f"Expected:\n{chr(10).join(expected_lines)}\n"
            f"Actual:\n{chr(10).join(actual_lines)}"
        )

    for i, (expected, actual) in enumerate(zip(expected_lines, actual_lines)):
        if expected != actual:
            raise AssertionError(
                f"Output mismatch on line {i + 1} in {example.path} (lines {example.start_line}-{example.end_line}):\n"
                f"Expected: {expected!r}\n"
                f"Actual: {actual!r}"
            )


# Get all examples and group them by group ID
_all_examples = list(find_examples("docs"))
_examples_by_group = {}
for ex in _all_examples:
    if ex.prefix == "python":
        if ex.group not in _examples_by_group:
            _examples_by_group[ex.group] = []
        _examples_by_group[ex.group].append(ex)


@pytest.mark.parametrize(
    "group_examples",
    list(_examples_by_group.values()),
    ids=_make_group_id,
)
def test_doc_example_group(group_examples: list[CodeExample]) -> None:
    """
    Test a group of related Python code examples from the documentation.

    Examples are grouped by their pytest-examples group ID, allowing dependent
    examples to share state (e.g., a variable defined in one example used in the next).

    If any example would be skipped but later examples depend on it, the entire
    group is skipped.

    Output can be validated by adding #> style assertions:
        print("hello")
        #> hello
    """
    # Execute all examples in the group sequentially to share state
    namespace = {}
    any_skipped = False

    for example in group_examples:
        if _should_skip_example(example):
            any_skipped = True
            continue

        # If we're about to execute an example but we've skipped some,
        # we might have missing dependencies
        if any_skipped:
            pytest.skip(
                "Earlier examples in group were skipped; dependencies may be missing"
            )

        # Execute the example code with shared namespace and capture output
        try:
            # Capture stdout to check output assertions
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                exec(example.source, namespace)
                captured_output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            # Validate output if there are expected output assertions
            _validate_output(example, captured_output)

        except NameError as e:
            # If a NameError occurs, it might be due to skipped dependencies
            if any_skipped:
                pytest.skip(f"Skipped dependencies caused NameError: {e}")
            raise AssertionError(
                f"Code example in {example.path} (lines {example.start_line}-{example.end_line}) failed:\n"
                f"{example.source}\n\n{e}"
            ) from e
        except Exception as e:
            raise AssertionError(
                f"Code example in {example.path} (lines {example.start_line}-{example.end_line}) failed:\n"
                f"{example.source}\n\n{e}"
            ) from e
