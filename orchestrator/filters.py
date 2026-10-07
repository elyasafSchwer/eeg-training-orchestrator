"""Metadata filtering: turns `age>=20` style expressions into a selection of visits."""

import json
import operator
import sys
from dataclasses import dataclass
from pathlib import Path

# Two-character operators first, so ">=" is not parsed as ">".
OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
    "=": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}


@dataclass(frozen=True)
class Filter:
    key: str
    op: str
    value: object

    def __str__(self):
        return f"{self.key}{self.op}{self.value}"


@dataclass(frozen=True)
class Visit:
    visit_id: str
    eeg_path: Path
    metadata: dict


def parse_filter(expr):
    """'age>=20' -> Filter('age', '>=', 20)"""
    for op in OPERATORS:
        key, found, raw_value = expr.partition(op)
        if found and key and raw_value:
            return Filter(key.strip(), op, _coerce(raw_value.strip()))
    raise ValueError(f"cannot parse filter {expr!r}, expected something like 'age>=20' or 'gender=female'")


def _coerce(raw):
    """Give the filter value the same type the metadata uses: bool, number or string."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def matches(metadata, visit_filters):
    return all(_matches_one(metadata, f) for f in visit_filters)


def _matches_one(metadata, f):
    if f.key not in metadata:
        return False  # a visit missing the field cannot satisfy a filter on it
    actual, expected = metadata[f.key], f.value
    both_numeric = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    if not both_numeric and type(actual) is not type(expected):
        actual, expected = str(actual), str(expected)
    return OPERATORS[f.op](actual, expected)


def select_visits(data_dir, visit_filters):
    """Every <visit_id>.json in data_dir that matches the filters and has EEG data next to it."""
    selected = []
    for metadata_path in sorted(Path(data_dir).glob("*.json")):
        eeg_path = metadata_path.with_suffix(".npy")
        if not eeg_path.exists():
            print(f"skipping {metadata_path.stem}: metadata without EEG data", file=sys.stderr)
            continue
        metadata = json.loads(metadata_path.read_text())
        if matches(metadata, visit_filters):
            selected.append(Visit(metadata_path.stem, eeg_path, metadata))
    return selected
