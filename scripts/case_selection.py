"""Shared case-directory selection helpers for comparison plots."""

from __future__ import annotations

import re
from pathlib import Path


def natural_case_key(path: Path) -> tuple[str, float, str]:
    match = re.search(r"_nu(?P<nu>[0-9.]+)$", path.name)
    if match is None:
        return path.name, float("inf"), path.name
    return path.name[: match.start()], float(match.group("nu")), path.name


def resolve_case_dirs(root: Path, case_regex: str) -> list[Path]:
    try:
        pattern = re.compile(case_regex)
    except re.error as exc:
        raise ValueError(f"Invalid --case-regex {case_regex!r}: {exc}") from exc

    if not root.is_dir():
        raise FileNotFoundError(f"Case root directory does not exist: {root}")

    case_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and pattern.fullmatch(path.name)
        ),
        key=natural_case_key,
    )
    if not case_dirs:
        raise FileNotFoundError(
            f"No case directories under {root} matched --case-regex {case_regex!r}"
        )
    return case_dirs
