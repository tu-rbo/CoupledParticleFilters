#!/usr/bin/env python3
"""Convert float64 precomputed artifacts to float32 without changing layout."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ConversionStats:
    scanned: int = 0
    converted: int = 0
    bytes_saved: int = 0
    failures: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert float64 arrays in configured precomputed artifacts to float32."
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yml"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_project_path(value: str | Path, working_directory: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (working_directory / path).resolve()


def artifact_roots(project_config: Path, working_directory: Path | None = None) -> list[Path]:
    """Return configured artifact-output roots while excluding source dataset data."""

    with project_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"project config must contain a mapping: {project_config}")

    working_directory = working_directory or Path.cwd()
    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("project config outputs must be a mapping")
    paths = config.get("paths", {})
    dataset_values = [Path("data/rbo_dataset")]
    if isinstance(paths, dict) and "rbo_dataset" in paths:
        dataset_values.append(Path(paths["rbo_dataset"]))
    excluded = {
        _resolve_project_path(value, working_directory) for value in dataset_values
    }

    roots: list[Path] = []
    seen: set[Path] = set()
    for value in outputs.values():
        root = _resolve_project_path(value, working_directory)
        if any(root == dataset_root or dataset_root in root.parents for dataset_root in excluded):
            continue
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def convert_float64(value: Any) -> tuple[Any, bool]:
    """Recursively replace float64 arrays/scalars while preserving object arrays."""

    if isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            return value.astype(np.float32), True
        if value.dtype == object:
            converted = np.empty_like(value)
            changed = False
            for index in np.ndindex(value.shape):
                item, item_changed = convert_float64(value[index])
                converted[index] = item
                changed |= item_changed
            return (converted if changed else value), changed
        return value, False
    if isinstance(value, np.float64):
        return np.float32(value), True
    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        changed = False
        for key, item in value.items():
            converted_item, item_changed = convert_float64(item)
            converted[key] = converted_item
            changed |= item_changed
        return (converted if changed else value), changed
    if isinstance(value, list):
        converted_items = []
        changed = False
        for item in value:
            converted_item, item_changed = convert_float64(item)
            converted_items.append(converted_item)
            changed |= item_changed
        return (converted_items if changed else value), changed
    if isinstance(value, tuple):
        converted_items = []
        changed = False
        for item in value:
            converted_item, item_changed = convert_float64(item)
            converted_items.append(converted_item)
            changed |= item_changed
        return (tuple(converted_items) if changed else value), changed
    return value, False


def _atomic_save(path: Path, value: Any, mode: int) -> int:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        return path.stat().st_size
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_file(path: Path) -> tuple[bool, int]:
    """Convert one artifact and return whether it changed and net bytes saved."""

    before = path.stat()
    try:
        # Numeric arrays can be inspected from their header without reading a
        # multi-megabyte float32 artifact into memory.  Object arrays need a
        # normal load so their nested dict values can be converted recursively.
        loaded = np.load(path, allow_pickle=True, mmap_mode="r")
    except ValueError:
        loaded = np.load(path, allow_pickle=True)
    converted, changed = convert_float64(loaded)
    if not changed:
        return False, 0
    after_size = _atomic_save(path, converted, stat.S_IMODE(before.st_mode))
    return True, before.st_size - after_size


def convert_roots(roots: list[Path], *, verbose: bool = False) -> ConversionStats:
    stats = ConversionStats()
    for root in roots:
        if not root.is_dir():
            if verbose:
                print(f"Skipping missing artifact root: {root}")
            continue
        for path in root.rglob("*.npy"):
            if not path.is_file():
                continue
            stats.scanned += 1
            try:
                changed, bytes_saved = convert_file(path)
            except Exception as exc:
                stats.failures += 1
                print(f"Failed to convert {path}: {exc}", file=sys.stderr)
                continue
            if changed:
                stats.converted += 1
                stats.bytes_saved += bytes_saved
                if verbose:
                    print(f"Converted {path} ({bytes_saved:+d} bytes)")
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_config = args.project_config.expanduser().resolve()
    roots = artifact_roots(project_config)
    stats = convert_roots(roots, verbose=args.verbose)
    print(
        "Artifact float32 conversion: "
        f"scanned={stats.scanned} converted={stats.converted} "
        f"bytes_saved={stats.bytes_saved} failures={stats.failures}"
    )
    return int(stats.failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
