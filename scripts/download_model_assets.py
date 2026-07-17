#!/usr/bin/env python3
"""Download and verify research-model assets declared in YAML."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from urllib.request import urlopen

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "model_assets.yml"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assets = document.get("assets") if isinstance(document, dict) else None
    if not isinstance(assets, dict) or not assets:
        raise ValueError(f"No assets are declared in {path}")
    return assets


def resolve_destination(repository_root: Path, relative_path: str) -> Path:
    destination = (repository_root / relative_path).resolve()
    repository_root = repository_root.resolve()
    if not destination.is_relative_to(repository_root):
        raise ValueError(f"Asset destination escapes the repository: {relative_path}")
    return destination


def verify_asset(path: Path, asset: dict[str, object]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    expected_size = int(asset["size_bytes"])
    if path.stat().st_size != expected_size:
        return False, f"size mismatch ({path.stat().st_size} != {expected_size})"
    actual_digest = sha256_file(path)
    expected_digest = str(asset["sha256"])
    if actual_digest != expected_digest:
        return False, f"SHA-256 mismatch ({actual_digest} != {expected_digest})"
    return True, "verified"


def download_asset(
    name: str,
    asset: dict[str, object],
    repository_root: Path,
    *,
    force: bool,
) -> Path:
    destination = resolve_destination(repository_root, str(asset["destination"]))
    valid, detail = verify_asset(destination, asset)
    if valid and not force:
        print(f"{name}: already verified at {destination}")
        return destination
    if destination.exists() and not force:
        raise RuntimeError(
            f"{name}: existing file is invalid ({detail}); rerun with --force to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with urlopen(str(asset["url"])) as response, partial.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        if byte_count != int(asset["size_bytes"]):
            raise RuntimeError(
                f"{name}: downloaded {byte_count} bytes; expected {asset['size_bytes']}"
            )
        if digest.hexdigest() != str(asset["sha256"]):
            raise RuntimeError(
                f"{name}: downloaded SHA-256 {digest.hexdigest()}; expected {asset['sha256']}"
            )
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print(f"{name}: downloaded and verified at {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        help="Asset name to process; repeat to select multiple assets (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an invalid/existing asset")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files without downloading",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    assets = load_manifest(manifest_path)
    selected = args.assets or list(assets)
    unknown = sorted(set(selected) - set(assets))
    if unknown:
        raise ValueError(f"Unknown assets: {', '.join(unknown)}")

    failed = False
    for name in selected:
        asset = assets[name]
        destination = resolve_destination(REPOSITORY_ROOT, str(asset["destination"]))
        if args.verify_only:
            valid, detail = verify_asset(destination, asset)
            print(f"{name}: {detail} ({destination})")
            failed |= not valid
        else:
            download_asset(name, asset, REPOSITORY_ROOT, force=args.force)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
