"""
Build script for ChurnGuard AI Vercel deployment.
Packages all Python source files and artifacts into the stlite_build/ directory
as base64-encoded JSON manifests for stlite to load.
"""

import base64
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "stlite_build"


def collect_python_files():
    """Collect all Python source files with their module paths."""
    files = {}

    # Directories to scan for .py files
    scan_dirs = [
        ("app", PROJECT_ROOT / "app"),
        ("src", PROJECT_ROOT / "src"),
        ("config", PROJECT_ROOT / "config"),
    ]

    for prefix, directory in scan_dirs:
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            rel_path = py_file.relative_to(PROJECT_ROOT)
            module_path = str(rel_path).replace("\\", "/")
            with open(py_file, "r", encoding="utf-8") as f:
                files[module_path] = f.read()

    return files


def collect_binary_files():
    """Collect binary artifacts as base64."""
    binaries = {}
    
    # Artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    for artifact in artifacts_dir.iterdir():
        if artifact.is_file():
            rel_path = f"artifacts/{artifact.name}"
            with open(artifact, "rb") as f:
                binaries[rel_path] = base64.b64encode(f.read()).decode("ascii")

    return binaries


def collect_data_files():
    """Collect CSV data files."""
    data_files = {}
    data_dir = PROJECT_ROOT / "data"
    
    for csv_file in data_dir.rglob("*.csv"):
        if "__pycache__" in str(csv_file):
            continue
        rel_path = str(csv_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
        with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
            data_files[rel_path] = f.read()

    return data_files


def collect_asset_files():
    """Collect image assets as base64."""
    assets = {}
    assets_dir = PROJECT_ROOT / "app" / "assets"

    for asset in assets_dir.iterdir():
        if asset.is_file() and asset.suffix in (".png", ".ico", ".jpg", ".svg"):
            rel_path = f"app/assets/{asset.name}"
            with open(asset, "rb") as f:
                assets[rel_path] = base64.b64encode(f.read()).decode("ascii")

    return assets


def collect_config_files():
    """Collect .streamlit config."""
    configs = {}
    toml_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    if toml_path.exists():
        with open(toml_path, "r", encoding="utf-8") as f:
            configs[".streamlit/config.toml"] = f.read()
    return configs


def build_manifest():
    """Build the complete file manifest for stlite."""
    print("Collecting Python source files...")
    py_files = collect_python_files()
    print(f"  Found {len(py_files)} Python files")

    print("Collecting binary artifacts...")
    binaries = collect_binary_files()
    print(f"  Found {len(binaries)} binary artifacts")

    print("Collecting data files...")
    data_files = collect_data_files()
    print(f"  Found {len(data_files)} data files")

    print("Collecting image assets...")
    assets = collect_asset_files()
    print(f"  Found {len(assets)} asset files")

    print("Collecting config files...")
    configs = collect_config_files()
    print(f"  Found {len(configs)} config files")

    manifest = {
        "text_files": {**py_files, **data_files, **configs},
        "binary_files": {**binaries, **assets},
    }

    os.makedirs(BUILD_DIR, exist_ok=True)
    manifest_path = BUILD_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    size_mb = manifest_path.stat().st_size / (1024 * 1024)
    print(f"\nManifest written to {manifest_path} ({size_mb:.2f} MB)")
    print(f"Total files: {len(manifest['text_files']) + len(manifest['binary_files'])}")


if __name__ == "__main__":
    build_manifest()
