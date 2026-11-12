"""Copy the handoff source set into transfer-package/source-snapshot.

Run from the repository root:
    python transfer-package/snapshot_source.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


def read_manifest(path: Path) -> tuple[list[str], list[str]]:
    required: list[str] = []
    optional: list[str] = []
    section = "required"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = "optional" if "optional" in line else "required"
            continue
        (optional if section == "optional" else required).append(line)
    return required, optional


def copy_entry(repo: Path, destination: Path, relative: str) -> int:
    source = (repo / relative).resolve()
    if repo not in source.parents and source != repo:
        raise RuntimeError(f"refusing path outside repository: {relative}")
    if not source.exists():
        raise FileNotFoundError(f"manifest path is missing: {relative}")
    if source.is_dir():
        count = 0
        for child in source.rglob("*"):
            if (
                child.is_file()
                and "__pycache__" not in child.relative_to(repo).parts
                and child.suffix != ".pyc"
                and "transfer-package" not in child.relative_to(repo).parts
            ):
                rel = child.relative_to(repo)
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)
                count += 1
        return count
    if "transfer-package" in source.relative_to(repo).parts:
        return 0
    target = destination / source.relative_to(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return 1


def main() -> int:
    repo = Path.cwd().resolve()
    package = repo / "transfer-package"
    manifest = package / "source-manifest.txt"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    required, optional = read_manifest(manifest)
    destination = package / "source-snapshot"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    copied = 0
    missing_required: list[str] = []
    for relative in required:
        try:
            copied += copy_entry(repo, destination, relative)
        except FileNotFoundError:
            missing_required.append(relative)
    if missing_required:
        raise SystemExit(
            "Required manifest paths are missing:\n" + "\n".join(f"  - {p}" for p in missing_required)
        )

    optional_copied = 0
    for relative in optional:
        # Keep binary PDFs/checkpoints out of the handoff even when present.
        if Path(relative).suffix.lower() in {".pdf", ".pt", ".pth", ".ckpt"}:
            continue
        if (repo / relative).exists():
            optional_copied += copy_entry(repo, destination, relative)
    print(f"Copied {copied} required files and {optional_copied} optional files to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
