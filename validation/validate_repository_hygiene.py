#!/usr/bin/env python3
"""Impide versionar salidas de compilación fuera de dependencias permitidas."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


GENERATED_DIRECTORIES = {"build", "CMakeFiles", "__pycache__"}
GENERATED_NAMES = {"CMakeCache.txt", "cmake_install.cmake", "compile_commands.json"}
GENERATED_SUFFIXES = {
    ".a",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".jar",
    ".lib",
    ".o",
    ".obj",
    ".pyc",
    ".pyo",
    ".so",
}
VENDORED_BINARY_PREFIXES = ("cuda/Common/FreeImage/lib/",)
NATIVE_MAGICS = {
    b"\x7fELF": "ejecutable u objeto ELF",
    b"\xfe\xed\xfa\xce": "binario Mach-O",
    b"\xce\xfa\xed\xfe": "binario Mach-O",
    b"\xfe\xed\xfa\xcf": "binario Mach-O",
    b"\xcf\xfa\xed\xfe": "binario Mach-O",
    b"\xca\xfe\xba\xbe": "binario universal Mach-O",
    b"\xbe\xba\xfe\xca": "binario universal Mach-O",
    b"MZ": "binario PE/Windows",
    b"!<arch>\n": "archivo de biblioteca ar",
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--report",
        type=Path,
        default=repository / "build" / "validation" / "repository-hygiene.json",
    )
    return parser.parse_args()


def tracked_paths(repository: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git ls-files falló: {message}")
    return [
        Path(os.fsdecode(raw_path))
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    ]


def binary_magic(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError as exc:
        return f"no se pudo inspeccionar: {exc}"
    for magic, description in NATIVE_MAGICS.items():
        if header.startswith(magic):
            return description
    return None


def reasons_for(relative: Path, repository: Path) -> list[str]:
    normalized = relative.as_posix()
    if normalized.startswith(VENDORED_BINARY_PREFIXES):
        return []

    reasons: list[str] = []
    if GENERATED_DIRECTORIES.intersection(relative.parts):
        reasons.append("directorio de construcción generado")
    if relative.name in GENERATED_NAMES:
        reasons.append("archivo generado por la herramienta de construcción")
    if relative.suffix.lower() in GENERATED_SUFFIXES:
        reasons.append(f"extensión compilada {relative.suffix.lower()}")

    absolute = repository / relative
    if absolute.is_file():
        magic = binary_magic(absolute)
        if magic:
            reasons.append(magic)
    return sorted(set(reasons))


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    errors: list[dict[str, object]] = []

    try:
        paths = tracked_paths(repository)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for relative in paths:
        if not (repository / relative).exists():
            continue
        reasons = reasons_for(relative, repository)
        if reasons:
            errors.append({"path": relative.as_posix(), "reasons": reasons})

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracked_files_checked": len(paths),
        "vendored_binary_prefixes": list(VENDORED_BINARY_PREFIXES),
        "errors": errors,
        "result": "failed" if errors else "passed",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Higiene del repositorio: {len(paths)} archivos rastreados, {len(errors)} errores")
    for error in errors:
        reasons = ", ".join(str(reason) for reason in error["reasons"])
        print(f"ERROR: {error['path']}: {reasons}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
