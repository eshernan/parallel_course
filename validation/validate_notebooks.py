#!/usr/bin/env python3
"""Ejecuta secuencialmente las celdas Python de todos los notebooks requeridos."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--report",
        type=Path,
        default=repository / "build" / "validation" / "notebooks.json",
    )
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} no contiene un objeto JSON")
    return value


def required_paths(repository: Path) -> list[Path]:
    manifest = load_object(repository / "validation" / "notebooks-manifest.json")
    paths: list[Path] = []
    for topic in manifest.get("topics", []):
        directory = topic["directory"]
        for notebook in topic["notebooks"]:
            paths.append(repository / "curso" / "notebooks" / directory / notebook["filename"])
    if len(paths) != manifest.get("total_notebooks"):
        raise ValueError("El total del manifiesto no coincide con sus notebooks")
    return paths


def execute_notebook(path: Path, repository: Path) -> dict[str, Any]:
    notebook = load_object(path)
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    stdout = io.StringIO()
    stderr = io.StringIO()
    executed = 0
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for index, cell in enumerate(notebook.get("cells", [])):
                if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                compiled = compile(source, f"{path}#cell-{index}", "exec")
                exec(compiled, namespace)
                executed += 1
    except Exception:
        return {
            "path": str(path.relative_to(repository)),
            "result": "failed",
            "code_cells_executed": executed,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "stdout": stdout.getvalue()[-8000:],
            "stderr": stderr.getvalue()[-8000:],
            "traceback": traceback.format_exc()[-12000:],
        }
    return {
        "path": str(path.relative_to(repository)),
        "result": "passed",
        "code_cells_executed": executed,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "stdout": stdout.getvalue()[-8000:],
        "stderr": stderr.getvalue()[-8000:],
    }


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    errors: list[str] = []
    try:
        paths = required_paths(repository)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: no se pudo cargar el inventario: {exc}", file=sys.stderr)
        return 1

    previous_cwd = Path.cwd()
    results: list[dict[str, Any]] = []
    try:
        os.chdir(repository)
        for path in paths:
            try:
                result = execute_notebook(path, repository)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                result = {"path": str(path.relative_to(repository)), "result": "failed", "error": str(exc)}
            results.append(result)
            if result["result"] != "passed":
                errors.append(f"{result['path']}: ejecución fallida")
    finally:
        os.chdir(previous_cwd)

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "required_notebooks": len(paths),
        "executed_notebooks": sum(result["result"] == "passed" for result in results),
        "results": results,
        "errors": errors,
        "result": "failed" if errors else "passed",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Notebooks ejecutados: {report['executed_notebooks']}/{len(paths)}, {len(errors)} errores")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        for result in results:
            if result["result"] != "passed":
                print(result.get("traceback", result.get("error", "")), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
