#!/usr/bin/env python3
"""Comprueba el índice del curso y los enlaces de retorno de los notebooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCAL_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
REQUIRED_TOPICS = tuple(f"{number:02d}" for number in range(9))


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--report",
        type=Path,
        default=repository / "build" / "validation" / "navigation.json",
    )
    return parser.parse_args()


def markdown_text(notebook: dict[str, Any]) -> list[str]:
    cells = notebook.get("cells", [])
    return [
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    ]


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    index = repository / "INDICE_CURSO.md"
    errors: list[str] = []

    if not index.is_file():
        errors.append("No existe INDICE_CURSO.md en la raíz del repositorio")
        index_text = ""
    else:
        index_text = index.read_text(encoding="utf-8")

    for topic in REQUIRED_TOPICS:
        if not re.search(rf"^\| {topic}\.", index_text, flags=re.MULTILINE):
            errors.append(f"El índice no contiene el tema {topic}")

    checked_links = 0
    for raw_link in LOCAL_LINK.findall(index_text):
        if raw_link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = raw_link.strip("<>").split("#", 1)[0]
        if not target:
            continue
        checked_links += 1
        resolved = (index.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"Enlace local inexistente en el índice: {raw_link}")

    notebooks_root = repository / "curso" / "notebooks"
    notebooks = sorted(notebooks_root.rglob("*.ipynb")) if notebooks_root.is_dir() else []
    notebook_records: list[dict[str, Any]] = []
    for notebook_path in notebooks:
        relative = notebook_path.relative_to(repository)
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"No se pudo leer {relative}: {exc}")
            continue
        markdown = markdown_text(notebook)
        opening_has_index = bool(markdown) and "INDICE_CURSO.md" in markdown[0]
        closing_has_index = len(markdown) >= 2 and "INDICE_CURSO.md" in markdown[-1]
        if not opening_has_index:
            errors.append(f"{relative}: la primera celda Markdown no enlaza INDICE_CURSO.md")
        if not closing_has_index:
            errors.append(f"{relative}: la última celda Markdown no enlaza INDICE_CURSO.md")
        notebook_records.append(
            {
                "path": str(relative),
                "opening_index_link": opening_has_index,
                "closing_index_link": closing_has_index,
            }
        )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": str(index.relative_to(repository)),
        "topics_required": list(REQUIRED_TOPICS),
        "local_links_checked": checked_links,
        "notebooks_found": len(notebooks),
        "notebooks": notebook_records,
        "errors": errors,
        "result": "failed" if errors else "passed",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Índice: {checked_links} enlaces locales, {len(notebooks)} notebooks, "
        f"{len(errors)} errores"
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
