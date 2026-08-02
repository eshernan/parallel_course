#!/usr/bin/env python3
"""Comprueba inventario, metadatos y enlaces de los notebooks del curso."""

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


def local_targets(text: str) -> list[str]:
    targets: list[str] = []
    for raw_link in LOCAL_LINK.findall(text):
        if raw_link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = raw_link.strip("<>").split("#", 1)[0]
        if target:
            targets.append(target)
    return targets


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} no contiene un objeto JSON")
    return value


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    index = repository / "INDICE_CURSO.md"
    manifest_path = repository / "validation" / "notebooks-manifest.json"
    notebooks_root = repository / "curso" / "notebooks"
    errors: list[str] = []

    try:
        index_text = index.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"No se pudo leer INDICE_CURSO.md: {exc}")
        index_text = ""

    try:
        manifest = load_object(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"No se pudo leer el manifiesto de notebooks: {exc}")
        manifest = {"topics": [], "total_notebooks": 0}
    if manifest.get("schema_version") != "2.0":
        errors.append("El manifiesto de notebooks debe usar schema_version 2.0")

    for topic in REQUIRED_TOPICS:
        if not re.search(rf"^\| {topic}\.", index_text, flags=re.MULTILINE):
            errors.append(f"El índice no contiene el tema {topic}")

    checked_index_links = 0
    for target in local_targets(index_text):
        checked_index_links += 1
        resolved = (index.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"Enlace local inexistente en el índice: {target}")

    required: dict[Path, dict[str, Any]] = {}
    manifest_topics: list[str] = []
    covered_sessions: set[int] = set()
    checked_topic_links = 0
    topics = manifest.get("topics", [])
    if not isinstance(topics, list):
        errors.append("validation/notebooks-manifest.json: topics debe ser una lista")
        topics = []
    for topic_record in topics:
        if not isinstance(topic_record, dict):
            errors.append("El manifiesto contiene un tema que no es objeto")
            continue
        topic = topic_record.get("topic")
        directory = topic_record.get("directory")
        notebook_records = topic_record.get("notebooks", [])
        if not isinstance(topic, str) or not isinstance(directory, str) or not isinstance(notebook_records, list):
            errors.append(f"Registro de tema inválido: {topic_record}")
            continue
        manifest_topics.append(topic)
        topic_readme = notebooks_root / directory / "README.md"
        try:
            readme_text = topic_readme.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Falta o no se puede leer {topic_readme.relative_to(repository)}: {exc}")
            readme_text = ""
        for target in local_targets(readme_text):
            checked_topic_links += 1
            resolved = (topic_readme.parent / target).resolve()
            if not resolved.is_relative_to(repository):
                errors.append(f"{topic_readme.relative_to(repository)}: enlace sale del repositorio: {target}")
            elif not resolved.exists():
                errors.append(f"{topic_readme.relative_to(repository)}: enlace local inexistente: {target}")
        for notebook_record in notebook_records:
            if not isinstance(notebook_record, dict):
                errors.append(f"Tema {topic}: registro de notebook inválido")
                continue
            filename = notebook_record.get("filename")
            if not isinstance(filename, str):
                errors.append(f"Tema {topic}: notebook sin filename válido")
                continue
            relative = Path("curso") / "notebooks" / directory / filename
            if relative in required:
                errors.append(f"Notebook repetido en el manifiesto: {relative}")
            required[relative] = {**notebook_record, "topic": topic, "directory": directory}
            sessions = notebook_record.get("sessions", [])
            if not isinstance(sessions, list) or not sessions or not all(isinstance(value, int) for value in sessions):
                errors.append(f"{relative}: sessions debe ser una lista no vacía de enteros")
            else:
                covered_sessions.update(sessions)
            if f"]({filename})" not in readme_text:
                errors.append(f"{topic_readme.relative_to(repository)} no enlaza {filename}")
            if relative.as_posix() not in index_text:
                errors.append(f"INDICE_CURSO.md no enlaza directamente {relative.as_posix()}")

    if tuple(manifest_topics) != REQUIRED_TOPICS:
        errors.append(f"Temas del manifiesto: {manifest_topics}; esperados: {list(REQUIRED_TOPICS)}")
    declared_total = manifest.get("total_notebooks")
    if declared_total != len(required):
        errors.append(f"El manifiesto declara {declared_total} notebooks, pero enumera {len(required)}")
    if len(required) != 23:
        errors.append(f"El inventario obligatorio debe contener 23 notebooks, contiene {len(required)}")
    expected_sessions = set(range(1, 39))
    if covered_sessions != expected_sessions:
        missing_sessions = sorted(expected_sessions - covered_sessions)
        unexpected_sessions = sorted(covered_sessions - expected_sessions)
        errors.append(
            f"Cobertura de sesiones inválida; faltan {missing_sessions}, fuera de rango {unexpected_sessions}"
        )

    actual = {
        path.relative_to(repository)
        for path in notebooks_root.rglob("*.ipynb")
        if path.is_file()
    } if notebooks_root.is_dir() else set()
    missing = sorted(set(required) - actual)
    unexpected = sorted(actual - set(required))
    for relative in missing:
        errors.append(f"Falta notebook obligatorio: {relative.as_posix()}")
    for relative in unexpected:
        errors.append(f"Notebook fuera del manifiesto: {relative.as_posix()}")

    notebook_records: list[dict[str, Any]] = []
    checked_notebook_links = 0
    for relative in sorted(set(required) & actual):
        notebook_path = repository / relative
        expected = required[relative]
        try:
            notebook = load_object(notebook_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"No se pudo leer {relative}: {exc}")
            continue
        cells = notebook.get("cells", [])
        if not isinstance(cells, list):
            errors.append(f"{relative}: cells debe ser una lista")
            cells = []
        markdown = markdown_text(notebook)
        code_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"]
        opening_has_index = bool(markdown) and "INDICE_CURSO.md" in markdown[0]
        closing_has_index = len(markdown) >= 2 and "INDICE_CURSO.md" in markdown[-1]
        opening_has_topic = bool(markdown) and "README.md" in markdown[0]
        closing_has_topic = len(markdown) >= 2 and "README.md" in markdown[-1]
        if not opening_has_index or not closing_has_index:
            errors.append(f"{relative}: apertura y cierre deben enlazar INDICE_CURSO.md")
        if not opening_has_topic or not closing_has_topic:
            errors.append(f"{relative}: apertura y cierre deben enlazar README.md del tema")
        if len(markdown) < 17:
            errors.append(f"{relative}: contiene {len(markdown)} celdas Markdown; se esperaban al menos 17")
        if len(code_cells) < 3:
            errors.append(f"{relative}: contiene {len(code_cells)} celdas de código; se esperaban al menos 3")
        complete_markdown = "\n".join(markdown)
        required_sections = (
            "# " + str(expected.get("title", "")),
            "**Pregunta guía:**",
            "## Cómo usar este notebook",
            "## Antes de empezar",
            "## Resultados de aprendizaje",
            "## Explicación paso a paso",
            "## Mapa visual",
            "## Ejemplo resuelto:",
            "## Ejemplo guiado:",
            "## Comprueba tu comprensión",
            "## Ejercicios progresivos",
            "## Errores frecuentes",
            "## Criterios de aceptación",
            "## Síntesis",
            "## Referencias y material relacionado",
        )
        for section in required_sections:
            if section not in complete_markdown:
                errors.append(f"{relative}: falta la sección o identidad `{section}`")
        route_sections = required_sections[2:]
        positions = [complete_markdown.find(section) for section in route_sections]
        if positions != sorted(positions) or any(position < 0 for position in positions):
            errors.append(f"{relative}: las secciones no siguen la ruta pedagógica declarada")
        word_count = len(re.findall(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b", complete_markdown))
        if word_count < 500:
            errors.append(f"{relative}: explicación demasiado breve ({word_count} palabras; mínimo 500)")
        if notebook.get("nbformat") != 4:
            errors.append(f"{relative}: nbformat debe ser 4")
        course = notebook.get("metadata", {}).get("course", {})
        if course.get("topic") != expected["topic"]:
            errors.append(f"{relative}: metadata.course.topic no coincide con {expected['topic']}")
        if course.get("directory") != expected["directory"]:
            errors.append(f"{relative}: metadata.course.directory no coincide con {expected['directory']}")
        if course.get("title") != expected.get("title"):
            errors.append(f"{relative}: metadata.course.title no coincide con el manifiesto")
        if course.get("sessions") != expected.get("sessions"):
            errors.append(f"{relative}: metadata.course.sessions no coincide con el manifiesto")
        if course.get("pedagogy") != expected.get("pedagogy"):
            errors.append(f"{relative}: metadata.course.pedagogy no coincide con el manifiesto")
        expected_images = expected.get("images", [])
        if not isinstance(expected_images, list) or not expected_images:
            errors.append(f"{relative}: el manifiesto no declara imágenes pedagógicas")
            expected_images = []
        if course.get("images") != expected_images:
            errors.append(f"{relative}: metadata.course.images no coincide con el manifiesto")
        for image_name in expected_images:
            if not isinstance(image_name, str) or not image_name.endswith(".svg"):
                errors.append(f"{relative}: imagen inválida en manifiesto: {image_name!r}")
                continue
            image_reference = f"../../images/{image_name}"
            if image_reference not in complete_markdown:
                errors.append(f"{relative}: no incorpora la imagen declarada {image_name}")
            image_path = repository / "curso" / "images" / image_name
            if not image_path.is_file():
                errors.append(f"{relative}: falta la imagen compartida {image_name}")
        for cell_index, cell in enumerate(code_cells):
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(f"{relative}: celda de código {cell_index} conserva ejecución o salidas")
        for target in local_targets("\n".join(markdown)):
            checked_notebook_links += 1
            resolved = (notebook_path.parent / target).resolve()
            if not resolved.is_relative_to(repository):
                errors.append(f"{relative}: enlace sale del repositorio: {target}")
            elif not resolved.exists():
                errors.append(f"{relative}: enlace local inexistente: {target}")
        notebook_records.append(
            {
                "path": relative.as_posix(),
                "topic": expected["topic"],
                "sessions": expected.get("sessions"),
                "markdown_cells": len(markdown),
                "code_cells": len(code_cells),
                "word_count": word_count,
                "images": expected_images,
                "pedagogical_route": course.get("pedagogy"),
                "opening_index_link": opening_has_index,
                "closing_index_link": closing_has_index,
                "opening_topic_link": opening_has_topic,
                "closing_topic_link": closing_has_topic,
            }
        )

    report = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": str(index.relative_to(repository)),
        "manifest": str(manifest_path.relative_to(repository)),
        "topics_required": list(REQUIRED_TOPICS),
        "sessions_covered": sorted(covered_sessions),
        "required_notebooks": len(required),
        "notebooks_found": len(actual),
        "index_links_checked": checked_index_links,
        "topic_readme_links_checked": checked_topic_links,
        "notebook_links_checked": checked_notebook_links,
        "notebooks": notebook_records,
        "errors": errors,
        "result": "failed" if errors else "passed",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Navegación: {len(actual)}/{len(required)} notebooks, "
        f"{checked_index_links + checked_topic_links + checked_notebook_links} enlaces locales, {len(errors)} errores"
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
