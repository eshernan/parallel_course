#!/usr/bin/env python3
"""Valida el inventario y construye los ejercicios registrados del curso."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUS = {"active", "planned", "retired"}
VALID_LANGUAGES = {"c", "cxx", "cuda", "hip", "sycl"}
VALID_OS = {"linux", "macos", "windows"}
VALID_ARCHITECTURES = {"x86_64", "arm64", "arm32"}
VALID_CPU_VENDORS = {"any", "intel", "amd", "apple", "arm"}
VALID_ACCELERATORS = {"none", "nvidia", "amd", "any"}
VALID_CAPABILITIES = {
    "cpu",
    "pthreads",
    "openmp",
    "mpi",
    "openmp-target",
    "cuda",
    "hip",
    "sycl",
    "kokkos",
    "raja",
}
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "id",
    "topic",
    "title",
    "status",
    "language",
    "standard",
    "build",
    "requirements",
    "platforms",
}
ALLOWED_TOP_LEVEL = REQUIRED_TOP_LEVEL | {"notes"}


def normalize_os(value: str | None = None) -> str:
    raw = (value or platform.system()).lower()
    if raw.startswith("darwin") or raw.startswith("mac"):
        return "macos"
    if raw.startswith("win"):
        return "windows"
    if raw.startswith("linux"):
        return "linux"
    return raw


def normalize_architecture(value: str | None = None) -> str:
    raw = (value or platform.machine()).lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "armv7l": "arm32",
        "armv8l": "arm32",
    }
    return aliases.get(raw, raw)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} debe contener un objeto JSON")
    return value


def string_list(value: Any, field: str, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} debe ser una lista no vacía de cadenas")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} no admite valores repetidos")
    if allowed is not None:
        invalid = sorted(set(value) - allowed)
        if invalid:
            raise ValueError(f"{field} contiene valores no admitidos: {', '.join(invalid)}")
    return value


def validate_manifest(data: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_TOP_LEVEL - data.keys())
    extra = sorted(data.keys() - ALLOWED_TOP_LEVEL)
    if missing:
        errors.append(f"faltan campos: {', '.join(missing)}")
    if extra:
        errors.append(f"campos no reconocidos: {', '.join(extra)}")
    if missing:
        return [f"{path}: {error}" for error in errors]

    scalar_checks = (
        (data.get("schema_version") == "1.0", "schema_version debe ser 1.0"),
        (isinstance(data.get("id"), str) and bool(data["id"]), "id debe ser una cadena no vacía"),
        (isinstance(data.get("topic"), str) and bool(data["topic"]), "topic debe ser una cadena no vacía"),
        (isinstance(data.get("title"), str) and len(data["title"]) >= 5, "title debe tener al menos cinco caracteres"),
        (data.get("status") in VALID_STATUS, "status no es válido"),
        (data.get("language") in VALID_LANGUAGES, "language no es válido"),
        (isinstance(data.get("standard"), str) and bool(data["standard"]), "standard debe ser una cadena no vacía"),
    )
    errors.extend(message for condition, message in scalar_checks if not condition)

    build = data.get("build")
    if not isinstance(build, dict):
        errors.append("build debe ser un objeto")
    else:
        allowed_build = {"system", "target", "tests", "configure_args"}
        missing_build = {"system", "target", "tests"} - build.keys()
        extra_build = build.keys() - allowed_build
        if missing_build:
            errors.append(f"build: faltan campos: {', '.join(sorted(missing_build))}")
        if extra_build:
            errors.append(f"build: campos no reconocidos: {', '.join(sorted(extra_build))}")
        if build.get("system") != "cmake":
            errors.append("build.system debe ser cmake")
        if not isinstance(build.get("target"), str) or not build.get("target"):
            errors.append("build.target debe ser una cadena no vacía")
        if not isinstance(build.get("tests"), bool):
            errors.append("build.tests debe ser booleano")
        configure_args = build.get("configure_args", [])
        if not isinstance(configure_args, list) or not all(isinstance(arg, str) for arg in configure_args):
            errors.append("build.configure_args debe ser una lista de cadenas")

    requirements = data.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("requirements debe ser un objeto")
    else:
        if set(requirements) != {"tools", "capabilities"}:
            errors.append("requirements solo admite tools y capabilities, ambos obligatorios")
        try:
            string_list(requirements.get("tools"), "requirements.tools")
            string_list(requirements.get("capabilities"), "requirements.capabilities", VALID_CAPABILITIES)
        except ValueError as exc:
            errors.append(str(exc))

    platforms = data.get("platforms")
    if not isinstance(platforms, dict):
        errors.append("platforms debe ser un objeto")
    else:
        expected = {"operating_systems", "architectures", "cpu_vendors", "accelerator"}
        if set(platforms) != expected:
            errors.append("platforms debe definir operating_systems, architectures, cpu_vendors y accelerator")
        try:
            string_list(platforms.get("operating_systems"), "platforms.operating_systems", VALID_OS)
            string_list(platforms.get("architectures"), "platforms.architectures", VALID_ARCHITECTURES)
            string_list(platforms.get("cpu_vendors"), "platforms.cpu_vendors", VALID_CPU_VENDORS)
        except ValueError as exc:
            errors.append(str(exc))
        if platforms.get("accelerator") not in VALID_ACCELERATORS:
            errors.append("platforms.accelerator no es válido")

    return [f"{path}: {error}" for error in errors]


def excluded(path: Path, root: Path, excluded_directories: set[str]) -> bool:
    relative = path.relative_to(root)
    return any(part in excluded_directories for part in relative.parts)


def find_unregistered_sources(
    root: Path,
    manifests: list[Path],
    source_extensions: set[str],
    excluded_directories: set[str],
) -> list[str]:
    manifest_directories = {path.parent.resolve() for path in manifests}
    unregistered: list[str] = []
    for source in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in source_extensions):
        if excluded(source, root, excluded_directories):
            continue
        current = source.parent.resolve()
        root_resolved = root.resolve()
        registered = False
        while current == root_resolved or root_resolved in current.parents:
            if current in manifest_directories:
                registered = True
                break
            if current == root_resolved:
                break
            current = current.parent
        if not registered:
            unregistered.append(str(source.relative_to(root)))
    return unregistered


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {
            "command": command,
            "return_code": completed.returncode,
            "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
            "output": completed.stdout[-12000:],
        }
    except OSError as exc:
        return {
            "command": command,
            "return_code": 127,
            "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
            "output": str(exc),
        }


def build_exercise(manifest: dict[str, Any], source_dir: Path, build_dir: Path, run_tests: bool) -> tuple[str, list[dict[str, Any]]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        *manifest["build"].get("configure_args", []),
    ]
    commands = [run_command(configure, source_dir)]
    if commands[-1]["return_code"] != 0:
        return "configure_failed", commands

    build = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--parallel",
        "2",
        "--target",
        manifest["build"]["target"],
    ]
    commands.append(run_command(build, source_dir))
    if commands[-1]["return_code"] != 0:
        return "build_failed", commands

    if run_tests and manifest["build"]["tests"]:
        list_tests = [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--show-only=json-v1",
        ]
        commands.append(run_command(list_tests, source_dir))
        if commands[-1]["return_code"] != 0:
            return "test_discovery_failed", commands
        try:
            discovered = json.loads(commands[-1]["output"])
        except json.JSONDecodeError:
            return "test_discovery_failed", commands
        if not discovered.get("tests"):
            return "no_tests_registered", commands
        test = ["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"]
        commands.append(run_command(test, source_dir))
        if commands[-1]["return_code"] != 0:
            return "test_failed", commands
        return "tested", commands
    return "compiled", commands


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def append_github_summary(report: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    summary = report["summary"]
    lines = [
        "### Validación de ejercicios",
        "",
        f"- Resultado: `{report['result']}`",
        f"- Manifiestos: {summary['manifests']}",
        f"- Ejercicios activos: {summary['active']}",
        f"- Compilados: {summary['compiled']}",
        f"- Pruebas ejecutadas: {summary['tested']}",
        f"- Planeados: {summary['planned']}",
        f"- Omitidos por plataforma/capacidad: {summary['skipped']}",
        f"- Errores: {len(report['errors'])}",
        "",
    ]
    with open(summary_path, "a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercises-root", type=Path, default=repository / "curso" / "ejercicios")
    parser.add_argument("--policy", type=Path, default=repository / "validation" / "policy.json")
    parser.add_argument("--build-root", type=Path, default=repository / "build" / "exercise-validation")
    parser.add_argument("--report", type=Path, default=repository / "build" / "exercise-validation" / "report.json")
    parser.add_argument("--compile", action="store_true", dest="compile_exercises")
    parser.add_argument("--test", action="store_true", dest="run_tests")
    parser.add_argument("--capability", choices=sorted(VALID_CAPABILITIES))
    parser.add_argument("--accelerator", choices=["none", "nvidia", "amd"], default="none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.exercises_root.resolve()
    errors: list[str] = []

    try:
        policy = load_json(args.policy.resolve())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    minimum_active = policy.get("minimum_active_exercises")
    if not isinstance(minimum_active, int) or minimum_active < 0:
        print("minimum_active_exercises debe ser un entero no negativo", file=sys.stderr)
        return 2
    minimum_by_capability = policy.get("minimum_by_capability", {})
    if not isinstance(minimum_by_capability, dict) or any(
        capability not in VALID_CAPABILITIES or not isinstance(value, int) or value < 0
        for capability, value in minimum_by_capability.items()
    ):
        print("minimum_by_capability debe asociar capacidades válidas con enteros no negativos", file=sys.stderr)
        return 2
    try:
        source_extensions = set(string_list(policy.get("source_extensions"), "source_extensions"))
        excluded_value = policy.get("excluded_directories", [])
        if not isinstance(excluded_value, list) or not all(isinstance(item, str) for item in excluded_value):
            raise ValueError("excluded_directories debe ser una lista de cadenas")
        excluded_directories = set(excluded_value)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not root.is_dir():
        errors.append(f"No existe el directorio administrado: {root}")
        manifests: list[Path] = []
    else:
        manifests = sorted(
            path for path in root.rglob("exercise.json") if not excluded(path, root, excluded_directories)
        )

    records: list[dict[str, Any]] = []
    ids: dict[str, Path] = {}
    valid_manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in manifests:
        try:
            data = load_json(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        manifest_errors = validate_manifest(data, path)
        errors.extend(manifest_errors)
        exercise_id = data.get("id")
        if isinstance(exercise_id, str):
            if exercise_id in ids:
                errors.append(f"id duplicado {exercise_id}: {ids[exercise_id]} y {path}")
            ids[exercise_id] = path
        if not manifest_errors:
            valid_manifests.append((path, data))

    if root.is_dir():
        unregistered_sources = find_unregistered_sources(
            root, manifests, source_extensions, excluded_directories
        )
        errors.extend(f"Fuente sin exercise.json: {source}" for source in unregistered_sources)
    else:
        unregistered_sources = []

    current_os = normalize_os()
    current_arch = normalize_architecture()
    active_count = sum(data["status"] == "active" for _, data in valid_manifests)
    if active_count < minimum_active:
        errors.append(
            f"Hay {active_count} ejercicios activos; la política exige al menos {minimum_active}"
        )
    selected_active_count = sum(
        data["status"] == "active"
        and (args.capability is None or args.capability in data["requirements"]["capabilities"])
        for _, data in valid_manifests
    )
    if args.capability and selected_active_count < minimum_by_capability.get(args.capability, 0):
        errors.append(
            f"Hay {selected_active_count} ejercicios activos con capacidad {args.capability}; "
            f"la política exige al menos {minimum_by_capability[args.capability]}"
        )

    compiled = 0
    tested = 0
    skipped = 0
    failures = 0
    planned = 0
    retired = 0

    for path, data in valid_manifests:
        record: dict[str, Any] = {
            "id": data["id"],
            "manifest": str(path.relative_to(root)),
            "status": data["status"],
            "language": data["language"],
            "standard": data["standard"],
            "capabilities": data["requirements"]["capabilities"],
        }
        if data["status"] == "planned":
            record["validation"] = "planned"
            planned += 1
            records.append(record)
            continue
        if data["status"] == "retired":
            record["validation"] = "retired"
            retired += 1
            records.append(record)
            continue

        accelerator_required = data["platforms"]["accelerator"]
        accelerator_supported = (
            accelerator_required == "none"
            or accelerator_required == args.accelerator
            or (accelerator_required == "any" and args.accelerator != "none")
        )
        supported = (
            current_os in data["platforms"]["operating_systems"]
            and current_arch in data["platforms"]["architectures"]
            and accelerator_supported
        )
        selected = args.capability is None or args.capability in data["requirements"]["capabilities"]
        if not supported or not selected:
            record["validation"] = "skipped"
            record["reason"] = "unsupported_platform" if not supported else "capability_filter"
            skipped += 1
            records.append(record)
            continue

        missing_tools = [tool for tool in data["requirements"]["tools"] if shutil.which(tool) is None]
        if missing_tools:
            record["validation"] = "missing_tools"
            record["missing_tools"] = missing_tools
            errors.append(f"{data['id']}: faltan herramientas: {', '.join(missing_tools)}")
            failures += 1
            records.append(record)
            continue

        source_dir = path.parent
        if not (source_dir / "CMakeLists.txt").is_file():
            record["validation"] = "missing_cmake"
            errors.append(f"{data['id']}: no existe {source_dir / 'CMakeLists.txt'}")
            failures += 1
            records.append(record)
            continue

        if not args.compile_exercises:
            record["validation"] = "inventory_valid"
            records.append(record)
            continue

        status, commands = build_exercise(
            data,
            source_dir,
            args.build_root.resolve() / data["id"],
            args.run_tests,
        )
        record["validation"] = status
        record["commands"] = commands
        if status == "compiled":
            compiled += 1
        elif status == "tested":
            compiled += 1
            tested += 1
        else:
            failures += 1
            errors.append(f"{data['id']}: {status}")
        records.append(record)

    if errors:
        result = "failed"
    elif active_count == 0:
        result = "inventory-valid-no-active-exercises"
    else:
        result = "passed"

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "managed_root": str(root),
        "policy": str(args.policy.resolve()),
        "platform": {"operating_system": current_os, "architecture": current_arch},
        "selection": {"capability": args.capability, "accelerator": args.accelerator},
        "summary": {
            "manifests": len(manifests),
            "active": active_count,
            "planned": planned,
            "retired": retired,
            "source_files": sum(
                1
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in source_extensions
                and not excluded(path, root, excluded_directories)
            ) if root.is_dir() else 0,
            "compiled": compiled,
            "tested": tested,
            "skipped": skipped,
            "failures": failures,
        },
        "unregistered_sources": unregistered_sources,
        "exercises": records,
        "errors": errors,
        "result": result,
    }
    write_report(args.report.resolve(), report)
    append_github_summary(report)

    print(
        f"{result}: {active_count} activos, {compiled} compilados, "
        f"{tested} probados, {len(errors)} errores"
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
