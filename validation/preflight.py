#!/usr/bin/env python3
"""Comprueba la plataforma y construye los ejercicios antes de ejecutarlos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "local-cpu": {"platform": None, "capability": None, "accelerator": "none"},
    "linux-intel": {"platform": "linux-intel", "capability": None, "accelerator": "none"},
    "linux-amd": {"platform": "linux-amd", "capability": None, "accelerator": "none"},
    "linux-arm64": {"platform": "linux-arm64", "capability": None, "accelerator": "none"},
    "linux-nvidia-cuda": {
        "platform": "linux-nvidia-cuda",
        "capability": "cuda",
        "accelerator": "nvidia",
    },
    "linux-amd-rocm": {
        "platform": "linux-amd-rocm",
        "capability": "hip",
        "accelerator": "amd",
    },
}


def run(command: list[str], cwd: Path) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="local-cpu",
        help="Perfil que debe corresponder a la máquina; local-cpu detecta el sistema sin fijar fabricante",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build") / "validation" / "preflight",
    )
    parser.add_argument("--expected-os", choices=["linux", "macos", "windows"])
    parser.add_argument("--expected-arch", choices=["x86_64", "arm64", "arm32"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parent.parent
    output = args.output_dir.resolve()
    profile = PROFILES[args.profile]
    python = sys.executable

    hygiene_command = [
        python,
        str(repository / "validation" / "validate_repository_hygiene.py"),
        "--report",
        str(output / "repository-hygiene.json"),
    ]
    if run(hygiene_command, repository) != 0:
        print("Preflight interrumpido: el repositorio contiene artefactos compilados rastreados.", file=sys.stderr)
        return 1

    canonical_notebooks_command = [
        python,
        str(repository / "tools" / "generate_course_notebooks.py"),
        "--check",
    ]
    if run(canonical_notebooks_command, repository) != 0:
        print("Preflight interrumpido: los notebooks no coinciden con el inventario canónico.", file=sys.stderr)
        return 1

    navigation_command = [
        python,
        str(repository / "validation" / "validate_navigation.py"),
        "--report",
        str(output / "navigation.json"),
    ]
    if run(navigation_command, repository) != 0:
        print("Preflight interrumpido: el índice del curso contiene enlaces inválidos.", file=sys.stderr)
        return 1

    notebooks_command = [
        python,
        str(repository / "validation" / "validate_notebooks.py"),
        "--report",
        str(output / "notebooks.json"),
    ]
    if run(notebooks_command, repository) != 0:
        print("Preflight interrumpido: al menos un notebook no se ejecuta de principio a fin.", file=sys.stderr)
        return 1

    platform_command = [
        python,
        str(repository / "validation" / "platform_manifest.py"),
        "--output",
        str(output / "platform.json"),
        "--require-command",
        "cmake",
    ]
    if profile["platform"]:
        platform_command.extend(["--profile", profile["platform"]])
    if args.expected_os:
        platform_command.extend(["--expected-os", args.expected_os])
    if args.expected_arch:
        platform_command.extend(["--expected-arch", args.expected_arch])
    if run(platform_command, repository) != 0:
        print("Preflight interrumpido: la máquina no corresponde al perfil solicitado.", file=sys.stderr)
        return 1

    harness_command = [
        python,
        str(repository / "validation" / "validate_exercises.py"),
        "--exercises-root",
        str(repository / "validation" / "fixtures"),
        "--policy",
        str(repository / "validation" / "fixtures" / "policy.json"),
        "--build-root",
        str(output / "harness-build"),
        "--report",
        str(output / "harness.json"),
        "--compile",
        "--test",
    ]
    if run(harness_command, repository) != 0:
        print("Preflight interrumpido: el entorno no construye los controles C17/C++20.", file=sys.stderr)
        return 1

    capability_args: list[str] = []
    if profile["capability"]:
        capability_args = ["--capability", profile["capability"]]
    accelerator_args = ["--accelerator", profile["accelerator"]]

    collections = (
        (
            "ejercicios",
            repository / "curso" / "ejercicios",
            repository / "validation" / "policy.json",
        ),
        (
            "soluciones",
            repository / "curso" / "ejercicios" / "soluciones",
            repository / "validation" / "solutions-policy.json",
        ),
    )
    for name, exercises_root, policy in collections:
        command = [
            python,
            str(repository / "validation" / "validate_exercises.py"),
            "--exercises-root",
            str(exercises_root),
            "--policy",
            str(policy),
            "--build-root",
            str(output / f"{name}-build"),
            "--report",
            str(output / f"{name}.json"),
            *capability_args,
            *accelerator_args,
            "--compile",
            "--test",
        ]
        if run(command, repository) != 0:
            print(f"Preflight interrumpido durante la validación de {name}.", file=sys.stderr)
            return 1

    print(f"Preflight satisfactorio. Informes disponibles en {output}")
    print("Puede continuar con los ejercicios compatibles con el perfil seleccionado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
