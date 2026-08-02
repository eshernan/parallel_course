#!/usr/bin/env python3
"""Registra y verifica la plataforma utilizada para construir los ejercicios."""

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


def normalize_os() -> str:
    raw = platform.system().lower()
    if raw == "darwin":
        return "macos"
    if raw.startswith("win"):
        return "windows"
    return raw


def command_output(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "available": True,
            "return_code": completed.returncode,
            "output": completed.stdout.strip()[:8000],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "return_code": None, "output": str(exc)}


def first_line(command: list[str]) -> dict[str, Any]:
    result = command_output(command)
    output = result["output"]
    result["output"] = output.splitlines()[0] if output else ""
    return result


def linux_cpu_info() -> tuple[str, str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unknown", platform.processor() or "unknown"
    values: dict[str, str] = {}
    with cpuinfo.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if ":" not in line:
                if values:
                    break
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            values.setdefault(key, value)
    raw_vendor = values.get("vendor_id", values.get("CPU implementer", "unknown"))
    model = values.get("model name", values.get("Processor", platform.processor() or "unknown"))
    if raw_vendor == "GenuineIntel":
        vendor = "intel"
    elif raw_vendor == "AuthenticAMD":
        vendor = "amd"
    elif normalize_architecture() in {"arm64", "arm32"}:
        vendor = "arm"
    else:
        vendor = raw_vendor.lower() if raw_vendor != "unknown" else "unknown"
    return vendor, model


def macos_cpu_info() -> tuple[str, str]:
    architecture = normalize_architecture()
    brand = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    model = brand["output"] if brand["return_code"] == 0 else ""
    if architecture == "arm64":
        if not model:
            hardware = command_output(["system_profiler", "SPHardwareDataType", "-json"])
            if hardware["return_code"] == 0:
                try:
                    record = json.loads(hardware["output"])["SPHardwareDataType"][0]
                    model = record.get("chip_type", "")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    model = ""
        return "apple", model or "Apple Silicon"
    vendor = "intel" if "intel" in model.lower() else "unknown"
    return vendor, model or platform.processor() or "unknown"


def windows_cpu_info() -> tuple[str, str]:
    model = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown")
    lower = model.lower()
    if "intel" in lower:
        vendor = "intel"
    elif "amd" in lower:
        vendor = "amd"
    elif "arm" in lower or normalize_architecture() == "arm64":
        vendor = "arm"
    else:
        vendor = "unknown"
    return vendor, model


def cpu_info(operating_system: str) -> tuple[str, str]:
    if operating_system == "linux":
        return linux_cpu_info()
    if operating_system == "macos":
        return macos_cpu_info()
    if operating_system == "windows":
        return windows_cpu_info()
    return "unknown", platform.processor() or "unknown"


def tool_inventory() -> dict[str, Any]:
    tools: dict[str, list[str]] = {
        "cmake": ["cmake", "--version"],
        "ctest": ["ctest", "--version"],
        "ninja": ["ninja", "--version"],
        "cc": ["cc", "--version"],
        "c++": ["c++", "--version"],
        "gcc": ["gcc", "--version"],
        "g++": ["g++", "--version"],
        "clang": ["clang", "--version"],
        "clang++": ["clang++", "--version"],
        "cl": ["cl"],
        "mpicc": ["mpicc", "--version"],
        "mpicxx": ["mpicxx", "--version"],
        "mpiexec": ["mpiexec", "--version"],
        "nvcc": ["nvcc", "--version"],
        "hipcc": ["hipcc", "--version"],
    }
    inventory: dict[str, Any] = {}
    for name, command in tools.items():
        location = shutil.which(name)
        if location is None:
            inventory[name] = {"available": False, "path": None, "version": ""}
            continue
        version = first_line(command)
        inventory[name] = {
            "available": True,
            "path": location,
            "version": version["output"],
            "probe_return_code": version["return_code"],
        }
    return inventory


def accelerator_inventory() -> dict[str, Any]:
    nvidia = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap",
            "--format=csv,noheader",
        ]
    )
    amd = command_output(["rocminfo"], timeout=20)
    if amd["output"]:
        amd["output"] = "\n".join(amd["output"].splitlines()[:120])
    return {"nvidia": nvidia, "amd_rocm": amd}


def github_context() -> dict[str, str]:
    names = [
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_SHA",
        "GITHUB_REF",
        "RUNNER_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
    ]
    return {name: os.environ[name] for name in names if name in os.environ}


def append_github_summary(manifest: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    host = manifest["host"]
    lines = [
        "### Plataforma de validación",
        "",
        f"- Sistema operativo: `{host['operating_system']} {host['release']}`",
        f"- Arquitectura: `{host['architecture']}`",
        f"- CPU: `{host['cpu_vendor']}` — {host['cpu_model']}",
        f"- Núcleos lógicos: {host['logical_cpus']}",
        f"- Verificación solicitada: `{manifest['verification']['result']}`",
        "",
    ]
    with open(summary_path, "a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["linux-intel", "linux-amd", "linux-arm64", "linux-nvidia-cuda", "linux-amd-rocm"],
        help="Perfil institucional; además de seleccionar el runner, se verifica contra el hardware detectado",
    )
    parser.add_argument("--expected-os", choices=["linux", "macos", "windows"])
    parser.add_argument("--expected-arch", choices=["x86_64", "arm64", "arm32"])
    parser.add_argument("--expected-cpu-vendor", choices=["intel", "amd", "apple", "arm"])
    parser.add_argument("--expected-accelerator", choices=["none", "nvidia", "amd"])
    parser.add_argument("--require-command", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = {
        "linux-intel": {"os": "linux", "arch": "x86_64", "cpu": "intel", "accelerator": "none", "commands": []},
        "linux-amd": {"os": "linux", "arch": "x86_64", "cpu": "amd", "accelerator": "none", "commands": []},
        "linux-arm64": {"os": "linux", "arch": "arm64", "cpu": None, "accelerator": "none", "commands": []},
        "linux-nvidia-cuda": {"os": "linux", "arch": "x86_64", "cpu": None, "accelerator": "nvidia", "commands": ["nvcc"]},
        "linux-amd-rocm": {"os": "linux", "arch": "x86_64", "cpu": None, "accelerator": "amd", "commands": ["hipcc"]},
    }
    if args.profile:
        profile = profiles[args.profile]
        args.expected_os = args.expected_os or profile["os"]
        args.expected_arch = args.expected_arch or profile["arch"]
        args.expected_cpu_vendor = args.expected_cpu_vendor or profile["cpu"]
        args.expected_accelerator = args.expected_accelerator or profile["accelerator"]
        args.require_command = [*profile["commands"], *args.require_command]
    operating_system = normalize_os()
    architecture = normalize_architecture()
    cpu_vendor, cpu_model = cpu_info(operating_system)
    tools = tool_inventory()
    accelerators = accelerator_inventory()
    errors: list[str] = []

    if args.expected_os and operating_system != args.expected_os:
        errors.append(f"Se esperaba sistema operativo {args.expected_os}, se detectó {operating_system}")
    if args.expected_arch and architecture != args.expected_arch:
        errors.append(f"Se esperaba arquitectura {args.expected_arch}, se detectó {architecture}")
    if args.expected_cpu_vendor and cpu_vendor != args.expected_cpu_vendor:
        errors.append(f"Se esperaba CPU {args.expected_cpu_vendor}, se detectó {cpu_vendor}")
    for command in args.require_command:
        if shutil.which(command) is None:
            errors.append(f"No se encontró la herramienta requerida: {command}")
    if args.expected_accelerator == "nvidia":
        probe = accelerators["nvidia"]
        if not probe["available"] or probe["return_code"] != 0 or not probe["output"]:
            errors.append("No se pudo verificar una GPU NVIDIA con nvidia-smi")
    elif args.expected_accelerator == "amd":
        probe = accelerators["amd_rocm"]
        if not probe["available"] or probe["return_code"] != 0:
            errors.append("No se pudo verificar una GPU AMD con rocminfo")

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "operating_system": operating_system,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel_platform": platform.platform(),
            "architecture": architecture,
            "machine": platform.machine(),
            "cpu_vendor": cpu_vendor,
            "cpu_model": cpu_model,
            "logical_cpus": os.cpu_count(),
        },
        "tools": tools,
        "accelerators": accelerators,
        "github_actions": github_context(),
        "verification": {
            "profile": args.profile,
            "expected_operating_system": args.expected_os,
            "expected_architecture": args.expected_arch,
            "expected_cpu_vendor": args.expected_cpu_vendor,
            "expected_accelerator": args.expected_accelerator,
            "required_commands": args.require_command,
            "errors": errors,
            "result": "failed" if errors else "passed",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    append_github_summary(manifest)

    print(
        f"{operating_system} {architecture}, CPU {cpu_vendor}: "
        f"{manifest['verification']['result']}"
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
