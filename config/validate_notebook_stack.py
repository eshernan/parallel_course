"""Valida las versiones directas del stack de notebooks sin instalar paquetes."""

from importlib import metadata
import sys


EXPECTED = {
    "jupyterlab": "4.6.2",
    "jupytext": "1.19.4",
    "matplotlib": "3.11.1",
    "nbconvert": "7.17.1",
    "numpy": "2.5.1",
    "pandas": "3.0.3",
}


def main() -> int:
    problems: list[str] = []
    for package, expected in EXPECTED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package}: no instalado (esperado {expected})")
            continue
        if actual != expected:
            problems.append(f"{package}: {actual} (esperado {expected})")

    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 1

    versions = ", ".join(f"{name}={version}" for name, version in EXPECTED.items())
    print(f"Stack de notebooks verificado: {versions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
