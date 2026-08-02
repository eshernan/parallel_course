# Configuración global

`course-toolchain.cmake` es la fuente única para compiladores, estándares e implementaciones:

- GCC/G++ 15.3.0;
- C17 y C++20;
- OpenMP 5.2;
- MPICH 5.0.1 / MPI 5.0;
- CUDA 13.0 y arquitecturas `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100` y `sm_120`;
- CPython 3.14.6 y el stack fijado en `requirements.lock`.

## Variables de entorno admitidas

- `COURSE_CC`: ruta absoluta a `gcc` 15.3.
- `COURSE_CXX`: ruta absoluta a `g++` 15.3.
- `COURSE_MPI_ROOT`: prefijo de instalación de MPICH 5.0.1.
- `COURSE_CUDA_ROOT`: prefijo de CUDA Toolkit 13.0.

## Preparación

```console
python3.14 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.lock
cmake --preset course-cpu
cmake --build --preset course-cpu
ctest --preset course-cpu
```

En un nodo Turing o posterior con CUDA 13:

```console
cmake --preset course-cuda
cmake --build --preset course-cuda
ctest --preset course-cuda
```

Las versiones estrictas pueden desactivarse solo para desarrollo del repositorio mediante `-DCOURSE_STRICT_VERSIONS=OFF`; las entregas y la imagen oficial siempre usan validación estricta.
