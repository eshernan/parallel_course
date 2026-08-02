# Configuración global

`course-toolchain.cmake` es la fuente única para compiladores, estándares e implementaciones:

- GCC/G++ 15.3.0;
- C17 y C++20;
- OpenMP 5.2;
- MPICH 5.0.1 / MPI 5.0;
- CUDA 13.0 y arquitecturas `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100` y `sm_120`;
- CPython 3.14.6 y el stack fijado en `requirements.lock`.

El material extracurricular de `topicos_avanzados/` añade perfiles separados para ROCm/HIP 7.2.3, SYCL 2020 revisión 11 con AdaptiveCpp 25.10.0, Kokkos 5.1.1 y RAJA 2025.12.2. Sus versiones y rutas se centralizan en `advanced-topics.cmake`; permanecen deshabilitadas para no convertirlas en requisitos del curso base.

## Variables de entorno admitidas

- `COURSE_CC`: ruta absoluta a `gcc` 15.3.
- `COURSE_CXX`: ruta absoluta a `g++` 15.3.
- `COURSE_MPI_ROOT`: prefijo de instalación de MPICH 5.0.1.
- `COURSE_CUDA_ROOT`: prefijo de CUDA Toolkit 13.0.
- `COURSE_ROCM_ROOT`: prefijo de ROCm/HIP 7.2.3.
- `COURSE_ADAPTIVECPP_ROOT`: prefijo de AdaptiveCpp 25.10.0.
- `COURSE_KOKKOS_ROOT`: prefijo de instalación de Kokkos 5.1.1.
- `COURSE_RAJA_ROOT`: prefijo de instalación de RAJA 2025.12.2.

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

## Tópicos avanzados

La detección se activa de forma explícita y para un solo perfil a la vez:

```console
cmake -S . -B build/advanced-hip \
  -DCMAKE_TOOLCHAIN_FILE=config/course-toolchain.cmake \
  -DCOURSE_ENABLE_ADVANCED_TOPICS=ON \
  -DCOURSE_ADVANCED_PROFILE=HIP
```

Los perfiles admitidos son `HIP`, `SYCL`, `KOKKOS` y `RAJA`. Esta comprobación confirma que el compilador o paquete existe; la compatibilidad con un backend y arquitectura concretos se validará mediante los ejemplos y CTest de cada cápsula. Consulte [`topicos_avanzados/README.md`](../topicos_avanzados/README.md).
