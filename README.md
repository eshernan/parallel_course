# Programación Paralela 2026

[![Portabilidad de ejercicios](https://github.com/eshernan/parallel_course/actions/workflows/exercise-portability.yml/badge.svg)](https://github.com/eshernan/parallel_course/actions/workflows/exercise-portability.yml)

Material docente para una asignatura universitaria de programación paralela. El curso estudia los fundamentos del área y su aplicación en CPU multinúcleo, clústeres y GPU, con énfasis en corrección, medición y análisis de rendimiento.

**Autor:** [Esteban Hernández B., PhD.](https://eshernan.github.io/)

**Perfiles:** [LinkedIn — HPC Colombia](https://www.linkedin.com/in/hpccol/) · [Página personal](https://eshernan.github.io/)

![Mapa técnico del curso: fundamentos, CPU, OpenMP, MPI, offload, CUDA con tiles y bibliotecas, programación híbrida y proyecto](docs/images/programacion-paralela-curso-2026-v2.png)

El mapa representa la secuencia de los módulos 01 a 08. El módulo 00 es transversal: establece los lenguajes, los compiladores y el entorno con los que se desarrollan los demás temas.

## Navegación del curso

Para navegar el material se debe seguir el **[Índice de navegación del curso](INDICE_CURSO.md)**. Allí se encuentran los temas en el orden de estudio, los notebooks definidos, los enlaces disponibles, los ejemplos, los ejercicios y el camino hacia el material de profundización. Cada notebook incorporado deberá ofrecer, al inicio y al cierre, un enlace de regreso a ese índice.

## Idioma de esta edición

La actualización 2026 se desarrolla en español porque la experiencia docente que sustenta el curso corresponde principalmente a grupos de estudiantes hispanohablantes. Una vez consolidada esta edición académica se preparará una versión en inglés, con la misma estructura, programas, experimentos y referencias.

## Descripción

La actualización 2026 reestructura el material para que la relación entre conceptos, programas, ejercicios, pruebas y evidencia experimental sea explícita. La asignatura está organizada en 19 semanas, con dos sesiones semanales de hasta dos horas. Esto corresponde a 38 sesiones y 76 horas de trabajo presencial. Los laboratorios y el proyecto requieren, además, trabajo independiente por parte del estudiante.

La secuencia comienza con modelos de costo y arquitectura de memoria. Luego se estudian Pthreads, concurrencia en C++20, OpenMP y MPI. El trabajo con aceleradores incluye OpenMP target y un módulo completo de CUDA C++. Las últimas semanas se dedican a programación híbrida, perfilado y desarrollo de un proyecto reproducible.

El repositorio incluye, por fuera del programa regular, [material de profundización](topicos_avanzados/README.md) sobre AMD ROCm/HIP, SYCL, Kokkos y RAJA. Estos temas pueden trabajarse en seminarios, cursos intersemestrales o proyectos de investigación y no modifican las 19 semanas ni las evaluaciones de la asignatura.

En las prácticas de laboratorio, toda medición parte de una implementación cuya corrección ya fue establecida frente a una referencia. Con esa base se estudian tiempos, escalabilidad y uso de recursos; las optimizaciones se sustentan con evidencia experimental y con una explicación del comportamiento observado en el hardware y en el sistema.

## Estructura académica

| Módulo | Contenido principal | Sesiones |
|---:|---|---:|
| 00 | Estándares C/C++, compiladores por sistema operativo, entorno reproducible, CPU, NUMA y afinidad | 2 |
| 01 | Fundamentos, Amdahl, Gustafson, escalado, memoria, Roofline y medición | 4 |
| 02 | Pthreads, sincronización, C++20, atomics y patrones de memoria compartida | 4 |
| 03 | OpenMP: datos, bucles, scheduling, reducciones, SIMD y tareas | 5 |
| 04 | MPI: punto a punto, no bloqueante, colectivas, topologías y Slurm | 6 |
| 05 | Aceleradores portables con OpenMP target | 3 |
| 06 | CUDA C++: modelo SIMT, memoria, tiling, reducciones, streams, bibliotecas aceleradas y Nsight | 6 |
| 07 | MPI+OpenMP, MPI+GPU, perfilado integral y diseño híbrido | 4 |
| 08 | Implementación, revisión de reproducibilidad, presentación y defensa del proyecto final | 4 |
| | **Total** | **38** |

Cada módulo se desarrolla en uno, dos o tres notebooks, según la extensión del tema. Allí se articulan la discusión conceptual, el modelo de ejecución del sistema, las instrucciones de compilación, los programas de ejemplo y el análisis de datos obtenidos en los experimentos.

## Distribución semana a semana

| Semana | Sesiones | Contenido y entregables |
|---:|---:|---|
| 1 | 1–2 | Presentación, diagnóstico, evolución de C/C++, soporte en Linux, macOS y Windows, entorno Linux, GCC/CMake, procesos, hilos, CPU, NUMA y afinidad. |
| 2 | 3–4 | Concurrencia y paralelismo, taxonomía de Flynn, descomposición, trabajo, span, Amdahl y Gustafson. |
| 3 | 5–6 | Cachés, coherencia, localidad, false sharing, vectorización, Roofline y metodología de medición. Evaluación de fundamentos. |
| 4 | 7–8 | Creación y ciclo de vida de Pthreads, partición, mutex, variables de condición, barreras y productor-consumidor. |
| 5 | 9–10 | `std::jthread`, RAII, atomics, modelo de memoria, carreras, deadlock y patrones compartidos. Laboratorio Pthreads/C++20. |
| 6 | 11–12 | Modelo fork-join de OpenMP, regiones paralelas, alcance de datos, `default(none)`, scheduling y afinidad. |
| 7 | 13–14 | Reducciones, atomics, SIMD, tareas, dependencias, granularidad y DAG. |
| 8 | 15–16 | Laboratorio OpenMP; introducción a MPI, comunicadores, rangos y ejecución con MPICH. |
| 9 | 17–18 | Comunicación punto a punto, tags, estados, deadlock, `MPI_Sendrecv` y comunicación no bloqueante. |
| 10 | 19–20 | Colectivas, costos, tipos derivados, topologías cartesianas y nociones de MPI-IO. |
| 11 | 21–22 | Escalado MPI y Slurm; inicio de aceleradores, offload, dispositivos y portabilidad. Laboratorio MPI. |
| 12 | 23–24 | OpenMP target, mapeo de datos, reducción, fallback en CPU y análisis transferencia/cómputo. |
| 13 | 25–26 | CUDA 13, `nvcc`, SIMT, grid/bloque/hilo, manejo de errores, memoria y coalescencia. |
| 14 | 27–28 | Memoria compartida, bancos, tiling, multiplicación de matrices, reducciones, primitivas de warp, Thrust y CUB. |
| 15 | 29–30 | cuBLAS/Lt, cuFFT, cuSPARSE, cuSOLVER y cuRAND; streams, ocupación, Nsight, Compute Sanitizer y laboratorio CUDA. |
| 16 | 31–32 | MPI+OpenMP, niveles de soporte de hilos, afinidad, MPI+CUDA/OpenMP target y asignación proceso-dispositivo. |
| 17 | 33–34 | Perfilado de extremo a extremo, comunicación/cómputo, reproducibilidad y revisión del diseño híbrido. |
| 18 | 35–36 | Clínica del proyecto, revisión cruzada de reproducibilidad y ensayo de defensa. |
| 19 | 37–38 | Presentaciones, defensa, retrospectiva y cierre del curso. |

La descripción sesión por sesión, evaluaciones, rúbricas y bibliografía están en la [planeación completa](docs/PLANEACION_CURSO.md).

## Tecnologías incluidas

### Lenguajes y estándares

- C17 — ISO/IEC 9899:2018.
- C++20 — ISO/IEC 14882:2020.
- C23 y C++23 son las revisiones publicadas vigentes; C2y y C++26 se estudian como trabajo en curso, no como requisito de las entregas.
- POSIX.1-2024 para Pthreads.
- OpenMP 5.2 como versión normativa del curso.
- MPI 5.0 como versión normativa.
- CUDA C++20 para el módulo NVIDIA.
- Python 3.14.6 para notebooks, análisis de resultados y visualización.

### Toolchain e implementaciones

- GCC/G++ 15.3.0 como compiladores abiertos principales.
- MPICH 5.0.1 como implementación MPI de referencia.
- CUDA Toolkit 13.0.x con GCC 15 como compilador host y GPU mínima Turing (`sm_75`).
- Bibliotecas CUDA 13: Thrust/CUB 3.0.1, cuBLAS/cuBLASLt, cuFFT, cuSPARSE, cuSOLVER y cuRAND.
- CMake 3.31, CTest y Ninja para construcción y pruebas.
- Slurm para ejecución en clúster.
- JupyterLab, NumPy, pandas y Matplotlib para explicación y gráficas.
- ThreadSanitizer, AddressSanitizer, `perf`, Compute Sanitizer, Nsight Systems y Nsight Compute para diagnóstico y perfilado.

La [hoja de ruta de C y C++](docs/ESTANDARES_C_CPP.md) resume las versiones publicadas, el estado de las revisiones en preparación y el soporte de GCC, Clang/Apple Clang y MSVC en Linux, macOS y Windows. El curso conserva C17/C++20 como base común: se trata de una decisión de reproducibilidad y compatibilidad con el entorno paralelo, no de desconocimiento de C23 o C++23.

### Material de profundización

- AMD ROCm 7.2.3 e HIP 7.2.3.
- SYCL 2020, revisión 11, con AdaptiveCpp 25.10.0 como implementación abierta de referencia.
- Kokkos 5.1.1 para espacios de ejecución/memoria, `View` y políticas paralelas.
- RAJA 2025.12.2 para segmentos, políticas, kernels y recursos.

Estas tecnologías se estudian fuera del calendario de 19 semanas. Cada backend requiere su propia configuración y hardware compatible. La portabilidad del código se analizará por separado de la portabilidad del rendimiento, pues una misma configuración rara vez resulta óptima en arquitecturas distintas.

CUDA es un tema obligatorio al que se asignan seis sesiones. Se estudian el modelo de programación, la escritura y validación de kernels, el trabajo por mosaicos, las primitivas jerárquicas y las bibliotecas aceleradas. Para cada biblioteca se revisan su propósito, la forma de integrarla y las condiciones en las que resulta conveniente. También se miden los costos de preparación, transferencia y conversión de datos. GCC continúa como compilador abierto principal de la asignatura; `nvcc`, el runtime y las herramientas de NVIDIA se emplean en el módulo CUDA.

## Configuración global

La configuración común de compiladores, estándares, MPI, CUDA y versiones está en [`config/course-toolchain.cmake`](config/course-toolchain.cmake). Las dependencias se verifican en [`cmake/CourseDependencies.cmake`](cmake/CourseDependencies.cmake) y el entorno de los notebooks se fija en [`config/requirements.lock`](config/requirements.lock).

Presets disponibles:

```console
cmake --preset course-cpu
cmake --build --preset course-cpu
ctest --preset course-cpu
```

Para un nodo con CUDA 13 y GPU compatible:

```console
cmake --preset course-cuda
cmake --build --preset course-cuda
ctest --preset course-cuda
```

La guía de variables `COURSE_CC`, `COURSE_CXX`, `COURSE_MPI_ROOT` y `COURSE_CUDA_ROOT` está en [`config/README.md`](config/README.md).

Los presets anteriores comprueban el entorno fijado para dictar la asignatura. La validación de portabilidad utiliza, además, los compiladores nativos de Linux, macOS y Windows para detectar dependencias accidentales de una sola plataforma.

## Validación de ejercicios y reproducibilidad

Cada ejercicio activo debe declarar en `exercise.json` su lenguaje, estándar, dependencias, objetivo de CMake, sistemas operativos, arquitecturas y hardware requerido. El validador rechaza fuentes sin manifiesto, construye por separado cada objetivo, ejecuta sus pruebas CTest y produce un informe JSON acompañado por el inventario del equipo utilizado.

La matriz automática de GitHub Actions cubre Linux, macOS y Windows en x86-64 y ARM64. Las ejecuciones que dependen de un fabricante o acelerador —CPU AMD, CUDA y ROCm/HIP— se realizan mediante runners institucionales conectados al hardware real. La etiqueta del runner se contrasta con la CPU o GPU detectada antes de aceptar el resultado.

Antes de ejecutar cualquier ejercicio se realiza la comprobación previa desde la raíz del repositorio:

```console
python3 validation/preflight.py
```

El comando identifica la plataforma, comprueba C17/C++20 y construye los ejercicios compatibles. Si encuentra una arquitectura no admitida, una dependencia ausente, una fuente sin registrar o una prueba fallida, termina con un código de error y no autoriza continuar. Para un nodo específico se utiliza, por ejemplo, `--profile linux-intel`, `--profile linux-amd`, `--profile linux-arm64`, `--profile linux-nvidia-cuda` o `--profile linux-amd-rocm`.

En el estado actual, las carpetas de ejercicios contienen la planeación temática, pero aún no tienen fuentes activas; por ello el informe correcto es **0 ejercicios activos, 0 compilados**. El flujo prueba su propio mecanismo con programas mínimos C17 y C++20, que no se contabilizan como ejercicios. A medida que se incorpora cada actividad, se agrega su manifiesto y se aumenta el mínimo exigido por la política. El [protocolo de reproducibilidad](docs/REPRODUCIBILIDAD_EJERCICIOS.md) presenta el procedimiento completo, los niveles de evidencia y la configuración para Intel, AMD, ARM, NVIDIA y AMD GPU.

### Plataforma comprobada para esta revisión

La revisión del 2 de agosto de 2026 se probó localmente en **macOS 26.5.2, ARM64, Apple M4 (10 núcleos)**, con Apple Clang 21.0.0 y CMake 3.31.5. Los controles C17 y C++20 compilaron y sus dos pruebas CTest terminaron satisfactoriamente.

El [flujo de GitHub Actions 30752349549](https://github.com/eshernan/parallel_course/actions/runs/30752349549), correspondiente al commit `844cab1`, ejecutó el mismo preflight en seis combinaciones:

| Sistema del runner | Arquitectura | CPU observada en esa ejecución | Compilador utilizado por CMake | Resultado del control |
|---|---:|---|---|---:|
| Ubuntu 24.04 | x86-64 | Intel Xeon Platinum 8573C | GCC 13.3 | 2/2 CTest |
| Ubuntu 24.04 | ARM64 | ARM `aarch64` | GCC 13.3 | 2/2 CTest |
| macOS 15 | x86-64 | Intel Core i7-8700B | Apple Clang 17 | 2/2 CTest |
| macOS 15 | ARM64 | Apple M1 virtual | Apple Clang 17 | 2/2 CTest |
| Windows Server 2025 | x86-64 | AMD64 Family 25 | MSVC 19.51 | 2/2 CTest |
| Windows 11 | ARM64 | ARMv8 | MSVC 19.44 | 2/2 CTest |

Esta evidencia acredita el mecanismo y los programas de control en esas plataformas; el inventario docente continúa en **0 ejercicios activos** y, por tanto, todavía no acredita ejercicios concretos. Los fabricantes observados en runners x86-64 se registran como evidencia de esa ejecución, no como una garantía de asignación futura de GitHub. CUDA, ROCm/HIP y el comportamiento sobre hardware institucional se documentarán únicamente después de ejecutar sus perfiles en los dispositivos correspondientes.

## Organización del repositorio

```text
parallel_course/
├── .github/workflows/               # Portabilidad y pruebas en hardware institucional
├── CMakeLists.txt
├── CMakePresets.json
├── INDICE_CURSO.md                  # Punto de entrada y navegación por notebooks
├── config/                         # Versiones, toolchain y entorno Python
├── cmake/                          # Descubrimiento y validación de librerías
├── curso/
│   ├── notebooks/                  # 1–3 notebooks por tema
│   │   ├── 00_entorno/
│   │   ├── 01_fundamentos/
│   │   ├── 02_memoria_compartida/
│   │   ├── 03_openmp/
│   │   ├── 04_mpi/
│   │   ├── 05_openmp_target/
│   │   ├── 06_cuda/
│   │   ├── 07_hibrido/
│   │   └── 08_proyecto/
│   ├── ejemplos/                   # Fuentes seriales y paralelos compilables
│   └── ejercicios/
│       ├── <tema>/<ejercicio>/     # Enunciado, esqueleto y pruebas públicas
│       └── soluciones/
│           └── <tema>/<ejercicio>/ # Solución y pruebas docentes
├── topicos_avanzados/               # ROCm/HIP, SYCL, Kokkos y RAJA; fuera del curso base
│   ├── notebooks/
│   ├── ejemplos/
│   └── ejercicios/soluciones/
├── docs/                            # Planeación, reproducibilidad, estándares e imágenes
└── validation/                      # Manifiestos, inventario de plataforma y compilación verificable
```

## Relación entre notebooks, ejemplos y ejercicios

El trabajo de cada tema relaciona cuatro componentes:

```text
Notebook conceptual
        ↓ ejecuta
Ejemplo serial + ejemplo paralelo
        ↓ producen datos y pruebas
Ejercicio con esqueleto y pruebas públicas
        ↓ se evalúa contra
Solución de referencia + pruebas docentes
```

### Ejemplos

Cada ejemplo incluye:

- fuente serial de referencia y versión paralela;
- construcción con CMake;
- manejo explícito de errores;
- pruebas de corrección antes de medir;
- exportación de métricas a CSV o JSON;
- README con compilación, ejecución y hardware requerido.

### Ejercicios

Cada ejercicio incluye enunciado, resultados esperados, código inicial, un conjunto pequeño de datos, salida de referencia, pruebas públicas, rúbrica y un manifiesto `exercise.json`. La valoración del rendimiento procede cuando las pruebas de corrección han sido superadas. La integración continua comprueba que el objetivo declarado construya en las plataformas admitidas.

### Respuestas y soluciones

Las soluciones se almacenan en `curso/ejercicios/soluciones/<tema>/<ejercicio>/`. Durante el semestre se mantienen en una rama privada o se excluyen de la distribución entregada a los estudiantes. Cada solución documenta la descomposición, la sincronización, la validación, la complejidad y el diseño del experimento.

## Lineamiento visual de notebooks

La ilustración de la cabecera establece el lenguaje visual para las imágenes conceptuales del curso:

- boceto técnico hecho a lápiz de grafito;
- papel marfil o fondo blanco cálido;
- líneas finas, flechas y pequeñas anotaciones de ingeniería;
- sombreado mediante rayado suave;
- composición limpia y académica;
- grafito monocromático con acentos azul grisáceo discretos;
- sin fotografías, render 3D, colores saturados ni estética corporativa brillante.

Las gráficas cuantitativas se generan a partir de los datos del experimento, con fondo claro y una paleta sobria. Las capturas de herramientas como Nsight se reservan para observaciones que no puedan expresarse mejor con una gráfica propia y se presentan sin modificaciones que alteren la evidencia.

## Estado de desarrollo

- Planeación de 38 sesiones: completada.
- Configuración global y presets: creados.
- Estructura de notebooks, ejemplos, ejercicios y soluciones: creada.
- Protocolo, manifiestos y flujos de validación multiplataforma: creados; el inventario actual contiene 0 ejercicios activos y la política se incrementará con cada actividad incorporada.
- Edición en inglés: prevista después de consolidar y revisar la edición académica en español.
- Material de profundización en ROCm/HIP, SYCL, Kokkos y RAJA: planeado y configurado; contenidos ejecutables pendientes.
- Desarrollo de todos los notebooks y soluciones: pendiente por tema.

Documentos de referencia:

- [Planeación semestral, evaluaciones y bibliografía](docs/PLANEACION_CURSO.md).
- [Índice de navegación del curso](INDICE_CURSO.md).
- [Protocolo de compilación y reproducibilidad de ejercicios](docs/REPRODUCIBILIDAD_EJERCICIOS.md).
- [Estándares de C y C++ y soporte por plataforma](docs/ESTANDARES_C_CPP.md).
- [Organización del material docente](curso/README.md).
- [Planeación del material de profundización](topicos_avanzados/README.md).

## Política de mantenimiento documental

Todo cambio de módulos, contenidos, semanas, tecnologías, notebooks, ejemplos, ejercicios o soluciones se acompaña de la actualización correspondiente en este README y en `INDICE_CURSO.md`. Cuando se incorpora o retira un ejercicio también se ajustan su manifiesto y el mínimo de la política de validación. Cuando se modifica la estructura académica, se revisa la ilustración de cabecera y se genera una nueva versión si la anterior dejó de representar el curso.

La revisión editorial utiliza español académico, directo y propio del contexto universitario colombiano. Esta es la edición principal mientras se prepara la versión en inglés. Se evitan eslóganes, fórmulas impersonales y afirmaciones generales que no estén respaldadas por el diseño del curso. Antes de cerrar un cambio se comprueban la suma de sesiones, la secuencia de semanas, los enlaces locales, la imagen referenciada y la correspondencia entre este resumen y `docs/PLANEACION_CURSO.md`.

## Autoría

La concepción académica, la selección de contenidos y la actualización 2026 de este curso son autoría de **Esteban Hernández B., PhD.**

- LinkedIn: [https://www.linkedin.com/in/hpccol/](https://www.linkedin.com/in/hpccol/)
- Página personal: [https://eshernan.github.io/](https://eshernan.github.io/)

El material de terceros conserva sus avisos y licencias correspondientes. Consulte [LICENSE](LICENSE) y la documentación de procedencia antes de redistribuir componentes externos.
