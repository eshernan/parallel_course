# Planeación semestral: Programación paralela

Fecha de versión: 2 de agosto de 2026.

## 1. Ficha del curso

- Duración: 19 semanas.
- Intensidad: 2 sesiones por semana, 2 horas por sesión.
- Total: 38 sesiones, 76 horas presenciales.
- Trabajo independiente estimado: 4 horas por semana.
- Nivel: pregrado avanzado; adaptable a primer semestre de posgrado.
- Prerrequisitos: programación en C y C++, estructuras de datos, Linux/terminal, compilación separada y nociones de arquitectura de computadores.
- Lenguajes: C17 y C++20. Python se emplea en los notebooks, el análisis de resultados y la elaboración de gráficas.
- Trabajo en clase: discusión conceptual, programa de referencia, práctica incremental y análisis reproducible de resultados.

## 2. Resultados de aprendizaje

Al finalizar el semestre se espera que el estudiante esté en capacidad de:

1. Elegir un modelo de paralelismo según dependencias, memoria, comunicación y hardware.
2. Diseñar programas correctos con Pthreads/C++20, OpenMP y MPI.
3. Explicar carreras, interbloqueos, consistencia de memoria, localidad y sincronización.
4. Paralelizar y optimizar un kernel para CPU multinúcleo, clúster y acelerador.
5. Medir escalado fuerte y débil, aceleración, eficiencia, ancho de banda y error numérico con un diseño experimental explícito.
6. Utilizar herramientas de diagnóstico y perfilado para sustentar cambios en una implementación.
7. Construir, probar y ejecutar una aplicación híbrida reproducible en Linux/Slurm.
8. Comunicar resultados con gráficas, evidencia, limitaciones y conclusiones verificables.

## 3. Entorno fijado

Durante la primera semana se entrega una imagen Linux de referencia y se establece la matriz de versiones que se utilizará durante el semestre. Las actualizaciones se reservan para correcciones que afecten la seguridad o la posibilidad de ejecutar las prácticas.

| Componente | Versión fijada | Uso |
|---|---:|---|
| GCC | 15.3 | Compilador abierto principal para C17/C++20 y OpenMP. Es una versión de corrección de la rama 15 y mantiene compatibilidad como *host compiler* de CUDA 13. |
| C | ISO/IEC 9899:2018 (C17) | `-std=c17 -Wall -Wextra -Wpedantic -Werror` en entregas. |
| C++ | ISO/IEC 14882:2020 (C++20) | `-std=c++20`; `std::jthread`, atomics y biblioteca estándar. |
| POSIX | POSIX.1-2024, Issue 8 | Contrato normativo para Pthreads. |
| OpenMP | 5.2 | Versión normativa enseñada; se documenta cualquier extensión 6.0 usada. |
| MPI | 5.0 | Versión normativa. Se enseña un subconjunto portable y se consulta la especificación. |
| MPICH | 5.0.1 | Implementación abierta de referencia con soporte completo de MPI 5.0. |
| CMake | 3.31.x | Construcción fuera del árbol y CTest. |
| Python | 3.14.6 | Ejecución de notebooks, procesamiento de resultados y gráficas. Los kernels evaluados se implementan en C o C++. |
| CUDA, obligatorio | Toolkit 13.0.x | Itinerario NVIDIA. `nvcc` no es el compilador abierto principal; usa GCC 15 como compilador *host*. Hardware mínimo: Turing, `sm_75`. |
| OpenACC, optativo | 3.4 | Solo comparación con OpenMP target. |
| OpenCL, optativo | 3.1 | Lectura de portabilidad, no API principal del curso. |

Enlaces normativos y de versión:

- [Versiones oficiales de GCC](https://gcc.gnu.org/releases.html).
- [ISO C17](https://www.iso.org/standard/74528.html) y [borrador público C++20 N4860](https://isocpp.org/files/papers/N4860.pdf).
- [POSIX.1-2024, Issue 8](https://pubs.opengroup.org/onlinepubs/9799919799/).
- [OpenMP 5.2 y 6.0](https://www.openmp.org/specifications/). La asignatura adopta 5.2 por su soporte estable en los compiladores seleccionados. Las características de 6.0 se consultan cuando el toolchain las implementa de manera verificable.
- [MPI 5.0](https://www.mpi-forum.org/docs/) y [MPICH 5.0.1](https://www.mpich.org/downloads/).
- [CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/) y [guía del compilador `nvcc`](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/).
- [OpenACC 3.4](https://www.openacc.org/specification).
- [OpenCL 3.1](https://registry.khronos.org/OpenCL/).

`config/course-toolchain.cmake`, utilizado por `CMakePresets.json`, concentra la selección de GCC/G++, los wrappers de MPICH, `mpiexec`, CUDA y los estándares de lenguaje. `cmake/CourseDependencies.cmake` comprueba Threads, OpenMP, MPI, CUDA y el entorno Python fijado en `config/requirements.lock`.

### Política para laboratorios con GPU antigua

CUDA 13 eliminó la compilación fuera de línea para arquitecturas anteriores a `compute capability 7.5`. Cuando un laboratorio dispone únicamente de equipos Maxwell, Pascal o Volta, se utiliza una imagen legada separada, con CUDA 12.9 y un controlador de la rama 580. Los archivos de construcción identifican con claridad cuál de los dos entornos utilizan.

## 4. Contenido académico de cada tema

Cada tema se desarrolla en uno, dos o tres notebooks. Los programas completos permanecen en archivos fuente compilables; el notebook invoca CMake/CTest o ejecutables ya construidos y carga los resultados CSV o JSON empleados en el análisis.

1. Motivación y pregunta guía.
2. Resultados de aprendizaje del tema.
3. Conceptos y modelo de ejecución/memoria.
4. Relación con hardware, sistema operativo, *runtime* y lenguaje.
5. API mínima y versión del estándar, con enlace normativo.
6. Instrucciones exactas de compilación y ejecución.
7. Ejemplo serial de referencia y prueba de corrección.
8. Transformación paralela paso a paso.
9. Experimento reproducible: hipótesis, variables, calentamiento, repeticiones y hardware.
10. Gráficas necesarias, con datos generados por el ejemplo.
11. Interpretación: costo, aceleración, eficiencia, saturación, error e incertidumbre.
12. Errores frecuentes y ejercicio de depuración.
13. Ejercicios, criterios de aceptación y bibliografía.

La selección de gráficas depende de la pregunta experimental. Entre las representaciones previstas están:

- Tiempo frente a tamaño del problema.
- Aceleración y eficiencia frente a hilos/procesos.
- Escalado fuerte y débil.
- Ancho de banda o FLOP/s y Roofline.
- Distribución de tiempos obtenida en varias repeticiones.
- Error numérico frente a paralelismo/precisión.
- Línea base serial y techo ideal claramente identificados.

## 5. Organización de notebooks y fuentes

| Tema | Notebooks previstos (máximo 3) | Ejemplos fuente |
|---|---|---|
| 00. Entorno | `00_entorno_reproducible.ipynb` | `hello_c`, `hello_cpp`, inventario de CPU/NUMA, CMake/CTest. |
| 01. Fundamentos | `01_modelos.ipynb`, `02_escalabilidad.ipynb`, `03_memoria_roofline.ipynb` | suma serial, microbenchmark de memoria, *false sharing*. |
| 02. Memoria compartida | `01_pthreads.ipynb`, `02_sincronizacion.ipynb`, `03_cpp20_atomics.ipynb` | creación/join, productor-consumidor, deadlock, reducción, cola de trabajo. |
| 03. OpenMP | `01_modelo_datos.ipynb`, `02_bucles_reducciones.ipynb`, `03_tareas_rendimiento.ipynb` | π, histograma, *stencil*, mergesort con tareas. |
| 04. MPI | `01_punto_a_punto.ipynb`, `02_colectivas_topologias.ipynb`, `03_escalabilidad_slurm.ipynb` | hello, anillo, π, halo 1D/2D y E/S básica. |
| 05. Aceleradores portables | `01_modelo_offload.ipynb`, `02_openmp_target.ipynb` | detección de dispositivo, vector add y reducción con OpenMP target y fallback validado en CPU. |
| 06. CUDA C++ | `01_modelo_cuda.ipynb`, `02_memoria_tiling.ipynb`, `03_bibliotecas_perfiles.ipynb` | vector add, reducción y GEMM ingenuo/tiled validados; Thrust/CUB, cuBLAS, cuFFT, cuSPARSE, cuSOLVER y cuRAND; errores, streams, ocupación y Nsight. |
| 07. Híbrido | `01_mpi_openmp.ipynb`, `02_mpi_gpu.ipynb`, `03_perfilado_reproducible.ipynb` | stencil híbrido, afinidad, CPU+GPU, Roofline y Slurm. |
| 08. Proyecto | `01_guia_proyecto.ipynb` | plantilla de proyecto con referencia serial, pruebas y exportación de métricas. |

## 6. Calendario de 38 sesiones

| Sesión | Tema | Actividad y evidencia |
|---:|---|---|
| 1 | Entorno reproducible | Presentación, diagnóstico, imagen Linux, versiones, estructura del repositorio y primer CMake/CTest. |
| 2 | Sistema | Procesos, hilos, CPU/NUMA, afinidad, inventario de hardware y ejecución por lotes. Evaluación diagnóstica aprobada/no aprobada. |
| 3 | Fundamentos | Concurrencia, paralelismo, taxonomía de Flynn, descomposición de datos/tareas y dependencias. |
| 4 | Rendimiento | Trabajo, *span*, overhead, Amdahl, Gustafson, escalado fuerte/débil y eficiencia. |
| 5 | Memoria | Cachés, coherencia, NUMA, localidad, *false sharing*, vectorización e intensidad aritmética. |
| 6 | Medición | Metodología experimental, Roofline, variabilidad, precisión y reporte. **Evaluación 1**. |
| 7 | Pthreads | Ciclo de vida, argumentos, partición, `join`, errores y referencia serial. |
| 8 | Sincronización | Mutex, condición, barrera, productor-consumidor y modelo *happens-before*. |
| 9 | Concurrencia C++20 | `std::jthread`, RAII, atomics, órdenes de memoria; carreras y ThreadSanitizer. |
| 10 | Patrones compartidos | Reducción, cola de trabajo, deadlock/livelock y revisión de código. **Evaluación 2**. |
| 11 | OpenMP básico | Modelo fork-join, regiones, `parallel for`, alcance de datos y `default(none)`. |
| 12 | Distribución | `schedule`, `collapse`, balance, afinidad, localidad y *false sharing*. |
| 13 | Reducciones y SIMD | `reduction`, `atomic`, `critical`, SIMD, precisión y validación. |
| 14 | Tareas | `task`, dependencias, DAG, granularidad y mergesort/*stencil*. |
| 15 | OpenMP aplicado | Perfilado, escalado y defensa breve del laboratorio. **Evaluación 3**. |
| 16 | MPI básico | Procesos, comunicadores, rangos, MPICH, `mpiexec` y primer programa portable. |
| 17 | Punto a punto | Bloqueante, tags, estado, interbloqueo y `MPI_Sendrecv`. |
| 18 | No bloqueante | `MPI_Isend/Irecv`, espera, progreso, solapamiento y halo. |
| 19 | Colectivas | Broadcast, scatter/gather, reduce/allreduce y costo algorítmico. |
| 20 | Estructuras MPI | Tipos derivados, topologías cartesianas y nociones de MPI-IO. |
| 21 | Clúster y escala | Slurm, mapeo, escalado fuerte/débil y reporte reproducible. **Evaluación 4**. |
| 22 | Aceleradores portables | Modelos de offload, dispositivos, memoria, portabilidad y límites del hardware. |
| 23 | OpenMP target | Regiones `target`, mapeo de datos, fallback en CPU y compilación con GCC. |
| 24 | OpenMP target aplicado | Reducción, transferencia/cómputo, validación y perfil. **Evaluación 5**. |
| 25 | CUDA: entorno y modelo | CUDA 13, `nvcc`, SIMT, jerarquía hilo-bloque-grid, primer kernel y manejo obligatorio de errores. |
| 26 | CUDA: memoria | Memoria host/device, transferencias, memoria unificada, coalescencia y medición con eventos. |
| 27 | CUDA: localidad y tiles | Memoria compartida, bancos, *tiling*, tratamiento de bordes y GEMM ingenuo/tiled contra referencia CPU. |
| 28 | CUDA: primitivas | Divergencia, atomics, warp, reducciones, Thrust/CUB, sincronización y precisión numérica. |
| 29 | CUDA: bibliotecas | Ciclo handle/plan/descriptor/workspace; cuBLAS/cuBLASLt, cuFFT y cuSPARSE con ejemplos guiados. |
| 30 | CUDA: selección y rendimiento | cuSOLVER y cuRAND; streams, ocupación, Nsight, Compute Sanitizer y comparación kernel/biblioteca incluyendo costo extremo a extremo. **Evaluación 6**. |
| 31 | Híbrido CPU | `MPI_Init_thread`, MPI+OpenMP, proceso/hilo/núcleo y afinidad. |
| 32 | Híbrido CPU+GPU | MPI+CUDA/OpenMP target, asignación proceso-dispositivo y comunicación de halos. |
| 33 | Rendimiento integral | Perfil de extremo a extremo, comunicación/cómputo, I/O, energía como extensión y reproducibilidad. |
| 34 | Taller híbrido | Revisión de diseño, referencia serial, pruebas y plan experimental. **Hito híbrido**. |
| 35 | Proyecto final | Clínica de implementación y revisión cruzada de arquitectura. |
| 36 | Proyecto final | Revisión de reproducibilidad, ensayo de defensa y consolidación de resultados. |
| 37 | Proyecto final | Presentaciones y defensa, grupo A. |
| 38 | Proyecto final | Presentaciones, grupo B; retrospectiva y cierre. **Evaluación final**. |

## 7. Evaluaciones

| Evaluación | Peso | Entrega mínima |
|---|---:|---|
| Diagnóstico de entorno | 0 %, requisito | Compila C17/C++20, ejecuta CTest y registra hardware/versiones. |
| 1. Informe de rendimiento | 10 % | Amdahl/Gustafson, escalado, gráfica reproducible y crítica de medición. |
| 2. Laboratorio Pthreads/C++20 | 10 % | Versión serial y paralela, prueba de carrera, corrección, speedup y revisión cruzada. |
| 3. Laboratorio OpenMP | 15 % | Dos estrategias, `default(none)`, tareas o SIMD, validación y perfil. |
| 4. Laboratorio MPI | 20 % | Halo/colectiva, ejecución Slurm, escalado fuerte/débil y análisis de comunicación. |
| 5. Laboratorio de offload portable | 5 % | OpenMP target, fallback CPU, mapeo de datos y análisis transferencia/cómputo. |
| 6. Laboratorio CUDA | 20 % | CUDA C++, GEMM tiled, una biblioteca acelerada, validación serial, Compute Sanitizer, perfil Nsight y análisis de tiempo de ejecución/extremo a extremo. |
| Hito híbrido | 5 % | Diseño MPI+OpenMP, hipótesis, pruebas y plan experimental del proyecto. |
| Proyecto final | 15 % | Código reproducible, referencia serial, pruebas, perfiles, informe y defensa. |

### Rúbrica común

- 30 % corrección y pruebas.
- 20 % diseño/descomposición.
- 20 % metodología de medición.
- 15 % interpretación y gráficas.
- 10 % reproducibilidad y calidad del repositorio.
- 5 % claridad de comunicación.

El componente de rendimiento se califica después de verificar la salida contra la referencia. Si el programa no supera esa comparación, puede recibir valoración en diseño o documentación, pero no en aceleración, eficiencia ni escalabilidad.

## 8. Bibliografía por tema

La bibliografía se selecciona por autoridad técnica, uso en docencia, vigencia de la edición y disponibilidad de erratas o material complementario. Las calificaciones comunitarias se consignan como información adicional y pueden cambiar con el tiempo. Al 2 de agosto de 2026, Goodreads reportaba 3,88/5 (64 valoraciones) para Pacheco/Malensek, 4,05/5 (148) para Kirk/Hwu/El Hajj y 4,13/5 (1.033) para Hennessy/Patterson.

### Texto transversal

- Peter Pacheco y Matthew Malensek, *An Introduction to Parallel Programming*, 2.ª ed., Morgan Kaufmann, 2021. Texto principal actual; cubre MPI, Pthreads, OpenMP y GPU, con recursos docentes. [Ficha editorial](https://www.educate.elsevier.com/book/details/9780128046050) y [valoración comunitaria](https://www.goodreads.com/book/show/34406040-an-introduction-to-parallel-programming).
- Ananth Grama et al., *Introduction to Parallel Computing*, 2.ª ed., Addison-Wesley, 2003. Texto fundacional para modelos, algoritmos y costos, que se complementa con las especificaciones vigentes.

### Fundamentos, arquitectura y rendimiento

- John L. Hennessy y David A. Patterson, *Computer Architecture: A Quantitative Approach*, 6.ª ed., Morgan Kaufmann, 2018. Referencia de arquitectura y método cuantitativo. [Valoración comunitaria](https://www.goodreads.com/book/show/70135.Computer_Architecture).
- Georg Hager y Gerhard Wellein, *Introduction to High Performance Computing for Scientists and Engineers*, CRC Press, 2010. Memoria, modelos de rendimiento y optimización.
- Raj Jain, *The Art of Computer Systems Performance Analysis*, Wiley, 1991. Referencia fundacional para diseño de experimentos; se trabajan capítulos seleccionados.
- Torsten Hoefler y Roberto Belli, “Scientific Benchmarking of Parallel Computing Systems”, SC15, 2015. Buenas prácticas modernas para *benchmarks*.

### Pthreads y C++20

- David R. Butenhof, *Programming with POSIX Threads*, Addison-Wesley, 1997. Referencia fundacional cuyas interfaces se contrastan con POSIX.1-2024.
- Anthony Williams, *C++ Concurrency in Action*, 2.ª ed., Manning, 2019. Presenta concurrencia moderna hasta C++17 y se complementa con C++20 y documentación del estándar.
- POSIX.1-2024, sección de [interfaces de hilos](https://pubs.opengroup.org/onlinepubs/9799919799/functions/V2_chap02.html).

### OpenMP

- Barbara Chapman, Gabriele Jost y Ruud van der Pas, *Using OpenMP*, MIT Press, 2007. Referencia fundacional para el modelo. La sintaxis avanzada se consulta en la especificación 5.2.
- OpenMP ARB, [OpenMP API 5.2 Specification y Examples](https://www.openmp.org/specifications/), 2021/2024. Referencia normativa.
- Ruud van der Pas, Eric Stotzer y Christian Terboven, *Using OpenMP—The Next Step*, MIT Press, 2017. Tareas, afinidad y rendimiento.

### MPI

- William Gropp, Ewing Lusk y Anthony Skjellum, *Using MPI*, 3.ª ed., MIT Press, 2014. Texto tutorial fundacional; 4,33/5 para la edición impresa con una muestra pequeña de tres valoraciones. [Ficha/valoraciones](https://www.goodreads.com/work/editions/516500-using-mpi---2nd-edition-portable-parallel-programming-with-the-message).
- MPI Forum, [MPI 5.0 Standard](https://www.mpi-forum.org/docs/), 2025. Referencia normativa actual.
- Peter Pacheco y Matthew Malensek, capítulos de MPI del texto transversal, 2021.

### Aceleradores portables

- OpenMP ARB, OpenMP 5.2, secciones `target` y ejemplos oficiales. Referencia normativa del itinerario abierto.
- Peter Pacheco y Matthew Malensek, capítulos de GPU y programación heterogénea del texto transversal, 2021.

### CUDA C++

- David B. Kirk, Wen-mei W. Hwu e Izzat El Hajj, *Programming Massively Parallel Processors*, 4.ª ed., Morgan Kaufmann, 2022. Texto principal GPU; [recursos editoriales](https://shop.elsevier.com/books/book-companion/9780323912310) y [valoración comunitaria](https://www.goodreads.com/en/book/show/59856387-programming-massively-parallel-processors).
- NVIDIA, [CUDA C++ Programming Guide 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/), 2025. Referencia del itinerario propietario.
- NVIDIA, [CUDA Toolkit 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/), para compatibilidad, arquitecturas retiradas y herramientas vigentes.
- NVIDIA, [cuBLAS](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/), [cuFFT](https://docs.nvidia.com/cuda/archive/13.0.0/cufft/), [cuSPARSE](https://docs.nvidia.com/cuda/archive/13.0.0/cusparse/), [cuSOLVER](https://docs.nvidia.com/cuda/archive/13.0.0/cusolver/) y [cuRAND](https://docs.nvidia.com/cuda/archive/13.0.0/curand/), referencias oficiales de CUDA 13.0.
- NVIDIA, [Thrust](https://nvidia.github.io/cccl/thrust/) y [CUB](https://nvidia.github.io/cccl/cub/), componentes de CUDA Core Compute Libraries. El curso fija la versión 3.0.1 incluida en CUDA 13.0.

### Híbrido y proyecto

- Georg Hager y Gerhard Wellein, texto citado, capítulos de afinidad, modelos y optimización.
- Rolf Rabenseifner, Georg Hager y Gabriele Jost, “Hybrid MPI/OpenMP Parallel Programming on Clusters of Multi-Core SMP Nodes”, 2009. Referencia fundacional para diseño híbrido; sus comandos y supuestos de hardware se actualizan para el entorno del curso.
- Timothy G. Mattson, Beverly A. Sanders y Berna L. Massingill, *Patterns for Parallel Programming*, Addison-Wesley, 2004. Patrones de descomposición y coordinación.

## 9. Organización de fuentes, ejercicios y soluciones

- Los ejemplos presentados en los notebooks se almacenan como fuentes independientes en `curso/ejemplos/<tema>/`; el notebook los construye sin duplicar manualmente el código en las celdas.
- Los ejercicios se ubican en `curso/ejercicios/<tema>/<id>/`, junto con el enunciado, la interfaz, datos pequeños y pruebas públicas.
- Las soluciones de referencia se ubican en `curso/ejercicios/soluciones/<tema>/<id>/` y permanecen por fuera de la rama estudiantil durante el semestre.
- Las mediciones se realizan sobre programas que hayan superado las pruebas de corrección.
- Los resultados de rendimiento se generan en `build/results/`. Solo se versionan conjuntos pequeños que sean necesarios como referencia.
- Las fuentes de terceros se registran en `THIRD_PARTY_NOTICES.md` con origen, versión y licencia.

## 10. Condiciones para dictar un tema

Antes de incorporar un tema al calendario se comprueba que cuente con:

- 1 a 3 notebooks ejecutados de principio a fin.
- Ejemplos que construyen con el entorno fijado.
- Prueba serial/paralela y manejo de errores.
- Gráficas regenerables y datos trazables.
- Ejercicios, solución y rúbrica.
- Bibliografía y enlaces normativos.
- Tiempo de clase ensayado dentro de dos horas.

Esta planeación establece el trabajo previsto para la nueva versión. Las carpetas creadas para organizar el material indican su ubicación, pero el tema se considera disponible cuando cumple las condiciones anteriores.

## 11. Material de profundización

ROCm/HIP, SYCL, Kokkos y RAJA se ofrecen por fuera de las 19 semanas, las 38 sesiones y las evaluaciones regulares. La planeación, la guía conceptual, los entornos de compilación y los requisitos de clúster se encuentran en [`topicos_avanzados/README.md`](../topicos_avanzados/README.md).
