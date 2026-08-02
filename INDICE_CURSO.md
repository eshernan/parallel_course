# Índice de navegación del curso

Este documento es el punto de entrada al material de **Programación Paralela 2026**. Desde aquí se consulta la secuencia académica, se abre el notebook correspondiente y se localizan sus ejemplos, ejercicios y soluciones. Para regresar desde un notebook se utiliza el enlace **Volver al índice del curso**, ubicado en su primera y última sección.

## Antes de comenzar

1. Consulte en el [README principal](README.md) el alcance, las tecnologías y la distribución de las 19 semanas.
2. Desde la raíz del repositorio ejecute `python3 validation/preflight.py`. No continúe si la comprobación informa un error de plataforma, cadena de herramientas o construcción.
3. Siga los temas en el orden de este índice. Cada notebook indica los prerrequisitos adicionales de su práctica.
4. Los ejemplos acompañan la explicación; los ejercicios son el trabajo del estudiante y las soluciones se administran por separado durante la evaluación.

## Notebooks del curso

La edición 2026 contiene los 23 notebooks definidos por la planeación. Cada enlace abre el archivo correspondiente y cada notebook ofrece navegación de regreso a este índice y a la guía de su tema.

| Tema | Semanas | Notebooks | Guía del tema |
|---:|---:|---|---|
| 00. Entorno y lenguajes | 1 | [`Entorno reproducible`](curso/notebooks/00_entorno/00_entorno_reproducible.ipynb) · [`Estándares y compiladores`](curso/notebooks/00_entorno/01_estandares_compiladores.ipynb) | [Tema 00](curso/notebooks/00_entorno/README.md) |
| 01. Fundamentos | 2–3 | [`Modelos`](curso/notebooks/01_fundamentos/01_modelos.ipynb) · [`Escalabilidad`](curso/notebooks/01_fundamentos/02_escalabilidad.ipynb) · [`Memoria y Roofline`](curso/notebooks/01_fundamentos/03_memoria_roofline.ipynb) | [Tema 01](curso/notebooks/01_fundamentos/README.md) |
| 02. Memoria compartida | 4–5 | [`Pthreads`](curso/notebooks/02_memoria_compartida/01_pthreads.ipynb) · [`Sincronización`](curso/notebooks/02_memoria_compartida/02_sincronizacion.ipynb) · [`C++20 y atomics`](curso/notebooks/02_memoria_compartida/03_cpp20_atomics.ipynb) | [Tema 02](curso/notebooks/02_memoria_compartida/README.md) |
| 03. OpenMP | 6–8 | [`Modelo de datos`](curso/notebooks/03_openmp/01_modelo_datos.ipynb) · [`Bucles y reducciones`](curso/notebooks/03_openmp/02_bucles_reducciones.ipynb) · [`Tareas`](curso/notebooks/03_openmp/03_tareas_rendimiento.ipynb) | [Tema 03](curso/notebooks/03_openmp/README.md) |
| 04. MPI | 8–11 | [`Punto a punto`](curso/notebooks/04_mpi/01_punto_a_punto.ipynb) · [`Colectivas y topologías`](curso/notebooks/04_mpi/02_colectivas_topologias.ipynb) · [`Escalabilidad y Slurm`](curso/notebooks/04_mpi/03_escalabilidad_slurm.ipynb) | [Tema 04](curso/notebooks/04_mpi/README.md) |
| 05. OpenMP target | 11–12 | [`Modelo de offload`](curso/notebooks/05_openmp_target/01_modelo_offload.ipynb) · [`OpenMP target`](curso/notebooks/05_openmp_target/02_openmp_target.ipynb) | [Tema 05](curso/notebooks/05_openmp_target/README.md) |
| 06. CUDA C++ | 13–15 | [`Modelo CUDA`](curso/notebooks/06_cuda/01_modelo_cuda.ipynb) · [`Memoria y tiling`](curso/notebooks/06_cuda/02_memoria_tiling.ipynb) · [`Bibliotecas y perfiles`](curso/notebooks/06_cuda/03_bibliotecas_perfiles.ipynb) | [Tema 06](curso/notebooks/06_cuda/README.md) |
| 07. Programación híbrida | 16–17 | [`MPI + OpenMP`](curso/notebooks/07_hibrido/01_mpi_openmp.ipynb) · [`MPI + GPU`](curso/notebooks/07_hibrido/02_mpi_gpu.ipynb) · [`Perfilado reproducible`](curso/notebooks/07_hibrido/03_perfilado_reproducible.ipynb) | [Tema 07](curso/notebooks/07_hibrido/README.md) |
| 08. Proyecto final | 18–19 | [`Guía del proyecto`](curso/notebooks/08_proyecto/01_guia_proyecto.ipynb) | [Tema 08](curso/notebooks/08_proyecto/README.md) |

La distribución sesión por sesión y las evaluaciones se consultan en la [planeación semestral](docs/PLANEACION_CURSO.md#6-calendario-de-38-sesiones). El [protocolo de reproducibilidad](docs/REPRODUCIBILIDAD_EJERCICIOS.md) explica cómo interpretar los informes de plataforma y compilación.

## Organización del trabajo práctico

| Recurso | Ubicación | Uso |
|---|---|---|
| Notebooks | [`curso/notebooks/`](curso/notebooks/README.md) | Explicación conceptual, experimentos y gráficas. |
| Ejemplos | [`curso/ejemplos/`](curso/ejemplos/README.md) | Especificación de las fuentes compilables que acompañarán las prácticas; su disponibilidad se comprueba antes de usarlas. |
| Ejercicios | [`curso/ejercicios/`](curso/ejercicios/README.md) | Enunciados, código inicial, manifiestos y pruebas públicas. |
| Soluciones | [`curso/ejercicios/soluciones/`](curso/ejercicios/soluciones/README.md) | Resolución docente y pruebas de referencia. |
| Configuración | [`config/`](config/README.md) | Compiladores, MPI, CUDA, librerías y versiones fijadas. |
| Validación | [`validation/`](docs/REPRODUCIBILIDAD_EJERCICIOS.md) | Comprobación previa, inventario del equipo y construcción reproducible. |

## Material de profundización

ROCm/HIP, SYCL, Kokkos y RAJA no forman parte de las 38 sesiones del curso base. Su recorrido comienza en el [índice de tópicos avanzados](topicos_avanzados/README.md), donde se indican conceptos, toolchains, hardware, configuración de clúster, ejemplos y ejercicios previstos.

## Regla de mantenimiento

La incorporación, cambio de nombre o retiro de un notebook exige actualizar este índice y el README principal en el mismo cambio. Todo `.ipynb` ubicado bajo `curso/notebooks/` debe contener `INDICE_CURSO.md` en una celda Markdown de apertura y otra de cierre. `validation/validate_navigation.py`, ejecutado por el preflight y por GitHub Actions, verifica esas condiciones y comprueba que los enlaces locales de este documento existan.
