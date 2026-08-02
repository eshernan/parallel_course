# Índice de navegación del curso

Este documento es el punto de entrada al material de **Programación Paralela 2026**. Desde aquí se consulta la secuencia académica, se abre el notebook correspondiente y se localizan sus ejemplos, ejercicios y soluciones. Para regresar desde un notebook se utiliza el enlace **Volver al índice del curso**, ubicado en su primera y última sección.

## Antes de comenzar

1. Consulte en el [README principal](README.md) el alcance, las tecnologías y la distribución de las 19 semanas.
2. Desde la raíz del repositorio ejecute `python3 validation/preflight.py`. No continúe si la comprobación informa un error de plataforma, cadena de herramientas o construcción.
3. Siga los temas en el orden de este índice. Cada notebook indica los prerrequisitos adicionales de su práctica.
4. Los ejemplos acompañan la explicación; los ejercicios son el trabajo del estudiante y las soluciones se administran por separado durante la evaluación.

## Estado de los notebooks

La planeación define entre uno y tres notebooks por tema. En esta revisión todavía no se han incorporado archivos `.ipynb`; por ello no se publican enlaces que conduzcan a archivos inexistentes. La columna **Navegación actual** lleva a la guía disponible o a la planeación detallada. Cuando se incorpore un notebook, su nombre se convertirá aquí en un enlace directo y deberá incluir el enlace de retorno a este índice.

| Tema | Semanas | Notebooks definidos | Navegación actual |
|---:|---:|---|---|
| 00. Entorno y lenguajes | 1 | `00_entorno_reproducible.ipynb`, `01_estandares_compiladores.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) · [Estándares C/C++](docs/ESTANDARES_C_CPP.md) |
| 01. Fundamentos | 2–3 | `01_modelos.ipynb`, `02_escalabilidad.ipynb`, `03_memoria_roofline.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 02. Memoria compartida | 4–5 | `01_pthreads.ipynb`, `02_sincronizacion.ipynb`, `03_cpp20_atomics.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 03. OpenMP | 6–8 | `01_modelo_datos.ipynb`, `02_bucles_reducciones.ipynb`, `03_tareas_rendimiento.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 04. MPI | 8–11 | `01_punto_a_punto.ipynb`, `02_colectivas_topologias.ipynb`, `03_escalabilidad_slurm.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 05. OpenMP target | 11–12 | `01_modelo_offload.ipynb`, `02_openmp_target.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 06. CUDA C++ | 13–15 | `01_modelo_cuda.ipynb`, `02_memoria_tiling.ipynb`, `03_bibliotecas_perfiles.ipynb` | [Guía detallada del tema CUDA](curso/notebooks/06_cuda/README.md) · [Ejemplos previstos](curso/ejemplos/06_cuda/README.md) · [Ejercicios previstos](curso/ejercicios/06_cuda/README.md) |
| 07. Programación híbrida | 16–17 | `01_mpi_openmp.ipynb`, `02_mpi_gpu.ipynb`, `03_perfilado_reproducible.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |
| 08. Proyecto final | 18–19 | `01_guia_proyecto.ipynb` | [Descripción y fuentes previstas](docs/PLANEACION_CURSO.md#5-organización-de-notebooks-y-fuentes) |

La distribución sesión por sesión y las evaluaciones se consultan en la [planeación semestral](docs/PLANEACION_CURSO.md#6-calendario-de-38-sesiones). El [protocolo de reproducibilidad](docs/REPRODUCIBILIDAD_EJERCICIOS.md) explica cómo interpretar los informes de plataforma y compilación.

## Organización del trabajo práctico

| Recurso | Ubicación | Uso |
|---|---|---|
| Notebooks | [`curso/notebooks/`](curso/notebooks/README.md) | Explicación conceptual, experimentos y gráficas. |
| Ejemplos | [`curso/ejemplos/`](curso/ejemplos/README.md) | Fuentes completas invocadas por los notebooks. |
| Ejercicios | [`curso/ejercicios/`](curso/ejercicios/README.md) | Enunciados, código inicial, manifiestos y pruebas públicas. |
| Soluciones | [`curso/ejercicios/soluciones/`](curso/ejercicios/soluciones/README.md) | Resolución docente y pruebas de referencia. |
| Configuración | [`config/`](config/README.md) | Compiladores, MPI, CUDA, librerías y versiones fijadas. |
| Validación | [`validation/`](docs/REPRODUCIBILIDAD_EJERCICIOS.md) | Comprobación previa, inventario del equipo y construcción reproducible. |

## Material de profundización

ROCm/HIP, SYCL, Kokkos y RAJA no forman parte de las 38 sesiones del curso base. Su recorrido comienza en el [índice de tópicos avanzados](topicos_avanzados/README.md), donde se indican conceptos, toolchains, hardware, configuración de clúster, ejemplos y ejercicios previstos.

## Regla de mantenimiento

La incorporación, cambio de nombre o retiro de un notebook exige actualizar este índice y el README principal en el mismo cambio. Todo `.ipynb` ubicado bajo `curso/notebooks/` debe contener `INDICE_CURSO.md` en una celda Markdown de apertura y otra de cierre. `validation/validate_navigation.py`, ejecutado por el preflight y por GitHub Actions, verifica esas condiciones y comprueba que los enlaces locales de este documento existan.
