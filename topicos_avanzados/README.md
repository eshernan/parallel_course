# Tópicos avanzados de portabilidad de rendimiento

Esta sección amplía el repositorio con material de estudio independiente sobre ecosistemas heterogéneos y portabilidad de rendimiento. **No forma parte de las 19 semanas, las 38 sesiones ni las evaluaciones del curso base.** Puede utilizarse en seminarios, escuelas de verano, proyectos de investigación o como continuación del curso.

Los prerrequisitos generales son haber completado los módulos de OpenMP, MPI, OpenMP target y CUDA, además de dominar C++20, plantillas, lambdas, jerarquías de memoria y medición reproducible.

## Objetivos

Al completar las cápsulas, el estudiante podrá:

1. Programar y perfilar una GPU AMD mediante ROCm y HIP.
2. Explicar la diferencia entre una API directa de GPU, un estándar como SYCL y una capa de portabilidad como Kokkos o RAJA.
3. Implementar el mismo kernel sobre varios backends sin confundir portabilidad funcional con portabilidad de rendimiento.
4. Elegir espacios de ejecución, memoria, políticas y mecanismos de sincronización de manera explícita.
5. Comparar corrección, mantenibilidad y rendimiento contra implementaciones nativas OpenMP, CUDA o HIP.

## Versiones de referencia

Las versiones se congelan para reproducir los laboratorios avanzados y se revisan antes de cada edición. La fuente global es [`config/advanced-topics.cmake`](../config/advanced-topics.cmake).

| Componente | Versión de referencia | Función |
|---|---:|---|
| ROCm | 7.2.3 | Stack abierto de cómputo para GPU AMD sobre Linux. |
| HIP | 7.2.3 | API y compilación C++ para los ejemplos nativos AMD. |
| SYCL | SYCL 2020, revisión 11 | Estándar normativo de programación heterogénea en C++ de Khronos. |
| AdaptiveCpp | 25.10.0 | Implementación abierta de referencia de SYCL, con backends para CPU y GPU. |
| Kokkos | 5.1.1 | Modelo de ejecución y memoria para portabilidad de rendimiento. |
| RAJA | 2025.12.2 | Abstracciones de políticas para bucles y kernels portables. |
| C++ | C++20 | Lenguaje común de las cuatro cápsulas. |

ROCm/HIP exige una GPU y un sistema operativo incluidos en la matriz de compatibilidad de AMD. SYCL, Kokkos y RAJA no garantizan que un único binario o una única configuración funcione sobre todos los fabricantes. Cada resultado debe registrar compilador, backend, arquitectura, controlador y hardware.

## Método de explicación

La conceptualización es el centro de cada cápsula. Antes de mostrar sintaxis, el tema debe responder:

1. ¿Qué problema de portabilidad o programación resuelve?
2. ¿Qué abstracciones introduce y qué oculta?
3. ¿Cómo se expresa el trabajo paralelo?
4. ¿Quién administra memoria, ejecución, sincronización y errores?
5. ¿Qué parte depende del compilador, runtime, backend, dispositivo y clúster?
6. ¿Qué garantías ofrece y cuáles no ofrece?

Cada tema seguirá la [plantilla avanzada](PLANTILLA_TEMA.md): mapa conceptual, manera de trabajo, ejemplo mínimo, experimento, interpretación y ejercicio. Se usarán entre uno y tres notebooks; no se fragmentará una explicación únicamente para aumentar el número de archivos.

Las figuras conceptuales mantendrán el estilo de boceto técnico a lápiz del curso. Las gráficas cuantitativas serán poco densas: una pregunta por figura, pocas series claramente identificadas, unidades visibles, línea base y una conclusión breve. No se aceptan tableros saturados ni capturas de perfiladores sin una pregunta de análisis.

## Cápsulas propuestas

### A1. AMD ROCm y HIP

**Conceptos clave:** stack ROCm; arquitectura AMD; modelo grid/bloque/hilo; wavefront; memoria host/device; transferencias; eventos; streams; errores asíncronos; rocPRIM/rocBLAS; perfilado y relación HIP–CUDA.

**Manera de trabajo:** inventariar hardware y software → compilar con `amdclang++`/`hipcc` → validar kernel contra CPU → comprobar errores y sincronización → perfilar transferencias y kernels → ajustar acceso a memoria y ocupación.

**Notebooks previstos:** `01_rocm_entorno_modelo.ipynb`, `02_hip_memoria_kernels.ipynb` y `03_hip_streams_perfilado.ipynb`.

**Ejemplos ilustrativos:** inventario del dispositivo, suma vectorial, reducción y multiplicación de matrices con referencia CPU. `hipify` se trata como ayuda de migración, nunca como prueba de portabilidad o corrección.

**Entorno mínimo:** Linux soportado por ROCm 7.2.3, GPU AMD incluida en su matriz de compatibilidad, controlador correspondiente y acceso a los perfiladores. En clúster, el ejercicio debe declarar módulo ROCm, partición GPU, recurso GRES equivalente y arquitectura `gfx*` efectiva.

**Evaluación opcional:** portar y perfilar un kernel CUDA ya validado, documentando diferencias semánticas, cambios manuales, resultados numéricos y cuellos de botella.

### A2. SYCL

**Conceptos clave:** estándar frente a implementación; plataformas y dispositivos; selectores; colas; grupos de comandos; `range`/`nd_range`; USM; buffers/accessors; eventos; dependencias; backends y perfilado.

**Manera de trabajo:** seleccionar implementación/backend → enumerar dispositivos → construir la cola → expresar dependencias y kernel → validar → capturar eventos → comparar el mismo código en los backends realmente disponibles.

**Notebooks previstos:** `01_sycl_modelo_colas.ipynb`, `02_sycl_memoria_ndrange.ipynb` y `03_sycl_portabilidad_medicion.ipynb`.

**Ejemplos ilustrativos:** consulta de dispositivos, suma vectorial, reducción y stencil con selección explícita del dispositivo y fallback controlado.

**Entorno mínimo:** compilador AdaptiveCpp 25.10.0 y un backend declarado. Un ejercicio CPU no presupone soporte GPU. En clúster se registran módulos del compilador/runtime, backend activo, dispositivo solicitado, partición y variables de selección visibles al proceso.

**Evaluación opcional:** ejecutar un kernel sobre dos backends disponibles, validar resultados y explicar qué partes del código, construcción y ajuste siguen siendo específicas de plataforma.

### A3. Kokkos

**Conceptos clave:** espacios de ejecución y memoria; `View`; layouts; mirrors; deep copies; `RangePolicy`, `MDRangePolicy` y `TeamPolicy`; `parallel_for`, `parallel_reduce`, `parallel_scan`; inicialización y Kokkos Tools.

**Manera de trabajo:** seleccionar un backend al configurar Kokkos → definir espacios y layout → expresar el patrón con una política → validar movimientos de datos → medir → ajustar políticas, equipos o layout para el hardware.

**Notebooks previstos:** `01_kokkos_views_policies.ipynb`, `02_kokkos_jerarquia_memoria.ipynb` y `03_kokkos_backends_rendimiento.ipynb`.

**Ejemplos ilustrativos:** reducción, stencil 2D y multiplicación de matrices con la misma interfaz y configuraciones de backend separadas.

**Entorno mínimo:** Kokkos 5.1.1 compilado para un backend y arquitectura declarados. Cada ejercicio indicará el compilador que construyó Kokkos, `Kokkos_DIR`, backend, arquitectura y combinación host/device. No se reutiliza una instalación de Kokkos construida para otro backend como si fuera intercambiable.

**Evaluación opcional:** desarrollar un kernel con dos políticas o layouts y comparar portabilidad funcional, trabajo de ajuste y rendimiento contra una línea base nativa.

### A4. RAJA

**Conceptos clave:** segmentos; políticas de ejecución; `RAJA::forall`; reducciones; `RAJA::kernel`; `RAJA::launch`; recursos; captura de lambdas y relación con Umpire/CHAI.

**Manera de trabajo:** aislar el cuerpo del bucle → definir segmento y política → resolver memoria fuera de la política → validar backend secuencial → activar OpenMP/GPU → medir y ajustar la política explícita.

**Notebooks previstos:** `01_raja_segmentos_politicas.ipynb`, `02_raja_kernels_recursos.ipynb` y `03_raja_backends_comparacion.ipynb`.

**Ejemplos ilustrativos:** suma vectorial, reducción y stencil con políticas secuencial, OpenMP y, según disponibilidad, CUDA o HIP.

**Entorno mínimo:** RAJA 2025.12.2 construido con el backend declarado, `RAJA_DIR`, compilador C++20 y dependencias de la suite. Para HIP se declara ROCm/rocPRIM; para CUDA, Toolkit/arquitectura. El script del clúster debe coincidir con ese backend.

**Evaluación opcional:** implementar un kernel con políticas intercambiables y justificar el costo de abstracción, las decisiones de memoria y las diferencias de rendimiento.

En RAJA 2025.12.2, los backends secuencial, OpenMP, CUDA y HIP son los más maduros. SYCL y OpenMP target se consideran experimentales o incompletos para varias capacidades; no deben presentarse como equivalentes de producción sin validación específica.

## Comparación transversal

| Aspecto | HIP | SYCL | Kokkos | RAJA |
|---|---|---|---|---|
| Tipo | API/lenguaje GPU | Estándar C++ heterogéneo | Ecosistema de abstracciones C++ | Biblioteca de políticas C++ |
| Abstracción principal | Kernel, grid, bloque e hilo | Cola, comando, `nd_range` y evento | Espacio, `View` y política | Segmento, política y recurso |
| Memoria | Explícita o administrada | USM o buffers/accessors | `View`, espacios y mirrors | Externa; recursos auxiliares |
| Backend docente inicial | AMD/ROCm | AdaptiveCpp CPU/GPU | Serial y OpenMP | Secuencial y OpenMP |
| Extensión GPU | AMD nativa | Según implementación | CUDA, HIP o SYCL | CUDA o HIP maduros |
| Riesgo pedagógico | Suponer identidad total con CUDA | Confundir estándar e implementación | Creer que cambiar backend basta para rendir | Omitir la gestión de memoria |

El estudio comparativo reutiliza los mismos kernels y conjuntos de datos. No se comparan tiempos entre máquinas distintas como si fueran evidencia de superioridad de un framework.

## Organización del material

```text
topicos_avanzados/
├── PLANTILLA_TEMA.md               # Contrato conceptual y experimental
├── ENTORNOS_CLUSTER.md             # Ficha de compilación y ejecución
├── notebooks/                      # 1–3 notebooks por cápsula
├── ejemplos/                       # Referencia serial y variantes portables
└── ejercicios/
    ├── <topico>/<ejercicio>/       # Enunciado, esqueleto y pruebas públicas
    └── soluciones/
        └── <topico>/<ejercicio>/   # Solución y pruebas docentes
```

Los nombres de tópico son `01_rocm_hip`, `02_sycl`, `03_kokkos` y `04_raja`. Se aplican las mismas reglas de corrección, trazabilidad, gráficas y estilo visual descritas para [`curso/`](../curso/README.md). Las soluciones deben mantenerse fuera de la distribución estudiantil cuando una cápsula se evalúe.

## Perfiles de entorno

No se construye todo el material avanzado en un único preset. Se mantendrán perfiles independientes:

- `advanced-hip`: ROCm/HIP sobre un nodo AMD compatible.
- `advanced-sycl`: AdaptiveCpp y un backend declarado.
- `advanced-kokkos-cpu`, `advanced-kokkos-hip` o `advanced-kokkos-cuda`.
- `advanced-raja-cpu`, `advanced-raja-hip` o `advanced-raja-cuda`.

La configuración acepta las raíces `COURSE_ROCM_ROOT`, `COURSE_ADAPTIVECPP_ROOT`, `COURSE_KOKKOS_ROOT` y `COURSE_RAJA_ROOT`. Los presets se añadirán junto con el primer ejemplo compilable de cada perfil; una carpeta vacía no se considera soporte de un backend.

## Bibliografía y especificaciones

### ROCm/HIP

- AMD, [ROCm 7.2.3 release notes](https://rocm.docs.amd.com/en/docs-7.2.3/about/release-notes.html).
- AMD, [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).
- AMD, [ROCm Linux compatibility and installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

### SYCL

- Khronos Group, [SYCL 2020 Specification, revision 11](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html).
- James Brodman et al., *Data Parallel C++*, 2.ª ed., Apress, 2023. Debe leerse distinguiendo las extensiones oneAPI del núcleo normativo de SYCL.
- AdaptiveCpp, [documentación y código fuente](https://github.com/AdaptiveCpp/AdaptiveCpp).

### Kokkos

- Kokkos Team, [Kokkos Core documentation](https://kokkos.org/kokkos-core-wiki/).
- H. Carter Edwards, Christian R. Trott y Daniel Sunderland, “Kokkos: Enabling manycore performance portability through polymorphic memory access patterns”, *Journal of Parallel and Distributed Computing*, 2014.
- David Hollman et al., “Kokkos 3: Programming model extensions for the exascale era”, *IEEE TPDS*, 2022.

### RAJA

- LLNL, [RAJA 2025.12.2 User Guide](https://raja.readthedocs.io/en/v2025.12.2/).
- Richard D. Hornung y Jeffrey A. Keasler, *The RAJA Portability Layer: Overview and Status*, LLNL-TR-661403, 2014.
- LLNL, [RAJA Performance Suite](https://github.com/LLNL/RAJAPerf), para comparaciones de kernels y backends.

## Criterio de finalización

Una cápsula solo estará lista para publicación cuando tenga notebooks ejecutados, ejemplos compilables en cada perfil declarado, referencia serial, pruebas de corrección, medición reproducible, ejercicios, soluciones, bibliografía y una matriz de hardware/software probada. Este README define la ruta avanzada; no certifica todavía que los cuatro toolchains estén instalados en el equipo local.
