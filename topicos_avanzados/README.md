# Temas avanzados de portabilidad de rendimiento

Esta sección contiene material de profundización sobre programación heterogénea y portabilidad de rendimiento. Se propone para seminarios, cursos intersemestrales y proyectos de investigación posteriores a la asignatura. Por esta razón, sus actividades no se cuentan dentro de las 19 semanas, las 38 sesiones ni las evaluaciones del programa regular.

Para abordar estos temas se requiere haber cursado OpenMP, MPI, OpenMP target y CUDA. También se espera manejo de C++20, plantillas, expresiones lambda, jerarquías de memoria y metodología de medición.

## Objetivos

El trabajo propuesto busca que el estudiante esté en capacidad de:

1. Programar y perfilar una GPU AMD mediante ROCm y HIP.
2. Explicar la diferencia entre una API directa de GPU, un estándar como SYCL y una capa de portabilidad como Kokkos o RAJA.
3. Implementar el mismo kernel sobre varios backends y distinguir la portabilidad funcional de la portabilidad del rendimiento.
4. Elegir espacios de ejecución, memoria, políticas y mecanismos de sincronización de manera explícita.
5. Comparar corrección, mantenibilidad y rendimiento contra implementaciones nativas OpenMP, CUDA o HIP.

## Versiones de referencia

Para cada edición se fija una combinación de versiones que permita reproducir las prácticas. La matriz se revisa antes de iniciar el trabajo académico y se registra en [`config/advanced-topics.cmake`](../config/advanced-topics.cmake).

| Componente | Versión de referencia | Función |
|---|---:|---|
| ROCm | 7.2.3 | Plataforma abierta de cómputo para GPU AMD sobre Linux. |
| HIP | 7.2.3 | API y compilación C++ para los ejemplos nativos AMD. |
| SYCL | SYCL 2020, revisión 11 | Estándar normativo de programación heterogénea en C++ de Khronos. |
| AdaptiveCpp | 25.10.0 | Implementación abierta de referencia de SYCL, con backends para CPU y GPU. |
| Kokkos | 5.1.1 | Modelo de ejecución y memoria para portabilidad de rendimiento. |
| RAJA | 2025.12.2 | Abstracciones de políticas para bucles y kernels portables. |
| C++ | C++20 | Lenguaje común de las cuatro unidades. |

Para trabajar con ROCm/HIP se necesita una GPU y un sistema operativo incluidos en la matriz de compatibilidad de AMD. En SYCL, Kokkos y RAJA, la posibilidad de compilar o ejecutar en varias plataformas no implica que una sola configuración sea adecuada para todos los fabricantes. Los informes registran compilador, backend, arquitectura, controlador y hardware.

## Enfoque docente

Cada unidad comienza con el modelo de programación y con las decisiones que el framework deja en manos del programador. La presentación de la sintaxis viene después de discutir las siguientes preguntas:

1. ¿Qué problema de portabilidad o programación resuelve?
2. ¿Qué abstracciones introduce y qué oculta?
3. ¿Cómo se expresa el trabajo paralelo?
4. ¿Quién administra memoria, ejecución, sincronización y errores?
5. ¿Qué parte depende del compilador, runtime, backend, dispositivo y clúster?
6. ¿Qué garantías ofrece y cuáles no ofrece?

La [plantilla para temas avanzados](PLANTILLA_TEMA.md) organiza el mapa conceptual, la práctica de laboratorio, el ejemplo inicial, el experimento y los ejercicios. Cada unidad utiliza entre uno y tres notebooks, de acuerdo con la extensión efectiva del contenido.

Las figuras conceptuales conservan el estilo de boceto técnico a lápiz definido para la asignatura. En las gráficas cuantitativas se plantea una pregunta por figura, con pocas series, unidades visibles y una línea base cuando corresponda. Las capturas de perfiladores se incluyen cuando permiten discutir un evento o un cuello de botella específico.

## Unidades de profundización

### A1. AMD ROCm y HIP

**Conceptos clave:** ecosistema ROCm; arquitectura AMD; modelo grid/bloque/hilo; wavefront; memoria host/device; transferencias; eventos; streams; errores asíncronos; rocPRIM/rocBLAS; perfilado y relación HIP–CUDA.

**Trabajo de laboratorio:** inventario de hardware y software → compilación con `amdclang++`/`hipcc` → validación del kernel frente a CPU → comprobación de errores y sincronización → perfil de transferencias y kernels → ajuste del acceso a memoria y la ocupación.

**Notebooks previstos:** `01_rocm_entorno_modelo.ipynb`, `02_hip_memoria_kernels.ipynb` y `03_hip_streams_perfilado.ipynb`.

**Ejemplos ilustrativos:** inventario del dispositivo, suma vectorial, reducción y multiplicación de matrices con referencia en CPU. `hipify` se utiliza como apoyo para una migración inicial; la portabilidad y la corrección se establecen mediante compilación, pruebas y revisión del código resultante.

**Entorno de trabajo:** Linux soportado por ROCm 7.2.3, GPU AMD incluida en su matriz de compatibilidad, controlador correspondiente y acceso a los perfiladores. En el clúster se registran el módulo ROCm, la partición GPU, el recurso GRES equivalente y la arquitectura `gfx*` efectiva.

**Evaluación opcional:** portar y perfilar un kernel CUDA ya validado, documentando diferencias semánticas, cambios manuales, resultados numéricos y cuellos de botella.

### A2. SYCL

**Conceptos clave:** estándar frente a implementación; plataformas y dispositivos; selectores; colas; grupos de comandos; `range`/`nd_range`; USM; buffers/accessors; eventos; dependencias; backends y perfilado.

**Trabajo de laboratorio:** selección de implementación y backend → inventario de dispositivos → construcción de la cola → expresión de dependencias y kernel → validación → captura de eventos → comparación en los backends disponibles.

**Notebooks previstos:** `01_sycl_modelo_colas.ipynb`, `02_sycl_memoria_ndrange.ipynb` y `03_sycl_portabilidad_medicion.ipynb`.

**Ejemplos ilustrativos:** consulta de dispositivos, suma vectorial, reducción y stencil con selección explícita del dispositivo y fallback controlado.

**Entorno de trabajo:** AdaptiveCpp 25.10.0 y un backend declarado. Una ejecución en CPU no se toma como evidencia de soporte GPU. En el clúster se registran los módulos del compilador y del runtime, el backend activo, el dispositivo solicitado, la partición y las variables de selección visibles para el proceso.

**Evaluación opcional:** ejecutar un kernel sobre dos backends disponibles, validar resultados y explicar qué partes del código, construcción y ajuste siguen siendo específicas de plataforma.

### A3. Kokkos

**Conceptos clave:** espacios de ejecución y memoria; `View`; layouts; mirrors; deep copies; `RangePolicy`, `MDRangePolicy` y `TeamPolicy`; `parallel_for`, `parallel_reduce`, `parallel_scan`; inicialización y Kokkos Tools.

**Trabajo de laboratorio:** selección del backend durante la configuración de Kokkos → definición de espacios y disposición de datos → expresión del patrón mediante una política → validación de movimientos de datos → medición → ajuste de políticas, equipos o layout para el hardware disponible.

**Notebooks previstos:** `01_kokkos_views_policies.ipynb`, `02_kokkos_jerarquia_memoria.ipynb` y `03_kokkos_backends_rendimiento.ipynb`.

**Ejemplos ilustrativos:** reducción, stencil 2D y multiplicación de matrices con la misma interfaz y configuraciones de backend separadas.

**Entorno de trabajo:** Kokkos 5.1.1 compilado para un backend y una arquitectura declarados. El ejercicio registra el compilador utilizado para construir Kokkos, `Kokkos_DIR`, el backend, la arquitectura y la combinación host/device. Las instalaciones destinadas a backends distintos se manejan como perfiles separados.

**Evaluación opcional:** desarrollar un kernel con dos políticas o layouts y comparar portabilidad funcional, trabajo de ajuste y rendimiento contra una línea base nativa.

### A4. RAJA

**Conceptos clave:** segmentos; políticas de ejecución; `RAJA::forall`; reducciones; `RAJA::kernel`; `RAJA::launch`; recursos; captura de lambdas y relación con Umpire/CHAI.

**Trabajo de laboratorio:** aislamiento del cuerpo del bucle → definición del segmento y la política → gestión de memoria por fuera de la política → validación del backend secuencial → activación de OpenMP o GPU → medición y ajuste de la política.

**Notebooks previstos:** `01_raja_segmentos_politicas.ipynb`, `02_raja_kernels_recursos.ipynb` y `03_raja_backends_comparacion.ipynb`.

**Ejemplos ilustrativos:** suma vectorial, reducción y stencil con políticas secuencial, OpenMP y, según disponibilidad, CUDA o HIP.

**Entorno de trabajo:** RAJA 2025.12.2 construido con el backend declarado, `RAJA_DIR`, compilador C++20 y dependencias de la suite. El perfil HIP registra ROCm y rocPRIM; el perfil CUDA registra el Toolkit y la arquitectura. El script de ejecución corresponde al backend utilizado en la construcción.

**Evaluación opcional:** implementar un kernel con políticas intercambiables y justificar el costo de abstracción, las decisiones de memoria y las diferencias de rendimiento.

En RAJA 2025.12.2, el soporte más consolidado corresponde a los backends secuencial, OpenMP, CUDA y HIP. SYCL y OpenMP target todavía tienen capacidades experimentales o incompletas; cualquier uso en un proyecto requiere una validación particular de las funciones empleadas.

## Comparación transversal

| Aspecto | HIP | SYCL | Kokkos | RAJA |
|---|---|---|---|---|
| Tipo | API/lenguaje GPU | Estándar C++ heterogéneo | Ecosistema de abstracciones C++ | Biblioteca de políticas C++ |
| Abstracción principal | Kernel, grid, bloque e hilo | Cola, comando, `nd_range` y evento | Espacio, `View` y política | Segmento, política y recurso |
| Memoria | Explícita o administrada | USM o buffers/accessors | `View`, espacios y mirrors | Externa; recursos auxiliares |
| Backend docente inicial | AMD/ROCm | AdaptiveCpp CPU/GPU | Serial y OpenMP | Secuencial y OpenMP |
| Extensión GPU | AMD nativa | Según implementación | CUDA, HIP o SYCL | CUDA o HIP maduros |
| Riesgo pedagógico | Suponer identidad total con CUDA | Confundir estándar e implementación | Creer que cambiar backend basta para rendir | Omitir la gestión de memoria |

El estudio comparativo utiliza los mismos kernels y conjuntos de datos. Cuando intervienen máquinas distintas, los resultados se reportan como caracterización de cada plataforma y no como una clasificación directa de los frameworks.

## Organización del material

```text
topicos_avanzados/
├── PLANTILLA_TEMA.md               # Contrato conceptual y experimental
├── ENTORNOS_CLUSTER.md             # Ficha de compilación y ejecución
├── notebooks/                      # 1–3 notebooks por unidad
├── ejemplos/                       # Referencia serial y variantes portables
└── ejercicios/
    ├── <topico>/<ejercicio>/       # Enunciado, esqueleto y pruebas públicas
    └── soluciones/
        └── <topico>/<ejercicio>/   # Solución y pruebas docentes
```

Los directorios temáticos se denominan `01_rocm_hip`, `02_sycl`, `03_kokkos` y `04_raja`. Se aplican los mismos criterios de corrección, trazabilidad y presentación definidos para [`curso/`](../curso/README.md). Si una unidad se utiliza como evaluación, sus soluciones se excluyen de la distribución estudiantil mientras la actividad esté abierta.

## Perfiles de entorno

El material avanzado se construye con perfiles independientes:

- `advanced-hip`: ROCm/HIP sobre un nodo AMD compatible.
- `advanced-sycl`: AdaptiveCpp y un backend declarado.
- `advanced-kokkos-cpu`, `advanced-kokkos-hip` o `advanced-kokkos-cuda`.
- `advanced-raja-cpu`, `advanced-raja-hip` o `advanced-raja-cuda`.

La configuración acepta las raíces `COURSE_ROCM_ROOT`, `COURSE_ADAPTIVECPP_ROOT`, `COURSE_KOKKOS_ROOT` y `COURSE_RAJA_ROOT`. Cada preset se incorpora cuando exista al menos un ejemplo compilable y probado con ese perfil.

## Bibliografía y especificaciones

### ROCm/HIP

- AMD, [ROCm 7.2.3 release notes](https://rocm.docs.amd.com/en/docs-7.2.3/about/release-notes.html).
- AMD, [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).
- AMD, [ROCm Linux compatibility and installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

### SYCL

- Khronos Group, [SYCL 2020 Specification, revision 11](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html).
- James Brodman et al., *Data Parallel C++*, 2.ª ed., Apress, 2023. Durante su lectura se distinguen las extensiones oneAPI del núcleo normativo de SYCL.
- AdaptiveCpp, [documentación y código fuente](https://github.com/AdaptiveCpp/AdaptiveCpp).

### Kokkos

- Kokkos Team, [Kokkos Core documentation](https://kokkos.org/kokkos-core-wiki/).
- H. Carter Edwards, Christian R. Trott y Daniel Sunderland, “Kokkos: Enabling manycore performance portability through polymorphic memory access patterns”, *Journal of Parallel and Distributed Computing*, 2014.
- David Hollman et al., “Kokkos 3: Programming model extensions for the exascale era”, *IEEE TPDS*, 2022.

### RAJA

- LLNL, [RAJA 2025.12.2 User Guide](https://raja.readthedocs.io/en/v2025.12.2/).
- Richard D. Hornung y Jeffrey A. Keasler, *The RAJA Portability Layer: Overview and Status*, LLNL-TR-661403, 2014.
- LLNL, [RAJA Performance Suite](https://github.com/LLNL/RAJAPerf), para comparaciones de kernels y backends.

## Condiciones para ofrecer una unidad

Una unidad podrá ofrecerse cuando cuente con notebooks ejecutados de principio a fin, ejemplos compilables en cada perfil declarado, referencia serial, pruebas de corrección, medición reproducible, ejercicios, soluciones, bibliografía y una matriz de hardware y software efectivamente probada. En su estado actual, este documento establece la planeación académica; la disponibilidad local de los cuatro toolchains se comprobará durante el desarrollo de los ejemplos.
