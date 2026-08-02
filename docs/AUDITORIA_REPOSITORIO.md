# Auditoría del repositorio para un curso de programación paralela

Fecha de corte: 2 de agosto de 2026.

## Dictamen ejecutivo

El repositorio es un banco de ejemplos de un curso dictado en 2020, no un curso universitario completo ni reproducible. Tiene una cobertura temática inicial valiosa —Pthreads, OpenMP, MPI, CUDA, OpenACC, OpenCL y concurrencia en C++—, pero carece de objetivos de aprendizaje, secuencia didáctica, notebooks, ejercicios, soluciones, evaluaciones, pruebas, automatización y un entorno versionado. Además, varios programas tienen errores de compilación o de corrección y algunos ejemplos de rendimiento miden operaciones incorrectas.

La recomendación es **reconstruir el curso alrededor de los conceptos y rescatar selectivamente el código**, no impartir el árbol actual en el orden en que está. Pthreads, OpenMP, MPI y aceleradores siguen siendo pertinentes. OpenCL, OpenACC y los algoritmos paralelos artesanales de C++ deben pasar a material optativo o histórico.

## Alcance y evidencia

Se revisaron los 223 archivos versionados, el historial de Git, los README, scripts, Makefiles/CMake, fuentes C/C++/CUDA/Fortran, binarios y bibliotecas incluidas. Se ejecutaron compilaciones con advertencias estrictas para el código CPU y MPI, y pruebas breves de ejemplos que sí construyen en el equipo disponible.

- 223 archivos versionados.
- 0 notebooks (`.ipynb`).
- 0 suites de pruebas y 0 configuración de integración continua.
- 18.072 líneas en archivos fuente, incluyendo código auxiliar de terceros.
- 31 artefactos binarios, objetos o archivos empaquetados versionados.
- Aproximadamente 78 MB de contenido versionado; `cuda/Common/FreeImage` concentra cerca de 73 MB en bibliotecas precompiladas para múltiples arquitecturas.
- Último commit del 18 de diciembre de 2020; el README declara explícitamente que corresponde al curso 2020-3.

## Qué está bien

### Cobertura técnica inicial

- El repositorio toca los tres modelos que deben formar el núcleo del curso: memoria compartida, memoria distribuida y aceleradores.
- Hay ejemplos pequeños que pueden explicarse en una sesión: creación de hilos, exclusión mutua, difusión MPI, anillo, suma vectorial y reducción.
- La progresión de CUDA desde suma vectorial hasta reducción, divergencia, ocupación, grupos cooperativos y precisión mixta tiene una intención pedagógica reconocible.
- `mpi/send_receive.c` usa `MPI_Sendrecv`, compila con MPICH 5.0.1 y produjo correctamente el paso en anillo con cuatro procesos.
- Los ejemplos `cpp_10/parallel_algorithms/for_each.cpp` y `sort.cpp` compilan y se ejecutan; pueden reutilizarse como estudios de caso sobre los riesgos de escribir un *runtime* casero, no como implementación recomendada.
- El código de Andy Thomason conserva su licencia MIT y gran parte de los auxiliares de NVIDIA conserva los encabezados de licencia originales.

### Casos de estudio rescatables

- Cálculo de π: útil para discutir descomposición, reducción, reproducibilidad y escalabilidad, después de corregir el generador aleatorio y parametrizar el tamaño.
- Simulación de tráfico: puede convertirse en proyecto de dominio si se aclara la procedencia, se eliminan binarios/duplicados y se moderniza la construcción.
- Reducciones: muy buenas para comparar localidad, sincronización, atomics, precisión numérica y ancho de banda.
- Multiplicación de matrices: adecuada para introducir intensidad aritmética, *tiling* y Roofline, pero las versiones actuales deben reescribirse y validarse.

## Qué está mal y bloquea su uso docente

### 1. No existe una arquitectura de curso

No hay resultados de aprendizaje, prerrequisitos verificables, calendario, rúbricas, política de evaluación, bibliografía por tema ni relación entre explicación, ejemplo y ejercicio. Tampoco existen notebooks, carpetas de ejercicios o soluciones. El README solo muestra una instalación genérica de GCC y un ejemplo Pthreads.

### 2. No hay reproducibilidad

- No se fija versión de compilador, estándar de lenguaje, MPI, OpenMP, CUDA ni sistema operativo.
- No hay contenedor, archivo de entorno, *toolchain file*, `CMakePresets.json` ni configuración central de CMake.
- Los Makefiles de CUDA generan código para `sm_35`, `sm_37`, `sm_50`, `sm_52`, `sm_60`, `sm_61` y `sm_70`. CUDA 13 eliminó la compilación fuera de línea para arquitecturas anteriores a Turing (`compute capability` 7.5).
- `cpp_10/parallel_algorithms/Makefile` y `cmake_install.cmake` son productos generados y contienen rutas absolutas de otra máquina.
- Los scripts Slurm contienen correo personal y rutas `/home/ehernandez/...`; además usan `mpirun` en lugar de una plantilla portable con `srun`/`mpiexec` elegida según el clúster.
- `mpi/MPI_PI/run.sh` entra en directorios `sequential` que no existen.

### 3. Hay errores de compilación y comportamiento indefinido

Ejemplos representativos:

- `shared_memory/cond_pthreads.c` y `shared_memory/test_pthreads.c` declaran `void main`, inválido en C alojado.
- `shared_memory/test_pthreads.c` usa un formato incorrecto para `pid_t` y escribe `n` en lugar de un salto de línea.
- Varias rutinas Pthreads prometen devolver `void *` pero terminan sin `return`; otras nunca hacen `pthread_join` y sustituyen la sincronización por bucles infinitos.
- `shared_memory/cond_pthreads.c` llama `pthread_cond_wait` sin bloquear el mutex asociado: el ejemplo es incorrecto, no solo incompleto.
- `pthreads/thread_creation.c`, `thread_unsafe.c` y varios ejemplos OpenMP contienen carreras de datos. Pueden conservarse únicamente si se marcan como versiones defectuosas y se acompañan de una versión corregida y una prueba con sanitizadores.
- `openmp/data_sharing.c` contiene C++ pese a usar extensión `.c`; imprime una variable privada no inicializada y modifica variables compartidas sin sincronización.
- `openmp/paralle_for.c` comparte `ID` y el búfer de hostname entre hilos, creando carreras.
- `cpp_10/example.cpp` no compila porque una frase explicativa quedó como código en la línea 24.
- `mpi/hello_mpi.c` no compila de forma portable por su uso de `HOST_NAME_MAX`; tampoco muestra `rank` o `size`.
- `mpi/bcast.c` imprime `buf` sin inicializar en todos los procesos no raíz antes de la difusión.

### 4. Hay errores numéricos y de GPU graves

- Los SGEMM de `cuda/03_cuda_thread_programming/02_cuda_occupancy` y `04_performance_limiter` indexan B con `i*K+col` en vez de `i*M+col`, copian A a `d_B` y `d_C`, no protegen los bordes y no verifican el resultado.
- El temporizador de `02_cuda_occupancy/sgemm.cu` se detiene sin una sincronización explícita del dispositivo; mide principalmente el lanzamiento asíncrono.
- `03_threadsync_and_reduction/reduction_global_kernel.cu` actualiza y lee posiciones solapadas dentro del mismo kernel, por lo que tiene carreras de datos.
- `10_atomic_operation/reduction_kernel.cu` no comprueba `idx_x < size` y el acumulador no se inicializa correctamente a cero; el *benchmark* copia la entrada sobre la salida antes de sumar.
- La reducción de `09_loop_unrolling` calcula mal el número de warps activos cuando el tamaño del bloque no es múltiplo exacto de 32.
- La mayoría de llamadas CUDA ignora códigos de error y no usa `cudaGetLastError`/`cudaPeekAtLastError`; una ejecución fallida puede presentarse como medición válida.
- Hay copias exactas de varios archivos CUDA y de reducciones, lo que facilita que una corrección quede aplicada solo en una variante.

### 5. El repositorio contiene productos de construcción y dependencias vendorizadas

Hay ejecutables Mach-O y ELF, objetos `.o`, bibliotecas `.a/.so/.dll/.lib`, una solución de Visual Studio, salidas de CMake y un `tar` que duplica el árbol de tráfico. Estos archivos no deben vivir en el repositorio pedagógico salvo que exista una razón documentada y una política de artefactos.

### 6. Procedencia y licencias requieren una revisión

La licencia MIT de la raíz no sustituye las licencias de terceros. Los auxiliares NVIDIA y FreeImage conservan avisos, pero falta un inventario `THIRD_PARTY_NOTICES`. `mpi/MPI_PI` atribuye a KiwenLau sin indicar licencia local; el origen/licencia de `mpi/traffic` debe verificarse antes de redistribuirlo como material del curso.

## Qué requiere revisión, por área

| Área actual | Decisión | Trabajo necesario |
|---|---|---|
| `pthreads/` | Conservar y reescribir | Terminar hilos, comprobar errores, eliminar bucles infinitos, añadir versiones `buggy/fixed` y ThreadSanitizer/Helgrind. |
| `shared_memory/` | Fusionar con Pthreads | `cond_pthreads.c` es peligroso como ejemplo; reemplazar por productor-consumidor correcto. |
| `openmp/` | Conservar como núcleo | Corregir extensiones, `default(none)`, carreras, validación y medición; añadir tareas, afinidad, *false sharing* y SIMD. |
| `mpi/` | Conservar como núcleo | Actualizar a MPI 5.0/MPICH 5.0.1, corregir π/semillas, scripts y errores; añadir no bloqueante, colectivas, tipos, topologías y escalado. |
| `cuda/` | Rescatar conceptos, reescribir código | Validación automática, manejo de errores, CMake moderno, arquitecturas 7.5+, Nsight/Compute Sanitizer y resultados reproducibles. |
| `openacc*/` | Optativo | El ejemplo actual depende de NVIDIA HPC SDK; no satisface por sí mismo el requisito de compilador abierto y duplica OpenMP target. |
| `opencl/` | Histórico/optativo | Los dos archivos son idénticos; `hello.c` es C++, usa el antiguo `CL/cl.hpp`, exige GPU+FP64 y no tiene construcción. OpenCL sigue vigente, pero no cabe como API principal del semestre. |
| `cpp_10/parallel_algorithms/` | Estudio de caso, no núcleo | El texto de 2016 anuncia C++17 “próximamente”; reemplazar por C++20 (`std::thread`, `std::jthread`, atomics) y comparar con algoritmos estándar sin vender el código artesanal como biblioteca. |
| `mpi/traffic/` | Proyecto opcional | Aclarar licencia, eliminar duplicados/binarios, añadir conjunto de datos, pruebas y versión serial de referencia. |

## Contenido que falta y debe añadirse

1. Modelos de costo: trabajo, *span*, overhead, ley de Amdahl, ley de Gustafson, escalado fuerte/débil y eficiencia.
2. Arquitectura: jerarquía de memoria, coherencia, NUMA, afinidad, *false sharing*, vectorización e intensidad aritmética/Roofline.
3. Corrección: modelo de memoria, carreras, interbloqueo, determinismo, reducción reproducible y precisión de punto flotante.
4. Metodología de rendimiento: calentamiento, repeticiones, medianas/intervalos, aislamiento, tamaño de problema, validación previa y perfiles.
5. Herramientas: sanitizadores, Valgrind/Helgrind donde aplique, `perf`, Nsight Systems/Compute, Compute Sanitizer, CMake/CTest y Slurm.
6. Patrones: *map*, *stencil*, reducción, *scan*, partición, cola de trabajo, tareas/DAG, *pipeline* y descomposición de dominio.
7. Programación híbrida MPI+OpenMP, niveles de soporte de hilos y mapeo proceso-hilo-núcleo.
8. Pruebas unitarias y de propiedades con referencia serial; CI para todo lo que no requiera GPU y pruebas GPU separadas.

## Qué no vale la pena dictar en el núcleo obligatorio

- La API OpenCL actual del repositorio. OpenCL 3.1 existe, pero modernizar el *host API* consumiría tiempo que aporta menos que profundizar en memoria, MPI, OpenMP y un modelo de acelerador.
- OpenACC y OpenMP target como dos unidades completas separadas. Elegir OpenMP target en el núcleo y dejar OpenACC como comparación de una sesión o lectura.
- La implementación manual `par::sort`/`par::for_each` como solución moderna. Sirve para crítica de diseño, medición y excepciones (`wait` frente a `get`).
- Los auxiliares completos de CUDA Samples y FreeImage. Ningún ejemplo activo necesita 73 MB de bibliotecas precompiladas.
- Arquitecturas CUDA anteriores a `sm_75` con CUDA 13. Si el laboratorio solo posee Pascal/Volta, debe congelarse explícitamente CUDA 12.9 y el driver 580 como entorno legado, no mezclarlo con el itinerario moderno.
- Ejercicios cuya única actividad sea cambiar el número de hilos y observar `printf`. Cada laboratorio debe pedir predicción, validación, medición y explicación.

## Prioridad de saneamiento

### P0 — antes de dictar

- Añadir estructura de curso, notebooks, ejercicios, soluciones privadas/públicas según política y rúbricas.
- Fijar versiones y crear un entorno Linux reproducible.
- Retirar binarios/objetos del control de versiones y añadir `.gitignore`.
- Corregir o aislar todos los ejemplos con comportamiento indefinido y resultados GPU incorrectos.
- Añadir validación serial y fallar ante cualquier error CUDA/MPI/Pthreads.
- Resolver procedencia/licencias de material de terceros.

### P1 — durante la reconstrucción

- Construcción central con CMake/CTest y CI CPU/MPI.
- Reemplazar scripts Slurm y Makefiles CUDA.
- Añadir gráficos generados desde datos, no imágenes estáticas: escalado, eficiencia, ancho de banda, Roofline y error numérico.
- Añadir guía docente y guía de instalación estudiantil.

### P2 — mejoras

- Laboratorio GPU remoto o cola Slurm reproducible.
- Autocalificación de ejercicios públicos y pruebas ocultas.
- Extensiones optativas OpenCL 3.1, OpenACC 3.4, SYCL o MPI avanzado.

## Conclusión

El repositorio tiene semillas técnicas útiles, especialmente para MPI, OpenMP y la progresión conceptual de CUDA, pero **no debe dictarse tal como está**. La reconstrucción propuesta en `PLANEACION_CURSO.md` conserva el valor de esos ejemplos, corrige la base metodológica y amplía el curso a 19 semanas para dar a CUDA un módulo propio de seis sesiones sin sacrificar fundamentos, MPI u OpenMP.
