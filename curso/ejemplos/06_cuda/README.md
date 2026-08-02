# Inventario previsto de ejemplos CUDA C++

Esta carpeta todavía no contiene fuentes CUDA compilables. La lista siguiente es la especificación de trabajo para incorporarlas en cambios posteriores; no constituye evidencia de implementación ni de ejecución sobre GPU.

La secuencia de ejemplos es la siguiente:

- `00_device_query`: versión, dispositivo, capacidad de cómputo y límites.
- `01_vector_add`: referencia CPU, kernel con límites y errores comprobados.
- `02_memory_access`: acceso coalescente/no coalescente y ancho de banda.
- `03_matmul_tiled`: versión ingenua y por mosaicos contra referencia CPU.
- `04_reduction`: variantes global, shared, warp y atomic correctas.
- `05_streams`: transferencias asíncronas y solapamiento.
- `06_cccl_primitives`: `transform`/`reduce` con Thrust y reducción con CUB, incluyendo reutilización de almacenamiento temporal.
- `07_cublas_gemm`: GEMM ingenuo, tiled y cuBLAS con dimensiones no cuadradas, layouts documentados y tiempos kernel/extremo a extremo.
- `08_cufft_batched`: R2C/C2R con creación y reutilización de plan, workspace y normalización verificada.
- `09_cusparse_spmv`: SpMV con CSR y API genérica, descriptor, consulta de buffer y matrices con distintas estructuras.
- `10_cusolver_dense`: sistema denso pequeño con factorización, workspace y comprobación de `info`.
- `11_curand_montecarlo`: generador host/device, semilla reproducible y consumo de muestras sin retorno innecesario al host.

Los futuros ejecutables deberán construirse con CUDA 13.0.x, C++20 y CMake para arquitecturas `sm_75` o posteriores. Los tiempos de GPU deberán obtenerse con `cudaEvent` y la revisión de memoria deberá realizarse con Compute Sanitizer. En los ejemplos de biblioteca se comprobarán los estados de retorno, se reutilizarán handles, planes y espacios de trabajo durante la región medida, y se validará el resultado contra una referencia en CPU o una propiedad numérica conocida.

La creación del contexto queda por fuera del tiempo de kernel y se incluye en el reporte extremo a extremo. Cuando el caso de uso supone datos residentes en la GPU, esa condición se conserva durante la medición. El análisis también identifica los tamaños para los cuales la transferencia o la conversión de formato absorbe la posible ganancia.
