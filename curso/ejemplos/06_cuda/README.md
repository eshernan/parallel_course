# Ejemplos CUDA C++

Secuencia prevista:

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

Todos los ejecutables usarán CUDA 13.0.x, C++20, CMake, `cudaEvent` para tiempos GPU, Compute Sanitizer y arquitectura mínima `sm_75`. Cada biblioteca debe comprobar su propio estado, reutilizar handles/planes/workspaces durante la región medida y validar contra una referencia CPU o propiedad numérica conocida.

Las comparaciones no incluirán creación de contexto en una medida de kernel, pero sí la reportarán en el tiempo extremo a extremo. Los ejemplos mantendrán los datos residentes cuando ese sea el escenario real y mostrarán explícitamente cuándo la transferencia o conversión de formato elimina la ventaja.
