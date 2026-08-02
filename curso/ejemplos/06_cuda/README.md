# Ejemplos CUDA C++

Secuencia prevista:

- `00_device_query`: versión, dispositivo, capacidad de cómputo y límites.
- `01_vector_add`: referencia CPU, kernel con límites y errores comprobados.
- `02_memory_access`: acceso coalescente/no coalescente y ancho de banda.
- `03_matmul_tiled`: versión ingenua y por mosaicos contra referencia CPU.
- `04_reduction`: variantes global, shared, warp y atomic correctas.
- `05_streams`: transferencias asíncronas y solapamiento.

Todos los ejecutables usarán CUDA 13.0.x, C++20, CMake, `cudaEvent` para tiempos GPU, Compute Sanitizer y arquitectura mínima `sm_75`.
