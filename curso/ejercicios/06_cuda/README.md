# Ejercicios CUDA

1. Corregir y validar una suma vectorial con tamaños no múltiplos del bloque.
2. Explicar y medir coalescencia con dos patrones de acceso.
3. Implementar matrix multiply por mosaicos y comparar contra referencia CPU.
4. Implementar dos reducciones y analizar precisión/rendimiento.
5. Solapar transferencias y cómputo con streams sin cambiar el resultado.
6. Sustituir una reducción manual por Thrust o CUB y justificar la elección, almacenamiento temporal y punto de cruce por tamaño.
7. Comparar GEMM ingenuo, tiled y cuBLAS usando tiempo de ejecución y tiempo extremo a extremo.
8. Elegir entre cuFFT, cuSPARSE, cuSOLVER o cuRAND para un caso dado, construir un ejemplo mínimo y explicar por qué la biblioteca ofrece —o no— ventaja.

Cada entrega debe pasar Compute Sanitizer, verificar errores CUDA y estados de biblioteca, y exportar métricas antes de producir gráficas.

El enunciado declarará CUDA Toolkit, capacidad de cómputo, memoria mínima, targets CMake, librerías, datos, precisión y tolerancia. En clúster indicará módulos, partición, recurso GPU, tiempo límite y comando de ejecución. La conclusión debe separar preparación, transferencia, ejecución y validación; no basta con afirmar que una biblioteca “es más rápida”.
