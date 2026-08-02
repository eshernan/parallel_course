# Imágenes compartidas del curso

Esta carpeta contiene diagramas SVG reutilizables por todos los temas. Cada archivo incluye `title` y `desc` para accesibilidad, usa una paleta común y conserva texto legible sin depender de un notebook ejecutado.

Los diagramas son material conceptual: explican relaciones, jerarquías o secuencias, pero no sustituyen mediciones ni capturas auténticas de herramientas. Se generan con `python3 tools/generate_course_diagrams.py`; el modo `--check` impide que un notebook enlace una versión divergente o ausente.

| Archivo | Relación principal |
|---|---|
| `ruta-reproducible.svg` | Secuencia desde la pregunta y la referencia serial hasta la validación, medición e informe. |
| `capas-toolchain.svg` | Capas desde el código fuente y el estándar hasta el compilador, runtime, sistema operativo y hardware. |
| `dag-camino-critico.svg` | Grafo acíclico leer, partir, tareas A y B, y combinar; el camino crítico pasa por A. |
| `escalabilidad.svg` | Ejes de recursos y aceleración con líneas ideal, limitada por Amdahl y observada. |
| `jerarquia-memoria.svg` | Niveles de registros, cachés, memoria principal y almacenamiento con capacidad y latencia crecientes. |
| `fork-join.svg` | Una región serial crea trabajadores, distribuye trabajo y espera su finalización antes de continuar. |
| `happens-before.svg` | El productor publica datos bajo sincronización y el consumidor los observa después de esperar la condición. |
| `distribucion-trabajo.svg` | Iteraciones repartidas entre cuatro trabajadores y combinación final de resultados parciales. |
| `mpi-comunicacion.svg` | Cuatro rangos intercambian mensajes punto a punto y participan en una operación colectiva. |
| `offload-host-device.svg` | El host prepara datos, transfiere al dispositivo, ejecuta un kernel, recupera resultados y valida. |
| `cuda-grid-tiling.svg` | Un grid contiene bloques, cada bloque contiene hilos y coopera sobre un tile con memoria compartida. |
| `topologia-hibrida.svg` | Dos nodos conectados; cada uno aloja procesos MPI, hilos OpenMP y un dispositivo asignado. |
| `metodo-rendimiento.svg` | Cadena de mediciones repetidas, resumen robusto, perfil por fases, hipótesis y nuevo experimento. |
