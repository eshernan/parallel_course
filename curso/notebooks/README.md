# Notebooks

[Volver al índice de navegación del curso](../../INDICE_CURSO.md)

Cada tema se desarrolla en uno, dos o tres notebooks, organizados en subcarpetas `00_` a `08_`. Los 23 notebooks previstos por la planeación están incorporados y se consultan desde el [índice del curso](../../INDICE_CURSO.md).

Las implementaciones extensas permanecen en archivos fuente y no se duplican en las celdas. Tampoco se guardan salidas voluminosas. En el itinerario CPU/MPI, la ejecución completa se podrá automatizar en integración continua.

## Inventario por tema

| Tema | Guía | Cantidad |
|---:|---|---:|
| 00 | [`Entorno y lenguajes`](00_entorno/README.md) | 2 |
| 01 | [`Fundamentos`](01_fundamentos/README.md) | 3 |
| 02 | [`Memoria compartida`](02_memoria_compartida/README.md) | 3 |
| 03 | [`OpenMP`](03_openmp/README.md) | 3 |
| 04 | [`MPI`](04_mpi/README.md) | 3 |
| 05 | [`OpenMP target`](05_openmp_target/README.md) | 2 |
| 06 | [`CUDA C++`](06_cuda/README.md) | 3 |
| 07 | [`Programación híbrida`](07_hibrido/README.md) | 3 |
| 08 | [`Proyecto final`](08_proyecto/README.md) | 1 |

Las celdas ejecutables utilizan la biblioteca estándar de Python para comprobar modelos, invariantes y cálculos en cualquier plataforma. Las prácticas que requieren MPI, OpenMP target o CUDA distinguen esas comprobaciones portables de la evidencia que debe obtenerse en el hardware real.

## Ruta pedagógica común

Los 23 notebooks siguen el mismo recorrido para reducir la carga cognitiva y hacer visible cuándo termina la explicación y cuándo comienza la práctica:

1. **Antes de empezar:** motivación, prerrequisitos y diagnóstico inicial.
2. **Explicación paso a paso:** conceptos y vocabulario antes de mostrar código.
3. **Mapa visual:** uno o más SVG reutilizables que explican una relación, jerarquía o secuencia.
4. **Ejemplo resuelto:** situación, razonamiento previo, código con aserciones y explicación del resultado.
5. **Ejemplo guiado:** predicción del estudiante, ejecución y lectura razonada.
6. **Comprensión y ejercicios:** preguntas conceptuales y actividades progresivas desde reproducción hasta evidencia reproducible.

`validation/validate_navigation.py` exige estas secciones en ese orden, un mínimo explicativo y la correspondencia exacta entre imágenes, manifiesto y metadatos.

## Estilo visual

Las ilustraciones conceptuales se almacenan en la carpeta común [`curso/images/`](../images/README.md) y toman como [referencia visual](../../docs/images/programacion-paralela-curso-2026-v2.png) el mapa técnico del curso: fondo marfil, líneas sobrias y acentos azul grisáceo discretos. Los SVG incluyen título y descripción accesibles y pueden reutilizarse desde cualquier capítulo.

Las gráficas científicas se generan con los datos de cada experimento, sobre fondo claro y con una paleta compatible con la referencia. Las capturas de perfiles o depuradores se conservan sin alteraciones que puedan cambiar su significado técnico.

Todo notebook enlaza el [índice del curso](../../INDICE_CURSO.md) y la guía de su tema en su primera y última celda Markdown. La comprobación previa exige el inventario completo, verifica los 13 diagramas compartidos, ejecuta las celdas de código y evita que un cambio de nombre deje rutas de navegación rotas.
