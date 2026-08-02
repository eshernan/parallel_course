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

## Estilo visual

Las ilustraciones conceptuales toman como [referencia visual](../../docs/images/programacion-paralela-curso-2026-v2.png) el mapa técnico del curso: boceto a lápiz de grafito sobre fondo marfil, rayado suave y acentos azul grisáceo discretos. No se emplean fotografías, renders tridimensionales ni colores saturados.

Las gráficas científicas se generan con los datos de cada experimento, sobre fondo claro y con una paleta compatible con la referencia. Las capturas de perfiles o depuradores se conservan sin alteraciones que puedan cambiar su significado técnico.

Todo notebook enlaza el [índice del curso](../../INDICE_CURSO.md) y la guía de su tema en su primera y última celda Markdown. La comprobación previa exige el inventario completo, ejecuta las celdas de código y evita que un cambio de nombre deje rutas de navegación rotas.
