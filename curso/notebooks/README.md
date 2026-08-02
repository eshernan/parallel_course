# Notebooks

Cada tema se desarrolla en uno, dos o tres notebooks, organizados en subcarpetas `00_` a `08_`. La estructura académica se encuentra en `docs/PLANEACION_CURSO.md`; las gráficas se construyen con resultados producidos por los ejecutables de `curso/ejemplos/`.

Las implementaciones extensas permanecen en archivos fuente y no se duplican en las celdas. Tampoco se guardan salidas voluminosas. En el itinerario CPU/MPI, la ejecución completa se podrá automatizar en integración continua.

El tema 00 prevé `01_estandares_compiladores.ipynb`, una explicación breve de la evolución de C y C++, sus revisiones publicadas y el soporte observable en Linux, macOS y Windows. El notebook utilizará pruebas de características y los datos resumidos en [`docs/ESTANDARES_C_CPP.md`](../../docs/ESTANDARES_C_CPP.md); no inferirá conformidad completa a partir de que el compilador acepte una bandera `-std` o `/std`.

## Estilo visual

Las ilustraciones conceptuales toman como [referencia visual](../../docs/images/programacion-paralela-curso-2026-v2.png) el mapa técnico del curso: boceto a lápiz de grafito sobre fondo marfil, rayado suave y acentos azul grisáceo discretos. No se emplean fotografías, renders tridimensionales ni colores saturados.

Las gráficas científicas se generan con los datos de cada experimento, sobre fondo claro y con una paleta compatible con la referencia. Las capturas de perfiles o depuradores se conservan sin alteraciones que puedan cambiar su significado técnico.
