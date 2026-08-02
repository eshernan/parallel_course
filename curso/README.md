# Estructura objetivo del curso

Esta carpeta separa el material docente nuevo del banco histórico situado en la raíz. La migración será selectiva: un ejemplo antiguo solo entra aquí después de corregirse, probarse, documentarse y declarar su procedencia.

## Carpetas

- `notebooks/`: explicación ejecutable y gráficas; entre uno y tres notebooks por tema.
- `ejemplos/`: código fuente compilable invocado por los notebooks.
- `ejercicios/`: enunciados, esqueletos y pruebas públicas.
- `ejercicios/soluciones/`: soluciones de referencia y pruebas docentes.

## Temas y prefijos

| Prefijo | Tema |
|---:|---|
| `00` | Entorno reproducible |
| `01` | Fundamentos, arquitectura y rendimiento |
| `02` | Memoria compartida: Pthreads y C++20 |
| `03` | OpenMP |
| `04` | MPI |
| `05` | Aceleradores portables: OpenMP target |
| `06` | CUDA C++ |
| `07` | Programación híbrida y perfilado |
| `08` | Proyecto final |

La planeación completa está en [`docs/PLANEACION_CURSO.md`](../docs/PLANEACION_CURSO.md). Los contenidos extracurriculares se planean de manera independiente en [`topicos_avanzados/README.md`](../topicos_avanzados/README.md).

## Regla de promoción

No se mueve código desde el árbol histórico: se crea una versión nueva con referencia al original. Antes de considerarla docente debe compilar con el entorno fijado, pasar pruebas, comprobar errores, producir resultados trazables y tener un ejercicio asociado.
