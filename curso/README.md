# Organización del material docente

Esta carpeta reúne el material preparado para la nueva versión de la asignatura y lo separa de los programas históricos que permanecen en la raíz. Cuando un ejemplo anterior resulta útil, se elabora aquí una versión revisada, con pruebas, documentación y referencia a su procedencia.

## Carpetas

- `notebooks/`: desarrollo conceptual, experimentos y gráficas; de uno a tres notebooks por tema.
- `ejemplos/`: programas compilables utilizados durante las explicaciones y prácticas.
- `ejercicios/`: enunciados, código inicial y pruebas públicas.
- `ejercicios/soluciones/`: soluciones de referencia y pruebas reservadas para el equipo docente.

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

La [planeación de la asignatura](../docs/PLANEACION_CURSO.md) contiene el calendario, las evaluaciones y la bibliografía. El [material de profundización](../topicos_avanzados/README.md) se organiza por fuera del semestre regular.

## Incorporación de material histórico

El código histórico no se traslada sin revisión. La versión docente se incorpora cuando compila con el entorno fijado, pasa las pruebas previstas, comprueba los errores de las APIs, produce resultados trazables y cuenta con un ejercicio asociado.
