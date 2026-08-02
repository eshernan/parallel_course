# Partición balanceada y suma por reducción

**Tema:** 01. Fundamentos · **Notebook relacionado:** [`01_modelos.ipynb`](../../../notebooks/01_fundamentos/01_modelos.ipynb)

## Enunciado

El notebook del tema resuelve en Python la partición balanceada de `n`
elementos entre `workers` rangos (`ranges(n, workers)`) y el DAG
`leer -> partir -> {A, B} -> combinar`. Este ejercicio pide la misma idea,
pero en C17 y verificada con pruebas automáticas:

1. Complete `partition_ranges` en `src/particion.c` para que devuelva
   `workers` rangos `[starts[w], ends[w])` que cubran `[0, n)` sin huecos
   ni solapamientos. El resto de `n / workers` debe repartirse entre los
   primeros trabajadores (uno de más cada uno), igual que la función
   `ranges()` del notebook.
2. Complete `partitioned_sum` para que sume cada rango de forma
   independiente ("partir") y combine esas sumas parciales en un único
   total ("combinar"), reproduciendo el DAG de reducción del notebook.

El esqueleto ya compila: `partition_ranges` devuelve rangos vacíos y
`partitioned_sum` devuelve `0.0`. Las pruebas fallan hasta completar
ambas funciones; esa falla inicial es esperada y no indica un error de
construcción.

## Compilar y probar

```sh
python3 validation/preflight.py   # desde la raíz del repositorio
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Casos que se verifican

- Cobertura exacta de `[0, n)` para varias combinaciones de `n` y
  `workers`, incluyendo `n` no divisible entre `workers`, `workers > n`
  y `n = 0`.
- La suma reducida coincide con una suma serial de referencia (tolerancia
  relativa `1e-9`) para 1, 2, 3, 5, 7, 16, 1000 y 2000 trabajadores sobre
  un arreglo de 1000 elementos.

## Criterios de aceptación

- `ctest` termina en verde para todos los casos anteriores.
- El código compila sin advertencias con `-Wall -Wextra -Wpedantic`.
- `partition_ranges` no depende de que `workers` divida a `n`.
- `partitioned_sum` no recorre el arreglo fuera de los rangos calculados.

## Referencias

- [Notebook 01: modelos de paralelismo y descomposición](../../../notebooks/01_fundamentos/01_modelos.ipynb)
- [Protocolo de reproducibilidad](../../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md)
