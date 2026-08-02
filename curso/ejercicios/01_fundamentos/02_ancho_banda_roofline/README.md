# Vector triad, intensidad aritmética y régimen Roofline

**Tema:** 01. Fundamentos · **Notebook relacionado:** [`03_memoria_roofline.ipynb`](../../../notebooks/01_fundamentos/03_memoria_roofline.ipynb)

## Enunciado

El notebook calcula el techo Roofline y clasifica intensidades en
Python, con números ilustrativos (`peak_gflops = 800`,
`bandwidth_gbs = 120`). Este ejercicio traslada esa lógica a C17 y la
conecta con una medición real:

1. Complete `vector_triad` en `src/roofline.c`: `out[i] = a[i] + b[i] * c[i]`
   para cada `i`. Es el kernel *STREAM triad*, el mismo tipo de operación
   que sustenta el cálculo de ancho de banda del notebook.
2. Complete `triad_arithmetic_intensity`: cada elemento del triad hace
   2 FLOP (una multiplicación y una suma) y mueve 4 `double` de 8 bytes
   (3 lecturas + 1 escritura). Calcule FLOP/byte.
3. Complete `is_memory_bound` reproduciendo exactamente la regla de la
   celda **Límite Roofline** del notebook: régimen de memoria cuando
   `bandwidth_gbs * intensity < peak_gflops`.

El esqueleto ya compila; las tres funciones devuelven valores de relleno
(`0.0`, `0.0`, `-1`) y las pruebas fallan hasta completarlas.

## Compilar y probar (corrección)

```sh
python3 validation/preflight.py
cmake -S . -B build
cmake --build build --target exercise_02_ancho_banda_roofline
ctest --test-dir build --output-on-failure
```

`ctest` solo verifica corrección: el triad frente a una referencia
exacta, la intensidad exacta del triad y la clasificación de régimen
para casos por debajo y por encima del punto de transición. No mide
tiempo, siguiendo la rúbrica del curso: el rendimiento se evalúa
después de superar la corrección, no en su lugar.

## Práctica reproducible (medición real, no evaluada por ctest)

```sh
cmake --build build --target roofline_benchmark
./build/roofline_benchmark
```

`src/benchmark.c` mide el ancho de banda y el rendimiento reales de
`vector_triad` en su equipo. Con ese resultado:

1. Registre `achieved_gbs` y `achieved_gflops` junto con el manifiesto
   de plataforma (`validation/platform_manifest.py`).
2. Sustituya `peak_gflops`/`bandwidth_gbs` del notebook por valores de
   su propio equipo (o mida `bandwidth_gbs` adaptando el kernel a una
   copia pura) y recalcule el punto de transición.
3. Compare `achieved_gflops` contra ese nuevo techo Roofline: indique si
   su medición cae por debajo del techo (como debe ser) y a qué
   distancia relativa.

## Criterios de aceptación

- `ctest` termina en verde para las tres funciones.
- El código compila sin advertencias con `-Wall -Wextra -Wpedantic`.
- El informe de la práctica reproducible incluye ancho de banda medido,
  intensidad, techo recalculado y manifiesto de plataforma.

## Referencias

- [Notebook 03: jerarquía de memoria y Roofline](../../../notebooks/01_fundamentos/03_memoria_roofline.ipynb)
- [Protocolo de reproducibilidad](../../../../docs/REPRODUCIBILIDAD_EJERCICIOS.md)
