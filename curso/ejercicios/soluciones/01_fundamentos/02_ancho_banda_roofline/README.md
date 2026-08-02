# Solución: vector triad, intensidad aritmética y régimen Roofline

Implementación de referencia para
[`curso/ejercicios/01_fundamentos/02_ancho_banda_roofline`](../../../../ejercicios/01_fundamentos/02_ancho_banda_roofline/README.md).

`triad_arithmetic_intensity` deja explícitos los dos números que el
notebook trata como constantes (2 FLOP y 4 doubles de 8 bytes por
elemento) en vez de devolver `0.0625` directamente, para que el nombre
de cada variable documente de dónde sale la intensidad. `is_memory_bound`
traduce la comparación `bandwidth_gbs * intensity < peak_gflops` de la
celda "Límite Roofline" a un valor booleano entero, sin introducir una
tercera categoría para el punto de transición exacto (por eso las
pruebas evitan comparar justo en la cresta).

Compilar y probar igual que el ejercicio público:

```sh
cmake -S . -B build
cmake --build build --target exercise_02_ancho_banda_roofline
ctest --test-dir build --output-on-failure
cmake --build build --target roofline_benchmark
./build/roofline_benchmark
```
