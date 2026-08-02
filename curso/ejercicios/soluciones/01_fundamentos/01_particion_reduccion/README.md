# Solución: partición balanceada y suma por reducción

Implementación de referencia para
[`curso/ejercicios/01_fundamentos/01_particion_reduccion`](../../../../ejercicios/01_fundamentos/01_particion_reduccion/README.md).

`partition_ranges` reproduce `ranges(n, workers)` del notebook
`01_modelos.ipynb`: reparte el cociente entero a todos los trabajadores y
asigna el resto (uno de más) a los primeros `n % workers`. Con
`workers == 0` no se escribe ningún rango, evitando la división por cero.

`partitioned_sum` separa la suma en dos fases explícitas —una por rango
("partir") y la combinación de los totales parciales ("combinar")— en
lugar de sumar el arreglo completo en un solo bucle, para que el código
refleje el DAG `leer -> partir -> {rangos} -> combinar` del notebook y no
solo su resultado numérico.

Compilar y probar igual que el ejercicio público:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
