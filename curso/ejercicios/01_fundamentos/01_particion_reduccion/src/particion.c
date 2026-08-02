#include "particion.h"

void partition_ranges(size_t n, size_t workers, size_t *starts, size_t *ends) {
    /* TODO(01): repartir n elementos entre "workers" rangos balanceados.
     * Referencia: la función ranges(n, workers) de
     * curso/notebooks/01_fundamentos/01_modelos.ipynb.
     * - q = n / workers, r = n % workers.
     * - Los primeros r trabajadores reciben q + 1 elementos; el resto, q.
     * - starts[0] = 0; cada rango siguiente empieza donde termina el anterior. */
    (void)n;
    for (size_t worker = 0; worker < workers; ++worker) {
        starts[worker] = 0;
        ends[worker] = 0;
    }
}

double partitioned_sum(const double *data, size_t n, size_t workers) {
    /* TODO(02): usar partition_ranges para sumar cada rango por separado
     * y combinar los resultados parciales en un único total (leer -> partir
     * -> {sumar rangos} -> combinar, igual que el DAG del notebook 01_modelos). */
    (void)data;
    (void)n;
    (void)workers;
    return 0.0;
}
