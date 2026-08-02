#ifndef PARTICION_H
#define PARTICION_H

#include <stddef.h>

/* Calcula "workers" rangos [starts[w], ends[w]) que cubren exactamente
 * [0, n) sin huecos ni solapamientos. El resto de n / workers se reparte
 * entre los primeros trabajadores, uno de más cada uno. */
void partition_ranges(size_t n, size_t workers, size_t *starts, size_t *ends);

/* Suma data[0..n) calculando una suma parcial por cada rango de
 * partition_ranges y combinando esas sumas parciales en un único total. */
double partitioned_sum(const double *data, size_t n, size_t workers);

#endif
