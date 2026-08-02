#include "particion.h"

#include <stdlib.h>

void partition_ranges(size_t n, size_t workers, size_t *starts, size_t *ends) {
    if (workers == 0) {
        return;
    }
    const size_t quotient = n / workers;
    const size_t remainder = n % workers;
    size_t start = 0;
    for (size_t worker = 0; worker < workers; ++worker) {
        const size_t size = quotient + (worker < remainder ? 1 : 0);
        starts[worker] = start;
        ends[worker] = start + size;
        start += size;
    }
}

double partitioned_sum(const double *data, size_t n, size_t workers) {
    if (workers == 0) {
        return 0.0;
    }
    size_t *starts = calloc(workers, sizeof(size_t));
    size_t *ends = calloc(workers, sizeof(size_t));
    double total = 0.0;

    partition_ranges(n, workers, starts, ends);
    for (size_t worker = 0; worker < workers; ++worker) {
        double partial = 0.0;
        for (size_t i = starts[worker]; i < ends[worker]; ++i) {
            partial += data[i];
        }
        total += partial;
    }

    free(starts);
    free(ends);
    return total;
}
