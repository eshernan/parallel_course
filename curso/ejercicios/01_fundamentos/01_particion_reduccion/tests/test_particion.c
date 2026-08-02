#include "particion.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_coverage(size_t n, size_t workers) {
    size_t *starts = calloc(workers, sizeof(size_t));
    size_t *ends = calloc(workers, sizeof(size_t));
    char *seen = calloc(n == 0 ? 1 : n, 1);
    int ok = starts && ends && seen;

    if (ok) {
        partition_ranges(n, workers, starts, ends);
        for (size_t worker = 0; ok && worker < workers; ++worker) {
            if (ends[worker] < starts[worker] || ends[worker] > n) {
                ok = 0;
                break;
            }
            for (size_t i = starts[worker]; i < ends[worker]; ++i) {
                if (seen[i]) {
                    ok = 0;
                    break;
                }
                seen[i] = 1;
            }
        }
        for (size_t i = 0; ok && i < n; ++i) {
            if (!seen[i]) {
                ok = 0;
            }
        }
    }

    free(starts);
    free(ends);
    free(seen);
    return ok;
}

static double reference_sum(const double *data, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += data[i];
    }
    return total;
}

int main(void) {
    const struct { size_t n; size_t workers; } coverage_cases[] = {
        {23, 4}, {16, 4}, {1, 1}, {7, 3}, {100, 7}, {5, 8}, {0, 3},
    };
    for (size_t c = 0; c < sizeof(coverage_cases) / sizeof(coverage_cases[0]); ++c) {
        if (!check_coverage(coverage_cases[c].n, coverage_cases[c].workers)) {
            fprintf(stderr, "cobertura incorrecta para n=%zu workers=%zu\n",
                    coverage_cases[c].n, coverage_cases[c].workers);
            return 1;
        }
    }

    const size_t n = 1000;
    double *data = malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "no se pudo reservar memoria de prueba\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        data[i] = (double)(i % 13) - 6.0 + 0.5;
    }
    const double expected = reference_sum(data, n);

    const size_t worker_counts[] = {1, 2, 3, 5, 7, 16, 1000, 2000};
    for (size_t k = 0; k < sizeof(worker_counts) / sizeof(worker_counts[0]); ++k) {
        const double got = partitioned_sum(data, n, worker_counts[k]);
        if (fabs(got - expected) > 1e-9 * (fabs(expected) + 1.0)) {
            fprintf(stderr, "suma incorrecta workers=%zu: esperado=%.6f obtenido=%.6f\n",
                    worker_counts[k], expected, got);
            free(data);
            return 1;
        }
    }

    free(data);
    printf("particion_reduccion: todas las pruebas superadas\n");
    return 0;
}
