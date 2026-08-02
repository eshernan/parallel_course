#include "roofline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_triad_correctness(void) {
    const size_t n = 1000;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    int ok = a && b && c && out;

    if (ok) {
        for (size_t i = 0; i < n; ++i) {
            a[i] = (double)i * 0.5 - 3.0;
            b[i] = (double)(i % 7) + 1.0;
            c[i] = (double)(i % 5) - 2.0;
        }
        vector_triad(a, b, c, out, n);
        for (size_t i = 0; i < n; ++i) {
            const double expected = a[i] + b[i] * c[i];
            if (fabs(out[i] - expected) > 1e-12) {
                fprintf(stderr, "triad incorrecto en i=%zu: esperado=%.6f obtenido=%.6f\n",
                        i, expected, out[i]);
                ok = 0;
                break;
            }
        }
    }

    free(a);
    free(b);
    free(c);
    free(out);
    return ok;
}

static int check_intensity(void) {
    const double expected = 2.0 / 32.0;
    const double got = triad_arithmetic_intensity();
    if (fabs(got - expected) > 1e-9) {
        fprintf(stderr, "intensidad incorrecta: esperado=%.6f obtenido=%.6f\n", expected, got);
        return 0;
    }
    return 1;
}

static int check_regime(void) {
    const struct {
        double peak;
        double bandwidth;
        double intensity;
        int expected_memory_bound;
    } cases[] = {
        {800.0, 120.0, 0.125, 1},
        {800.0, 120.0, 16.0, 0},
        {800.0, 120.0, 800.0 / 120.0 - 0.01, 1},
        {800.0, 120.0, 800.0 / 120.0 + 0.01, 0},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const int got = is_memory_bound(cases[i].peak, cases[i].bandwidth, cases[i].intensity);
        if (got != cases[i].expected_memory_bound) {
            fprintf(stderr, "regimen incorrecto para I=%.4f: esperado=%d obtenido=%d\n",
                    cases[i].intensity, cases[i].expected_memory_bound, got);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    if (!check_triad_correctness()) {
        return 1;
    }
    if (!check_intensity()) {
        return 1;
    }
    if (!check_regime()) {
        return 1;
    }
    printf("ancho_banda_roofline: todas las pruebas superadas\n");
    return 0;
}
