#include "roofline.h"

void vector_triad(const double *a, const double *b, const double *c, double *out, size_t n) {
    /* TODO(01): calcular out[i] = a[i] + b[i] * c[i] para cada i en [0, n). */
    (void)a;
    (void)b;
    (void)c;
    for (size_t i = 0; i < n; ++i) {
        out[i] = 0.0;
    }
}

double triad_arithmetic_intensity(void) {
    /* TODO(02): devolver los FLOP por byte del triad. Cada elemento hace
     * 2 FLOP (multiplicación + suma) y mueve 4 doubles de 8 bytes
     * (3 lecturas + 1 escritura). Ver curso/notebooks/01_fundamentos/03_memoria_roofline.ipynb,
     * sección "Límite Roofline". */
    return 0.0;
}

int is_memory_bound(double peak_gflops, double bandwidth_gbs, double intensity) {
    /* TODO(03): reproducir la regla de la celda "Límite Roofline" del
     * notebook: devolver 1 si bandwidth_gbs * intensity < peak_gflops,
     * 0 en caso contrario. */
    (void)peak_gflops;
    (void)bandwidth_gbs;
    (void)intensity;
    return -1;
}
