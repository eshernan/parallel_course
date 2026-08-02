#include "roofline.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Mide el ancho de banda y el rendimiento alcanzados por vector_triad en
 * este equipo, para compararlos con el techo Roofline calculado en
 * curso/notebooks/01_fundamentos/03_memoria_roofline.ipynb. Este programa
 * no forma parte de ctest: la corrección se verifica por separado en
 * tests/test_roofline.c, y la medición de rendimiento solo tiene sentido
 * después de superar esa prueba (ver docs/PLANEACION_CURSO.md, rúbrica). */
int main(void) {
    const size_t n = 5 * 1000 * 1000;
    const int repetitions = 10;

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    double *out = malloc(n * sizeof(double));
    if (!a || !b || !c || !out) {
        fprintf(stderr, "no se pudo reservar memoria para el benchmark\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = (double)(i % 97) * 0.01;
    }

    struct timespec start, end;
    timespec_get(&start, TIME_UTC);
    for (int rep = 0; rep < repetitions; ++rep) {
        vector_triad(a, b, c, out, n);
    }
    timespec_get(&end, TIME_UTC);

    const double elapsed = (double)(end.tv_sec - start.tv_sec) +
                            (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    const double seconds_per_rep = elapsed / repetitions;
    const double bytes_moved = 4.0 * (double)n * (double)sizeof(double);
    const double flops = 2.0 * (double)n;
    const double achieved_gbs = bytes_moved / seconds_per_rep / 1e9;
    const double achieved_gflops = flops / seconds_per_rep / 1e9;
    const double intensity = triad_arithmetic_intensity();

    printf("n=%zu repeticiones=%d\n", n, repetitions);
    printf("tiempo por repeticion: %.6f s\n", seconds_per_rep);
    printf("ancho de banda alcanzado: %.2f GB/s\n", achieved_gbs);
    printf("rendimiento alcanzado: %.3f GFLOP/s\n", achieved_gflops);
    printf("intensidad aritmetica del triad: %.4f FLOP/byte\n", intensity);
    printf(
        "Reemplace peak_gflops/bandwidth_gbs del notebook por valores de este \n"
        "equipo (o mida bandwidth_gbs con esta misma rutina en modo copia) y \n"
        "compare achieved_gflops contra el techo Roofline resultante.\n");

    free(a);
    free(b);
    free(c);
    free(out);
    return 0;
}
