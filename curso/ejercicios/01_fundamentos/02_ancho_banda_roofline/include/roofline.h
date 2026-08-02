#ifndef ROOFLINE_H
#define ROOFLINE_H

#include <stddef.h>

/* out[i] = a[i] + b[i] * c[i] para i en [0, n). Vector triad (STREAM). */
void vector_triad(const double *a, const double *b, const double *c, double *out, size_t n);

/* Intensidad aritmética del triad, en FLOP/byte: 2 FLOP (una
 * multiplicación y una suma) por cada 4 doubles de 8 bytes movidos
 * (3 lecturas + 1 escritura). */
double triad_arithmetic_intensity(void);

/* Aplica el límite Roofline: régimen = min(peak_gflops, bandwidth_gbs * intensity).
 * Devuelve 1 si ese régimen queda limitado por ancho de banda
 * (bandwidth_gbs * intensity < peak_gflops), 0 si queda limitado por cómputo. */
int is_memory_bound(double peak_gflops, double bandwidth_gbs, double intensity);

#endif
