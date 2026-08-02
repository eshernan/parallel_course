#include "roofline.h"

void vector_triad(const double *a, const double *b, const double *c, double *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i] * c[i];
    }
}

double triad_arithmetic_intensity(void) {
    const double flops_per_element = 2.0;
    const double bytes_per_element = 4.0 * (double)sizeof(double);
    return flops_per_element / bytes_per_element;
}

int is_memory_bound(double peak_gflops, double bandwidth_gbs, double intensity) {
    return (bandwidth_gbs * intensity) < peak_gflops ? 1 : 0;
}
