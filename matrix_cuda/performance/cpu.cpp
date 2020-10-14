#include "mmult.h"

void mmult(int m, int n, int k, const float * a, const float * b, float * c)
{
    for (int i = 0; i < m; ++i) // filas
    {
        for (int j = 0; j < k; ++j) //columnas
        {
            c[i * k + j] = 0;

            for (int l = 0; l < n; ++l)
                c[i * k + j] += a[i * n + l] * b[l * k + j];
        }
    }
}
