
#include <cmath>

#include "gaussian.h"

void gaussian_pdf(Point* points, int n_points, float* result) {

#pragma acc kernels
    {
    for(int i = 0; i < n_points; i++) {
        result[i] = gaussian_pdf_of_point(points[i]);
    }
    }
}
