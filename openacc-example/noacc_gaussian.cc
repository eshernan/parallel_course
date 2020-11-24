
#include <cmath>

#include "gaussian.h"
#include "noacc_gaussian.h"

void noacc_gaussian_pdf(Point* points, int n_points, float* result) {
    for(int i = 0; i < n_points; i++) {
        result[i] = gaussian_pdf_of_point(points[i]);
    }
}
