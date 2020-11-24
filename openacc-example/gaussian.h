#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "point.h"

void gaussian_pdf(Point* points, int n_points, float* result);

inline float mahalanobis_distance(const Point& point) {
    return sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

inline float gaussian_pdf_of_point(const Point& point) {
    return exp(-0.5 * mahalanobis_distance(point)) / sqrt(pow(2 * M_PI, 3.0));
}
#endif
