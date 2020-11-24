
#include <iostream>
#include <random>
#include <chrono>

#include "point.h"
#include "gaussian.h"
#include "noacc_gaussian.h"

using namespace std;

int main(int argc, char *argv[])
{
    const int POINT_CLOUD_SIZE = 100000;
    Point* points = new Point[POINT_CLOUD_SIZE];
    float* probabilities = new float[POINT_CLOUD_SIZE];

    // Generate random point cloud.
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-3.0, 3.0);
    for(int i = 0; i < POINT_CLOUD_SIZE; i++) {
        points[i].x = distribution(generator);
        points[i].y = distribution(generator);
        points[i].z = distribution(generator);
    }

    // Compute the PDF at the demanded points.
    auto start = chrono::steady_clock::now();
    gaussian_pdf(points, POINT_CLOUD_SIZE, probabilities);

    auto accelerated_time = chrono::steady_clock::now() - start;
    noacc_gaussian_pdf(points, POINT_CLOUD_SIZE, probabilities);

    auto non_accelerated_time = chrono::steady_clock::now() - start - accelerated_time;

    // Print the times.
    std::cout << "Host time: " << chrono::duration<float, milli>(non_accelerated_time).count() << " ms." << endl;
    std::cout << "Device time: " << chrono::duration<float, milli>(accelerated_time).count() << " ms." << endl;
    std::cout << "Speedup: " << chrono::duration<float,milli>(non_accelerated_time).count() / chrono::duration<float, milli>(accelerated_time).count()  << "x." << endl;

    delete[] probabilities;
    delete[] points;

    return 0;
}
