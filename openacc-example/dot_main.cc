
#include <iostream>
#include <random>
#include <chrono>

#include "dot.h"
#include "noacc_dot.h"

using namespace std;

int main(int argc, char *argv[])
{
    const int VECTOR_SIZE = 10000000;

    float* v1 = new float[VECTOR_SIZE];
    float* v2 = new float[VECTOR_SIZE];

    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    for(int i = 0; i < VECTOR_SIZE; i++) {
        v1[i] = distribution(generator);
        v2[i] = distribution(generator);
    }

    auto start = chrono::steady_clock::now();

    float device_dot = dot_product(v1, v2, VECTOR_SIZE);
    auto accelerated_time = chrono::steady_clock::now() - start;

    float host_dot = noacc_dot_product(v1, v2, VECTOR_SIZE);
    auto non_accelerated_time = chrono::steady_clock::now() - start - accelerated_time;

    // Print the times.
    std::cout << "Host time: " << chrono::duration<float, milli>(non_accelerated_time).count() << " ms." << endl;
    std::cout << "Device time: " << chrono::duration<float, milli>(accelerated_time).count() << " ms." << endl;
    std::cout << "Speedup: " << chrono::duration<float,milli>(non_accelerated_time).count() / chrono::duration<float, milli>(accelerated_time).count()  << "x." << endl;

    delete[] v1;
    delete[] v2;

    return 0;
}

