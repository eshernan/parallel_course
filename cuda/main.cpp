#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "mmult.h"

int main()
{
    const int m = BLOCK_SIZE * 12, n = BLOCK_SIZE * 18, k = BLOCK_SIZE * 24;
    float * a, * b, * c, * a_gpu, * b_gpu, * c_gpu, * c_verify;

    a = new float[m * n];
    b = new float[n * k];
    c = new float[m * k];
    c_verify = new float[m * k];

    cudaMalloc((void **) &a_gpu, m * n * sizeof *a_gpu);
    cudaMalloc((void **) &b_gpu, n * k * sizeof *b_gpu);
    cudaMalloc((void **) &c_gpu, m * k * sizeof *c_gpu);

    for (int i = 0; i < m * n; ++i)
        a[i] = float(rand()) / RAND_MAX;

    for (int i = 0; i < n * k; ++i)
        b[i] = float(rand()) / RAND_MAX;

    cudaMemcpy(a_gpu, a, m * n * sizeof *a_gpu, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, n * k * sizeof *b_gpu, cudaMemcpyHostToDevice);

    float sTime;
    clock_t start, finish;
    
    start = clock();
    mmult(m, n, k, a, b, c);
    finish = clock();
    
    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on CPU: %lf sec", sTime);

    start = clock();
    mmult_gpu(m, n, k, a_gpu, b_gpu, c_gpu);
    cudaThreadSynchronize();
    cudaMemcpy(c_verify, c_gpu, m * k * sizeof *c_gpu, cudaMemcpyDeviceToHost);
    finish = clock();

    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on GPU: %lf sec",sTime);

    float difference = 0;

    for (int i = 0; i < m * k; ++i)
        difference += (c[i] - c_verify[i]) * (c[i] - c_verify[i]);

    if (difference < 1e-5f)
        std::cout << "Test passed.\n";
    else
        std::cout << "Test failed (diff = " << difference << ").\n";

    cudaMemset(c_gpu, 0, m * k * sizeof *c_gpu);
    cudaFree(c_gpu);
    cudaFree(b_gpu);
    cudaFree(a_gpu);
    
    delete [] c_verify;
    delete [] c;
    delete [] b;
    delete [] a;
}
