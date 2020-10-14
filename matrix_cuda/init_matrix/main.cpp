#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include "mmult.h"

int main()
{
    const int m = BLOCK_SIZE * 12, n = BLOCK_SIZE * 18, k = BLOCK_SIZE * 24;
    float * a, * b, * c,  * c_verify;

    c_verify = new float[m * k];

    cudaMallocManaged((void **) &a, m * n * sizeof (float));
    cudaMallocManaged((void **) &b, n * k * sizeof (float));
    cudaMallocManaged((void **) &c, m * k * sizeof (float));

    init_matrix(m, n,k, a, b);

//    cudaMemcpy(a_gpu, a, m * n * sizeof *a_gpu, cudaMemcpyHostToDevice);
//    cudaMemcpy(b_gpu, b, n * k * sizeof *b_gpu, cudaMemcpyHostToDevice);

    float sTime;
    clock_t start, finish;

    start = clock();
    mmult(m, n, k, a, b, c);
    finish = clock();

    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on CPU: %lf sec", sTime);

    start = clock();
    printf("\n Starting to compute on GPU \n");
    mmult_gpu(m, n, k, a, b, c);
//    cudaMemcpy(c_verify, c_gpu, m * k * sizeof *c_gpu, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    finish = clock();

    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on GPU: %lf sec",sTime);

    float difference = 0;

    for (int i = 0; i < m * k; ++i)
        difference += (c[i] - c_verify[i]) * (c[i] - c_verify[i]);

    if (difference < 1e-5f)
        std::cout << "\n Test passed.\n";
    else
        std::cout << "\n Test failed (diff = " << difference << ").\n";

//    cudaMemset(c_gpu, 0, m * k * sizeof *c_gpu);
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);

    delete [] c_verify;
}
