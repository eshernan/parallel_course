#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "mmult.h"
#include <cublas_v2.h>

int main()
{
//   cudaEvent_t start, stop;
// float time;

    const int m = BLOCK_SIZE * 12, n = BLOCK_SIZE * 18, k = BLOCK_SIZE * 24;
    float * a, * b, * c, * a_gpu, * b_gpu, * c_gpu, * c_verify;

    a = new float[m * n];
    b = new float[n * k];
    c = new float[m * k];

    c_verify = new float[m * k];


    cudaMalloc((void **) &a_gpu, m * n * sizeof *a_gpu);
    cudaMalloc((void **) &b_gpu, n * k * sizeof *b_gpu);
    cudaMalloc((void **) &c_gpu, m * k * sizeof *c_gpu);

    //cudaMallocManaged((void **) &a, m * n * sizeof (float));
    //cudaMallocManaged((void **) &b, n * k * sizeof (float));
    //  cudaMallocManaged((void **) &c, m * k * sizeof (float));

    //init_matrix(m, n, k, a_gpu, b_gpu);
    for (int i = 0; i < m * n; ++i)
        a[i] = float(rand()) / RAND_MAX;

    for (int i = 0; i < n * k; ++i)
        b[i] = float(rand()) / RAND_MAX;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    //
    // cudaEventRecord( start, 0 );

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
    printf("\n Starting to compute on GPU \n");

    gpu_blas_mmul(m, n, k, a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c_verify, c_gpu, m * k * sizeof *c_gpu, cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();
    finish = clock();

    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on GPU: %lf sec",sTime);

    // cudaEventRecord( stop, 0 );
    // cudaEventSynchronize( stop );

    float difference = 0;

    for (int i = 0; i < m * k; ++i)
        difference += (c[i] - c_verify[i]) * (c[i] - c_verify[i]);

    if (difference < 1e-5f)
        std::cout << "\n Test passed.\n";
    else
        std::cout << "\n Test failed (diff = " << difference << ").\n";

//    cudaMemset(c_gpu, 0, m * k * sizeof *c_gpu);

    // cudaEventElapsedTime( &time, start, stop );
    // cudaEventDestroy( start );
    // cudaEventDestroy( stop );

    cudaMemset(c_gpu, 0, m * k * sizeof *c_gpu);
    cudaFree(c_gpu);
    cudaFree(b_gpu);
    cudaFree(a_gpu);

    delete [] c_verify;
    delete [] c;
    delete [] b;
    delete [] a;
}
