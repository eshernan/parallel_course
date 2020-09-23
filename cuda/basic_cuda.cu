#include <stdio.h>
//code executed on devide 
__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

//code execute on host
int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
//reserve memory on host
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

//reserve memory on device
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
//inicialization on host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
//copy data to device 
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  //invoke kernel on device
  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

// copy results to hosts
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

//free memory on device and host
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
