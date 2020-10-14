#include "mmult.h"

__global__ void mmult_kernel(int m, int n, int k, const float * a, const float * b, float * c)
{
	int globx = blockIdx.x * blockDim.x + threadIdx.x;
	int globy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int l;

	for (l = 0; l < n; l++)
		c[globx * k + globy] += a[globx * n + l] * b[l * k + globy];
}


void mmult_gpu(int m, int n, int k, const float * a, const float * b, float * c)
{
	dim3 dim_Grid(m/BLOCK_SIZE, k/BLOCK_SIZE);
	dim3 dim_Block(BLOCK_SIZE,BLOCK_SIZE);
	mmult_kernel<<<dim_Grid, dim_Block>>>(m, n, k, a, b, c);
}
