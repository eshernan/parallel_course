#include "mmult.h"

__global__ void set_matrix(int m, int n, int k, float * a, float * b)
{
	int globx = blockIdx.x * blockDim.x + threadIdx.x;
	int globy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int l;

	for (l = 0; l < n; l++) {
		a[globx * k + globy] += 1.0f;
		b[globx * k + globy] += 1.0f;
	}
}

void init_matrix(int m, int n, int k, float * a, float * b)
{
	dim3 dim_Grid(m/BLOCK_SIZE, k/BLOCK_SIZE);
	dim3 dim_Block(BLOCK_SIZE, BLOCK_SIZE);
	set_matrix<<<dim_Grid, dim_Block>>>(m, n, k, a, b);
}
