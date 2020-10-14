#include "mmult.h"
#include <curand.h>

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

// // Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
// void init_matrix(int m, int n, int k, float * a, float * b) {
//    // Create a pseudo-random number generator
//    curandGenerator_t prng;
//    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//
//    // Set the seed for the random number generator using the system clock
//    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//
//    // Fill the array with random numbers on the device
//    curandGenerateUniform(prng, a, n * m);
// 	 curandGenerateUniform(prng, b, n * m);
// }


// void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
//      for(int i = 0; i < nr_rows_A; ++i){
//          for(int j = 0; j < nr_cols_A; ++j){
//              std::cout << A[j * nr_rows_A + i] << " ";
//          }
//          std::cout << std::endl;
//      }
//      std::cout << std::endl;
// }
