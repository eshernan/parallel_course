#ifndef MMULT_H
#define MMULT_H

#define BLOCK_SIZE 16

// Matrix multiplication functions
// c = a * b
// a: m x n
// b: n x k
// c: m x k
void mmult(int m, int n, int k, const float * a, const float * b, float * c);
void mmult_gpu(int m, int n, int k, const float * a, const float * b, float * c);
void gpu_blas_mmul(int m, int n, int k, const float * a, const float * b, float * c);

void init_matrix(int m, int n, int k, float * a, float * b);
#endif
