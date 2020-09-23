#include "stdio.h"

int main(){

	int n = 16;
	
	// host and device memory pointers
	int *h_a;
	int *d_a;
	
	// allocate host memory
	h_a = (int*)malloc(n * sizeof(int));
	
	// allocate device memory
	cudaMalloc((void**)&d_a, n * sizeof(int));		
	
	// set device memory to all zero's
	cudaMemset(d_a, 0, n * sizeof(int));
	
	// copy device memory back to host
	cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);	
	
	// print host memory
	for (int i = 0; i < n; i++){
		printf("%d ", h_a[i]);
	}
	printf("\n");
	
	// free buffers
	free(h_a);
	cudaFree(d_a);
	
	return 0;
	
}
