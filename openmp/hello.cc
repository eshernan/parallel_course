#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(){
// omp_set_num_threads(10);
  const int total_threads = omp_get_max_threads();
  printf("There are %d available threads.\n", total_threads); 
  fflush(stdout);

  //parallelize this part
#pragma omp parallel 
  {
    const int thread_id = omp_get_thread_num();
    printf("Hello world from thread %d\n", thread_id);
  }
}
