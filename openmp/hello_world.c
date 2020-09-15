#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
int main()
{
 #pragma omp parallel
 {
 int ID = omp_get_thread_num();
 printf("Hello  the current thread id is (%d)\n", ID);
 }
}
