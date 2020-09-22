#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// This function will run concurrently.
void* print_i(void *ptr) {
    printf("a\n");
    printf("b\n");

}

int main() {
  pthread_t t1;
  int i = 1;
  int iret1 = pthread_create(&t1, NULL, print_i, NULL);
  printf("c ---> %d\n",iret1);
}


