
// File: unsafe.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

char s1[] = "abcdefg";
char s2[] = "abc";

char* c;
void last_letter(char* a, int i) {
  printf("last letter a is %s and i is %d\n", a, i);
  sleep(i);
  c = NULL;   // comment out a rerun, what is new?
  sleep(i);
  c = a;
  sleep(i);
  while (*(c)) {
    c++;
  }
  printf("%c\n", *(c-1));
}


// This function will run concurrently.

void* aa(void *ptr) {
  last_letter(s2, 2);
}

int main() {
  pthread_t t1;
  int iret1 = pthread_create(&t1, NULL, aa, NULL);
  last_letter(s1, 5);
  sleep(10);
  printf("Ended nicely this time\n");
  exit(0);
 //never reached when c = NULL is not commented out.
}

