#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char **argv) 
{
  int ierr;
  char hostname[HOST_NAME_MAX + 1];
  gethostname(hostname, HOST_NAME_MAX + 1);

  ierr = MPI_Init(&argc, &argv);
  printf("Hello world   "); 
  printf("hostname: %s\n", hostname);
  ierr = MPI_Finalize();
}
