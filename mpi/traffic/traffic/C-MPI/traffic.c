#include <stdlib.h>
#include <stdio.h>

#include "traffic.h"

#define NCELL 100000

int main(int argc, char **argv)
{
  int *oldroad, *newroad, *bigroad;

  int i, iter, nmove, ncars;
  int maxiter, printfreq; 

  float density; 

  double tstart, tstop;

  int procid, nproc, nlocal;

  maxiter = 200000000/NCELL; 
  printfreq = maxiter/10; 

  // Set target density of cars

  density = 0.52;

  // Start message passing system

  mpstart(&nproc, &procid);

  if (procid == 0)
    {
      printf("Length of road is %d\n", NCELL);
      printf("Number of iterations is %d \n", maxiter);
      printf("Target density of cars is %f \n", density);
      printf("Running on %d processes\n", nproc);
    }

  nlocal = NCELL/nproc;

  oldroad = (int *) malloc((nlocal+2)*sizeof(int));
  newroad = (int *) malloc((nlocal+2)*sizeof(int));

  if (procid == 0)
    {
      bigroad = (int *) malloc(NCELL*sizeof(int));

      // Initialise road accordingly using random number generator

      printf("Initialising road ...\n");
  
      ncars = initroad(bigroad, NCELL, density, SEED);

      printf("...done\n");
      printf("Actual density is %f\n", (float) ncars / (float) NCELL);
      printf("Scattering data ...\n");
    }

  mpscatter(bigroad, &oldroad[1], nlocal);

  if (procid == 0)
    {
      printf("... done\n\n");
    }

  tstart = gettime();

  for (iter=1; iter<=maxiter; iter++)
    { 
      mpupdatebcs(oldroad, nlocal, procid, nproc);

      // Apply CA rules to all cells

      nmove = updateroad(newroad, oldroad, nlocal);

      // Globally sum the value

      mpgsum(&nmove);

      // Copy new to old array

      for (i=1; i<=nlocal; i++)
	{
	  oldroad[i] = newroad[i]; 
	}

      if (iter%printfreq == 0)
	{
	  if (procid == 0)
	    {
	      printf("At iteration %d average velocity is %f \n",
		     iter, (float) nmove / (float) ncars);
	    }
	} 
    }

  tstop = gettime();

  free(oldroad);
  free(newroad);

  if (procid == 0)
    {
      free(bigroad);

      printf("\nFinished\n");
      printf("\nTime taken was  %f seconds\n", tstop-tstart);
      printf("Update rate was %f MCOPs\n\n", \
	     1.e-6*((double) NCELL)*((double) maxiter)/(tstop-tstart));
    }

  mpstop();
}
