#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

int main(void)
{
    int reps = 1000;
    int N = 20;
    int a= 0;

#pragma omp parallel
    { // not a parallel for: just a bunch of reps
        for (int j = 0; j < reps; j++)
        {
#pragma omp for schedule(static, 1)
            for (int i = 0; i < N; i++)
            {
#pragma omp atomic
                a++;
            }
        }
    }
}