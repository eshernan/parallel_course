#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ejhernandezb@udistrital.edu.co    # Where to send mail	
#SBATCH --ntasks=3
#SBATCH --nodes=3                   # Run on a single CPU
#SBATCH --time=00:35:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
mpirun   /home/ehernandez/parallel/parallel_course/mpi/ring 
