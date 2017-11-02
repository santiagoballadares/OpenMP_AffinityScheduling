# OpenMP_AffinityScheduling
Simple implementation of the affinity scheduling algorithm, a loop scheduling algorithm for Open_MP, using work stealing to keep threads busy whenever possible with C and Open_MP.

Compile with: 

      > gcc -fopenmp omp_affinity_scheduling.c -o omp_affinity_scheduling

Run with:

      
      > set OMP_NUM_THREADS=8            # explicitly set 8 threads by the OMP_NUM_THREADS environment variable
      > omp_affinity_scheduling.exe
      

Setup for windows:
1. Install MinGW

For more details see:
- http://www.math.ucla.edu/~wotaoyin/windows_coding.html
