#include <stdio.h>
#include <math.h>

#define N 729
#define reps 10
#include <omp.h>

// Typedef enum for boolean values
typedef enum { false, true } bool;

double a[N][N], b[N][N], c[N];
int jmax[N];

void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);

int main(int argc, char *argv[]) {
  double start1, start2, end1, end2;
  int r;

  init1();

  start1 = omp_get_wtime();

  for (r=0; r<reps; ++r) {
    runloop(1);
  }

  end1  = omp_get_wtime();

  valid1();

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1));

  init2();

  start2 = omp_get_wtime();

  for (r=0; r<reps; ++r) {
    runloop(2);
  }

  end2  = omp_get_wtime();

  valid2();

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2));
}

void init1(void) {
  int i, j;

  for (i=0; i<N; ++i) {
    for (j=0; j<N; ++j) {
      a[i][j] = 0.0;
      b[i][j] = 3.142*(i+j);
    }
  }
}

void init2(void) {
  int i,j, expr;

  for (i=0; i<N; ++i) {
    expr =  i%( 3*(i/30) + 1);
    
    if ( expr == 0) {
      jmax[i] = N;
    } else {
      jmax[i] = 1;
    }
    
    c[i] = 0.0;
  }

  for (i=0; i<N; ++i) {
    for (j=0; j<N; ++j) {
      b[i][j] = (double) (i*j+1) / (double) (N*N);
    }
  }
}

void runloop(int loopid) {
  // Shared arrays for storing iterations left and which threads are busy/idle
  int iters_left[omp_get_max_threads()];
  int idle_threads[omp_get_max_threads()];

#pragma omp parallel default(none) shared(loopid, iters_left, idle_threads)
{
    // Calculate local set of iterations per thread: low and high boundaries
    int myid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    int ipt = (int) ceil((double)N/(double)nthreads);
    int lo = myid*ipt;
    int hi = (myid+1)*ipt;

    if (hi > N) {
      hi = N;
    }
    
    // Initialize shared arrays
    iters_left[myid] = hi - lo;
    idle_threads[myid] = 0;

// A barrier to make sure every thread has stored its own value for both arrays
#pragma omp barrier

    // Variables to keep track of the thread id and to control the loop calls
    bool itersRemaining = true;
    int loadedThreadId;

    // Let's start assuming there are iterations left
    while (itersRemaining) {
#pragma omp critical
{
      // First, check thread own iterations left
      if (iters_left[myid] > 0) {
        itersRemaining = true;
        loadedThreadId = myid;
      } else {
        // Else, let's find the thread with the most iterations left
        int i, loadedThreadVal = 0;
        for (i=0; i<nthreads; ++i) {
          if (idle_threads[i] == 0 && iters_left[i] > loadedThreadVal) {
            loadedThreadId = i;
            loadedThreadVal = iters_left[i];
          }
        }

        // Control that there are iterations left
        if (loadedThreadVal > 0) {
          itersRemaining = true;
        } else {
          itersRemaining = false;
        }
      }
}

      // If there are iterations left, calculate the inner set of iterations to be processed
      if (itersRemaining) {
        int inner_chunk, inner_lo, inner_hi;
#pragma omp critical
{
        // It is not known if "loadedThreadId" is the current thread id or other thread id. 
        // If it is other thread id and the other thread is the last one, its local hi could 
        // not be the same as the local hi of other threads, so we calculate the local hi.
        int h = (loadedThreadId + 1) * ipt;
        if (h > N) h = N;

        inner_chunk = (int) ceil((1.0 / (double)nthreads) * (double)iters_left[loadedThreadId]);
        inner_lo = h - iters_left[loadedThreadId];
        inner_hi = inner_lo + inner_chunk;

        // Update information in shared arrays
        idle_threads[loadedThreadId] = 1;
        iters_left[loadedThreadId] -= inner_chunk;
}

        // Call loop functions to process inner set of iterations
        switch (loopid) {
          case 1:
            loop1chunk(inner_lo, inner_hi);
          break;
          case 2:
            loop2chunk(inner_lo, inner_hi);
          break;
        }
#pragma omp critical
{
        // Finally, set thread flag back to idle.
        idle_threads[loadedThreadId] = 0;
}
      }
    }
  }
}

void loop1chunk(int lo, int hi) {
  int i, j;
  
  for (i=lo; i<hi; ++i) {
    for (j=N-1; j>i; --j) {
      a[i][j] += cos(b[i][j]);
    }
  }
}

void loop2chunk(int lo, int hi) {
  int i, j, k;
  double rN2;

  rN2 = 1.0 / (double) (N*N);

  for (i=lo; i<hi; ++i) {
    for (j=0; j < jmax[i]; ++j) {
      for (k=0; k<j; ++k) {
        c[i] += (k+1) * log (b[i][j]) * rN2;
      }
    }
  }
}

void valid1(void) {
  int i, j;
  double suma;
  
  suma = 0.0;
  for (i=0; i<N; ++i) {
    for (j=0; j<N; ++j) {
      suma += a[i][j];
    }
  }
  
  printf("Loop 1 check: Sum of a is %lf\n", suma);
} 

void valid2(void) {
  int i;
  double sumc;
  
  sumc = 0.0; 
  for (i=0; i<N; ++i) {
    sumc += c[i];
  }
  
  printf("Loop 2 check: Sum of c is %f\n", sumc);
}
