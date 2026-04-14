#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4   // vector size (columns of A)
#define K 3   // rows per process

int main(int argc, char *argv[]) {

  int rank, size;
  MPI_Status status;
  double t_begin, t_end;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int m = K * size;   // total rows

  double A[K][N];
  double x[N];
  double y[K];
  double rw_local[N];
  double rw_global[N];

  int i, j;
  
  t_begin = MPI_Wtime();
  /* ----------------------------
     Generate matrix A locally
     ---------------------------- */
  srand(rank + 1);

  for (i = 0; i < K; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = (double)(rand() % 5 + 1);
    }
  }

  /* ----------------------------
     Generate vector x
     ---------------------------- */
  for (j = 0; j < N; j++) {
    x[j] = 1.0;  
  }

 
  for (i = 0; i < K; i++) {
    y[i] = 0.0;
    for (j = 0; j < N; j++) {
      y[i] += A[i][j] * x[j];
    }
  }

 
  for (j = 0; j < N; j++) {
    rw_local[j] = 0.0;
    for (i = 0; i < K; i++) {
      rw_local[j] += A[i][j] * y[i];
    }
  }

  
  if (rank == 0) {

   
    for (j = 0; j < N; j++) {
      rw_global[j] = rw_local[j];
    }

    
    for (i = 1; i < size; i++) {
      double temp[N];

      MPI_Recv(temp, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

      for (j = 0; j < N; j++) {
        rw_global[j] += temp[j];
      }
    }

    /* print result */
    printf("Final rw_global = A^T A x:\n");
    for (j = 0; j < N; j++) {
      printf("%f ", rw_global[j]);
    }
    printf("\n");

  } else {

    /* send local contribution */
    MPI_Send(rw_local, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  t_end = MPI_Wtime();
  printf("Elapsed time : %1.2f\n", t_end-t_begin);
  MPI_Finalize();
  return 0;
}