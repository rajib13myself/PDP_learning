#include <stdio.h>
#include <stdlib.h>
#include "power.h"
#include <math.h>

double dot_prod(double *x, double *y, int local_n) {
    double local_sum = 0.0;

    for(int i = 0; i < local_n; i++) {
        local_sum += x[i] * y[i];
    }

    double global_sum = 0.0;

    MPI_Allreduce( &local_sum , &global_sum , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);

    return global_sum;
}

double vec_norm(double *x, int local_n) {
    return sqrt(dot_prod(x, x, local_n));
}

void normalize_vec(double *x, int local_n) {
    double norm_v = vec_norm(x, local_n);
    
    for(int i = 0; i < local_n; i++) {
        x[i] /= norm_v;
    }
}

/* Implementation of row distribution */
void rows_distribute(int N, int size, int rank, int *local_N, int *begin_row) {
    int initial_rows = N / size;
    int rest_rows = N % size;

    if(rank < rest_rows) {
        *local_N = initial_rows + 1;
        *begin_row = rank * (*local_N);
    } else {
        *local_N = initial_rows;
        *begin_row = rest_rows * (initial_rows + 1) + (rank - rest_rows) * initial_rows;
    }

}