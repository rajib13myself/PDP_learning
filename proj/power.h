#ifndef POWER_H
#define POWER_H

#include <mpi.h>

double vec_norm(
    double *x,
    int local_n
);

double dot_prod(
    double *x,
    double *y,
    int local_n
);

void normalize_vec(
    double *x,
    int local_n
);

/*Distribution of row*/
void rows_distribute(
    int N, int size, int rank, int *local_N, int *begin_row
);

#endif