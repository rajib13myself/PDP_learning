#ifndef R_MPI_UTILS_H
#define R_MPI_UTILS_H

void dist_rows(
    int n,
    int size,
    int rank,
    int *local_n,
    int *start_row
);

#endif