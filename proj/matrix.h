#ifndef MATRIX_H
#define MATRIX_H

#include <mpi.h>

typedef struct {
    int numrows;
    int numcols;
    int nnz;    //Number of non zero values

    double *values;
    int *col_index;
    int *row_ptr;

}CSRMatrix;

void gen_sparse_matrix(CSRMatrix *A, int N, int local_N, int begin_row);

void compute_local_matrix_vec_multiply(
        CSRMatrix *A, 
        double *x,
        double *y
);

void free_CSRMatrix(CSRMatrix *A);

#endif
