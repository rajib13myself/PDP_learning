#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

//Implementation of Matrix Vector Product of y_i = summation of j for a_ij x_j
void gen_sparse_matrix(CSRMatrix *A, int N, int local_N, int begin_row) {
    //int i;

    A->numrows = local_N;
    A->numcols = N;

    //Generate simple matrix
    //A->nnz = 3 * n - 2;
    A->nnz = 3 * local_N;

    //Allocate memory for all values
    A->values = malloc(A->nnz * sizeof(double));
    A->col_index = malloc(A->nnz * sizeof(int));
    A->row_ptr = malloc((local_N + 1) * sizeof(int));

    int index = 0;

    for(int i = 0; i < local_N; i++) {
        int global_row = begin_row + i;
        A->row_ptr[i] = index;

        if(global_row > 0) {
            A->values[index] = -1.0;
            A->col_index[index] = global_row - 1;
            index++;
        }
        
        A->values[index] = 2.0;
        A->col_index[index] = global_row;
        index++;

        if(global_row < N - 1) {
            A->values[index] = -1.0;
            A->col_index[index] = global_row + 1;
            index++;
        }

    }
    A->row_ptr[local_N] = index;
    A->nnz = index;
}

void compute_local_matrix_vec_multiply(CSRMatrix *A, double *x, double *local_y) {
    int rows; int j;
    for(rows = 0; rows < A->numrows; rows++) {
        double sum = 0.0;

        for(j = A->row_ptr[rows]; j < A->row_ptr[rows + 1]; j++) {
            int cols = A->col_index[j];
            sum += A->values[j] * x[cols];
        }
        local_y[rows] = sum;
    }
}

void free_CSRMatrix(CSRMatrix *A) {
    free(A->values);
    free(A->col_index);
    free(A->row_ptr);
}

