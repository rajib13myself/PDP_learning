//Main Function for compute power project

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "matrix.h"
#include "power.h"

#define default_N 10000000
#define MAX_ITER 500
#define EPS 1e-6

int main(int argc, char **argv) {
    
    //Initialize MPI 
    MPI_Init( &argc , &argv);
    int rank; int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int local_N; 
    int begin_row;
    int N = default_N

    if (argc>1) {
        N = atoi(argv[1]);
    }
    
    //Distribution of Rows
    rows_distribute(
     N, size, rank, &local_N, &begin_row
    );

    //Generate Sparse Matrix
    CSRMatrix A;
    gen_sparse_matrix(&A, N, local_N, begin_row);

    int *cnts = malloc(size * sizeof(int));
    int *displays = malloc(size * sizeof(int));

    for(int r = 0; r < size; r++) {

        cnts[r] = N / size;

        if(r < N % size) {
            cnts[r]++;
        }

        if(r == 0) {
            displays[r] = 0;
        } else {
            displays[r] = displays[r-1] + cnts[r-1];
        }
    }
    //Allocate Vectors in Memory
    double *x = malloc(N * sizeof(double));
    //double *y = malloc(N * sizeof(double));
    double *local_y = malloc(local_N * sizeof(double));

    //Initialize Random Vector
    for (int i = 0; i<N; i++) {
        x[i] = 1.0;
    }

    normalize_vec(x, N);
    int comiter;
    double lambda_old = 0.0;
    double start_time, end_time;
    //Compute Iteration loop
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (comiter = 0; comiter < MAX_ITER; comiter++) {
        compute_local_matrix_vec_multiply(&A, x, local_y);

        //norm_vec(y, local_n);

        //double lambda_com_local = dot_prod(x, y, N);
        //double lambda_com_local = dot_prod(&x[begin_row], local_y, local_N);
        double lambda_com_local = dot_prod(x, local_y, local_N) / (dot_prod(x, x, local_N) + 1e-12);
        
        normalize_vec(local_y, local_N);

        //double error = fabs(lambda_com_local - lambda_old);
        double error = fabs(lambda_com_local - lambda_old) / (fabs(lambda_old) + 1e-12);

        //double lambda_com_global;

        MPI_Allgatherv(local_y, local_N, MPI_DOUBLE, x, cnts, displays, MPI_DOUBLE , MPI_COMM_WORLD);

        if(rank == 0) {
            printf(
                "Interation %d gets lambda = %.10f and error = %.10e\n", comiter, lambda_com_local, error
            );
        }

        if (error < EPS) {
            break;
        }
        /*for (int i = 0; i < N; i++) {
            x[i] = local_y[i];
        }*/

        lambda_old = lambda_com_local;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double total_com_time = end_time - start_time;
    //Result file creation for MPI processes
    FILE *output_file;

    if (rank == 0) {
        output_file = fopen("results.csv", "r");

        if(output_file != NULL) {
            fprintf(output_file, "%d,%d,%d,%f\n", size, N, comiter, total_com_time);
            
            fclose(output_file);
        }
    }

    if (rank == 0) {
        //printf("Finished after %d iteration by taking %0.3f times\n", comiter, total_com_time);
        printf("\n");
        printf("-----------Results are below---------------");
        printf("MPI Processes : %d\n", size);
        printf("Matrix size : %d\n", N);
        printf("Total Iteration : %d\n", comiter);
        printf("Total computation time : %f seconds\n", total_com_time);
        printf("Final Lambda is : %.10f\n", lambda_old);
        printf("-----------Results End for MPI processes :%d/n ---------------", size);
    }

    free(x);
    free(local_y);
    free_CSRMatrix(&A);
    free(cnts); free(displays);

    MPI_Finalize();
    return 0;

}