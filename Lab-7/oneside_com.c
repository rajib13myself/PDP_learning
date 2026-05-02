#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

double work(int N)
{
    double x = 0.0;
    for (int i = 0; i < N; i++) {
        x += 0.0000001 * i;
    }
    return x;
}

int main(int argc, char *argv[])
{
    int rank, size;
    int iter = 10;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* -------------------------------------------------
       Shared window: p x iter array
    ------------------------------------------------- */
    double *win_array;
    MPI_Win win;

    win_array = (double *) malloc(size * iter * sizeof(double));

    /* initialize */
    for (int i = 0; i < size * iter; i++)
        win_array[i] = 0.0;

    /* create window */
    MPI_Win_create(win_array, size * iter * sizeof(double),
                   sizeof(double), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);

    srand(time(NULL) + rank);

    /* -------------------------------------------------
       START FENCE (RMA BEGIN)
    ------------------------------------------------- */
    MPI_Win_fence(0, win);

    for (int i = 0; i < iter; i++)
    {
        /* random workload */
        int N = 50000000 + (rank * 10000000);

        double start = MPI_Wtime();
        work(N);
        double end = MPI_Wtime();

        double time_taken = end - start;
        printf("Rank %d time = %f\n", rank, time_taken);

        /* -------------------------------------------------
           MPI_PUT: write time into ALL processes' windows
        ------------------------------------------------- */
        MPI_Put(
            &time_taken, 1, MPI_DOUBLE,
            rank,
            rank * iter + i,
            1, MPI_DOUBLE,
            win
        );
    }

    /* END FENCE */
    MPI_Win_fence(0, win);
    
    /* -------------------------------------------------
       Print result (each process has full matrix)
    ------------------------------------------------- */
    printf("\nProcess %d result matrix:\n", rank);
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < iter; c++) {
            printf("%8.5f ", win_array[r * iter + c]);
        }
        printf("\n");
    }
    
    MPI_Win_free(&win);
    free(win_array);

    MPI_Finalize();
    return 0;
}