#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {

    int rank, size;

    /* ==============================
       MPI INITIALIZATION
       ============================== */
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if(rank == 0)
            printf("Usage: mpirun -n <p> ./sum nsum\n");
        MPI_Finalize();
        return 1;
    }

    /* =====================================================
       Total number of elements: n = 2^nsum
       ===================================================== */
    int nsum = atoi(argv[1]);
    long long int n = 1LL << nsum;

    /* =====================================================
       TASK 2.1 — BLOCK PARTITIONING
       ===================================================== */
    long long int first = (rank * n) / size;
    long long int last  = ((rank + 1) * n) / size - 1;
    long long int local_n = last - first + 1;

    if(rank == 0){
        printf("Parallel Summation\n");
        printf("Number of elements (n) = %lld, Processes = %d\n", n, size);
        for(int i=0;i<size;i++){
            long long int f = (i*n)/size;
            long long int l = ((i+1)*n)/size - 1;
            printf("Process %d handles indices %lld to %lld\n", i, f, l);
        }
    }

    /* =====================================================
       RANDOM NUMBER GENERATOR
       ===================================================== */
    srand(time(NULL) + rank);

    /* =====================================================
       TIMER START (for performance experiments)
       ===================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    double time_start = MPI_Wtime();

    /* =====================================================
       LOCAL COMPUTATION
       ===================================================== */
    long long int local_sum = 0;
    for(long long int i = 0; i < local_n; i++) {
        long long int num_r = rand() % 100; // random numbers 0-99
        local_sum += num_r;
    }

    printf("Process %d local sum = %lld\n", rank, local_sum);

    /* =====================================================
       TASK 2.2 — TREE STRUCTURED GLOBAL SUM (REDUCTION)
       ===================================================== */
    int step = 1;
    int recv_cnt = 0, add_cnt = 0; // TASK 2.3 counters

    while(step < size) {
        if(rank % (2*step) == 0) {
            int source = rank + step;
            if(source < size) {
                long long int received;
                MPI_Recv(&received, 1, MPI_LONG_LONG, source, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_sum += received;

                if(rank == 0) { // count for cost analysis
                    recv_cnt++;
                    add_cnt++;
                }
            }
        } else {
            int dest = rank - step;
            MPI_Send(&local_sum, 1, MPI_LONG_LONG, dest, 111, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    /* =====================================================
       TIMER END
       ===================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    double time_end = MPI_Wtime();

    if(rank == 0) {
        printf("Global sum = %lld\n", local_sum);
        printf("Execution time: %f seconds\n", time_end - time_start);

        /* =====================================================
           TASK 2.3 — COST ANALYSIS
           ===================================================== */
        printf("Process 0 performed %d receives and %d additions\n", recv_cnt, add_cnt);
        printf("Cost formula: T(p) = %d r + %d a\n", recv_cnt, add_cnt);

        /* =====================================================
           TASK 2.4 — SPEEDUP AND EFFICIENCY TABLES
           ===================================================== */
        int n_values[] = {10,20,40,80,160,320};
        int p_values[] = {1,2,4,8,16,32,64,128};
        int n_len = 6;
        int p_len = 8;

        printf("\nSpeedup Table\np\\n\t");
        for(int j=0;j<n_len;j++) printf("%d\t", n_values[j]);
        printf("\n");

        for(int i=0;i<p_len;i++){
            printf("%d\t", p_values[i]);
            for(int j=0;j<n_len;j++){
                double Ts = n_values[j];
                double Tp = Ts / p_values[i] + log2(p_values[i]);
                double S = Ts / Tp;
                printf("%.2f\t", S);
            }
            printf("\n");
        }

        printf("\nEfficiency Table\np\\n\t");
        for(int j=0;j<n_len;j++) printf("%d\t", n_values[j]);
        printf("\n");

        for(int i=0;i<p_len;i++){
            printf("%d\t", p_values[i]);
            for(int j=0;j<n_len;j++){
                double Ts = n_values[j];
                double Tp = Ts / p_values[i] + log2(p_values[i]);
                double S = Ts / Tp;
                double E = S / p_values[i];
                printf("%.2f\t", E);
            }
            printf("\n");
        }

        /* =====================================================
           TASK 2.5 — SCALABILITY
           ===================================================== */
        double k = 2.0; // Example: double processes
        double p_current = (double) size;
        double p_new = k * p_current;
        double n_new = n * (p_new * log2(p_new)) / (p_current * log2(p_current));

        printf("\nTask 2.5 — Scalability Analysis\n");
        printf("If we increase p by a factor of %.1f (p=%d -> p=%.0f),\n", k, size, p_new);
        printf("we need to increase n from %lld to %.0f to maintain the same efficiency.\n",
               n, n_new);
    }

    MPI_Finalize();
    return 0;
}