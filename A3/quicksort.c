#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "quicksort.h"
#include "pivot.h"

/*
Global parallel quicksort section
*/
int global_sort(int **elements, int n, MPI_Comm comm, int pivot_strategy) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) return n;

    // 1. choose pivot
    int split = select_pivot(pivot_strategy, *elements, n, comm);

    int *low = *elements;
    int low_n = split;

    int *high = *elements + split;
    int high_n = n - split;

    int half = size / 2;
    int partner = (rank < half) ? rank + half : rank - half;
    int color = (rank < half) ? 0 : 1;

    int send_n = (color == 0) ? high_n : low_n;
    int recv_n;

    MPI_Sendrecv(&send_n, 1, MPI_INT,
                 partner, 0,
                 &recv_n, 1, MPI_INT,
                 partner, 0,
                 comm, MPI_STATUS_IGNORE);

    int *recv_buf = malloc(recv_n * sizeof(int));

    MPI_Sendrecv((color == 0 ? high : low), send_n, MPI_INT,
                 partner, 1,
                 recv_buf, recv_n, MPI_INT,
                 partner, 1,
                 comm, MPI_STATUS_IGNORE);

    int *keep = (color == 0) ? low : high;
    int keep_n = (color == 0) ? low_n : high_n;

    int new_n = keep_n + recv_n;
    int *new_elements = malloc(new_n * sizeof(int));

    merge_ascending(keep, keep_n, recv_buf, recv_n, new_elements);

    free(recv_buf);
    free(*elements);

    *elements = new_elements;

    MPI_Comm new_comm;
    MPI_Comm_split(comm, color, rank, &new_comm);

    int final_n = global_sort(elements, new_n, new_comm, pivot_strategy);

    MPI_Comm_free(&new_comm);

    return final_n;
}


//Helper function merge ascending
void merge_ascending(int *v1, int n1, int *v2, int n2, int *result) {
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2) {
        if (v1[i] <= v2[j]) {
            result[k++] = v1[i++];
        } else {
            result[k++] = v2[j++];
        }
    }

    while (i < n1) {
        result[k++] = v1[i++];
    }

    while (j < n2) {
        result[k++] = v2[j++];
    }
}

//Helper function Input file read
int read_input(char *file_name, int **elements) {
    FILE *fp = fopen(file_name, "r");
    if (!fp) {
        printf("Error opening input file\n");
        return -1;
    }

    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        printf("Error reading size\n");
        fclose(fp);
        return -1;
    }

    *elements = (int*) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%d", &((*elements)[i])) != 1) {
            printf("Error reading element %d\n", i);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return n;
}

//Helper function distribute data from root

int distribute_from_root(int *all_elements, int n, int **my_elements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendcounts = NULL;
    int *displs = NULL;

    int base = n / size;
    int rem = n % size;

    if (rank == ROOT) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        int offset = 0;

        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    int local_n;

    MPI_Scatter(sendcounts, 1, MPI_INT,
                &local_n, 1, MPI_INT,
                ROOT, MPI_COMM_WORLD);

    *my_elements = malloc(local_n * sizeof(int));

    MPI_Scatterv(all_elements, sendcounts, displs, MPI_INT,
                 *my_elements, local_n, MPI_INT,
                 ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        free(sendcounts);
        free(displs);
    }

    return local_n;
}

//Helper function Gather results to root
void gather_on_root(int *all_elements, int *my_elements, int local_n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == ROOT) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    MPI_Gather(&local_n, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    MPI_Gatherv(my_elements, local_n, MPI_INT,
                all_elements, recvcounts, displs, MPI_INT,
                ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        free(recvcounts);
        free(displs);
    }
}

//Helper function check and output

int sorted_ascending(int *elements, int n) {
    for (int i = 1; i < n; i++) {
        if (elements[i-1] > elements[i]) {
            return 0;
        }
    }
    return 1;
}

int check_and_print(int *elements, int n, char *file_name) {
    FILE *fp = fopen(file_name, "w");
    if (!fp) {
        printf("Error opening output file\n");
        return -2;
    }

    if (!sorted_ascending(elements, n)) {
        printf("ERROR: array not sorted correctly\n");
    }

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d ", elements[i]);
    }

    fclose(fp);
    return 0;
}


//Main program for execution

int main(int argc, char **argv) {
    MPI_Init( &argc , &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //check arguments for perfect I/O
    if (argc != 4) {
        if (rank == ROOT) {
            printf("Usage: %s input_file output_file pivot_strategy\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    int pivot_strategy = atoi(argv[3]);
    
    int *all_elements = NULL;
    int n = 0;

    //Read input file while at root
    if (rank == ROOT) {
        n = read_input(input_file, &all_elements);
    }

    //Broadcast size
    MPI_Bcast( &n , 1 , MPI_INT , ROOT , MPI_COMM_WORLD);

    // Distributed data
    int *local_elements;
    int local_n = distribute_from_root(all_elements, n, &local_elements);

    //Local sort
    qsort(local_elements, local_n, sizeof(int), compare);

    //Start timeing for sorting
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    //Global sort (step 3 & 4)
    local_n = global_sort(&local_elements, local_n, MPI_COMM_WORLD, pivot_strategy);

    MPI_Barrier( MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    //Collect result as per sorting
    if (rank == ROOT) {
        free(all_elements);
        all_elements = (int*) malloc(n * sizeof(int));
    }

    gather_on_root(all_elements, local_elements, local_n);

    //Print times and outputs in output_file
    if (rank == ROOT) {
        printf("%f\n", end_time - start_time);

        //Verify and write output
        check_and_print(all_elements, n, output_file);
    }

    free(local_elements);
    if (rank == ROOT) {
        free(all_elements);
    }
    
    MPI_Finalize();
    return 0;
    

}