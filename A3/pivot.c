#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "pivot.h"

/* Comparison for qsort */
int compare(const void *v1, const void *v2) {
    int i1 = *(int*)v1;
    int i2 = *(int*)v2;
    return i1 - i2;
}

/* Find first index where element > val */
int get_larger_index(int *elements, int n, int val) {
    int i;
    for (i = 0; i < n; i++) {
        if (elements[i] > val) return i;
    }
    return n;
}

/* Get median from sorted array */
int get_median(int *elements, int n) {
    if (n == 0) return 0;
    return elements[n / 2];
}

/* Dispatcher */
int select_pivot(int pivot_strategy, int *elements, int n, MPI_Comm comm) {
    switch (pivot_strategy) {
        case MEDIAN_ROOT:
            return select_pivot_median_root(elements, n, comm);
        case MEAN_MEDIAN:
            return select_pivot_mean_median(elements, n, comm);
        case MEDIAN_MEDIAN:
            return select_pivot_median_median(elements, n, comm);
        default:
            return select_pivot_median_root(elements, n, comm);
    }
}

/* Strategy 1: median on root */
int select_pivot_median_root(int *elements, int n, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int local_median = get_median(elements, n);
    int pivot;

    if (rank == ROOT) {
        pivot = local_median;
    }

    MPI_Bcast(&pivot, 1, MPI_INT, ROOT, comm);

    return get_larger_index(elements, n, pivot);
}

/* Strategy 2: mean of medians */
int select_pivot_mean_median(int *elements, int n, MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);

    int local_median = get_median(elements, n);
    int sum;

    MPI_Allreduce(&local_median, &sum, 1, MPI_INT, MPI_SUM, comm);

    int pivot = sum / size;

    return get_larger_index(elements, n, pivot);
}

/* Strategy 3: median of medians */
int select_pivot_median_median(int *elements, int n, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_median = get_median(elements, n);

    int *all_medians = NULL;

    if (rank == ROOT) {
        all_medians = (int*) malloc(size * sizeof(int));
    }

    MPI_Gather(&local_median, 1, MPI_INT,
               all_medians, 1, MPI_INT,
               ROOT, comm);

    int pivot;

    if (rank == ROOT) {
        qsort(all_medians, size, sizeof(int), compare);
        pivot = all_medians[size / 2];
        free(all_medians);
    }

    MPI_Bcast(&pivot, 1, MPI_INT, ROOT, comm);

    return get_larger_index(elements, n, pivot);
}

/* Optional: smallest element on root */
int select_pivot_smallest_root(int *elements, int n, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int pivot;

    if (rank == ROOT) {
        pivot = elements[0];
    }

    MPI_Bcast(&pivot, 1, MPI_INT, ROOT, comm);

    return get_larger_index(elements, n, pivot);
}