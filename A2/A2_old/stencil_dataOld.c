#include "stencil.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            printf("Usage: stencil input_file output_file number_of_applications\n");
        MPI_Finalize();
        return 1;
    }

    char *input_name  = argv[1];
    char *output_name = argv[2];
    int num_steps     = atoi(argv[3]);

    double *input = NULL;
    int num_values;

    // =========================
    // READ INPUT
    // =========================
    if (rank == 0) {
        num_values = read_input(input_name, &input);
        if (num_values < 0) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // =========================
    // STENCIL SETUP
    // =========================
    double h = 2.0 * PI / num_values;
    const int STENCIL_WIDTH = 5;
    const int EXTENT = STENCIL_WIDTH / 2;

    const double STENCIL[5] = {
        1.0/(12*h), -8.0/(12*h), 0.0,
        8.0/(12*h), -1.0/(12*h)
    };

    int local_n = num_values / size;

    // =========================
    // BUFFERS
    // =========================
    double *local_input  = malloc((local_n + 2*EXTENT) * sizeof(double));
    double *local_output = malloc((local_n + 2*EXTENT) * sizeof(double));

    double *left_ghost  = malloc(EXTENT * sizeof(double));
    double *right_ghost = malloc(EXTENT * sizeof(double));

    if (!local_input || !local_output || !left_ghost || !right_ghost) {
        perror("malloc failed");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // =========================
    // SCATTER INPUT
    // =========================
    MPI_Scatter(input, local_n, MPI_DOUBLE,
                &local_input[EXTENT], local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    // =========================
    // INITIAL HALO EXCHANGE (CRITICAL)
    // =========================
    MPI_Sendrecv(&local_input[EXTENT], EXTENT, MPI_DOUBLE,
                 left, 0,
                 right_ghost, EXTENT, MPI_DOUBLE,
                 right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&local_input[EXTENT + local_n - EXTENT], EXTENT, MPI_DOUBLE,
                 right, 1,
                 left_ghost, EXTENT, MPI_DOUBLE,
                 left, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < EXTENT; i++) {
        local_input[i] = left_ghost[i];
        local_input[EXTENT + local_n + i] = right_ghost[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // =========================
    // TIME LOOP
    // =========================
    for (int s = 0; s < num_steps; s++) {

        // =========================
        // HALO EXCHANGE
        // =========================
        MPI_Sendrecv(&local_input[EXTENT], EXTENT, MPI_DOUBLE,
                    left, 0,
                    left_ghost, EXTENT, MPI_DOUBLE,
                    right, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&local_input[EXTENT + local_n - EXTENT], EXTENT, MPI_DOUBLE,
                    right, 1,
                    right_ghost, EXTENT, MPI_DOUBLE,
                    left, 1,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // =========================
        // STENCIL COMPUTATION
        // =========================
        for (int i = EXTENT; i < EXTENT + local_n; i++) {

            double result = 0.0;

            for (int j = 0; j < STENCIL_WIDTH; j++) {

                int idx = i - EXTENT + j;

                if (idx < EXTENT) {
                    result += STENCIL[j] * left_ghost[idx];
                }
                else if (idx >= EXTENT + local_n) {
                    result += STENCIL[j] * right_ghost[idx - EXTENT - local_n];
                }
                else {
                    result += STENCIL[j] * local_input[idx];
                }
            }

            local_output[i] = result;
        }

        // swap
        double *tmp = local_input;
        local_input = local_output;
        local_output = tmp;
    }

    // =========================
    // TIMING
    // =========================
    double elapsed = MPI_Wtime() - start_time;

    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================
    // GATHER
    // =========================
    double *output = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        output = malloc(num_values * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            recvcounts[i] = local_n;
            displs[i] = i * local_n;
        }
    }

    MPI_Gatherv(&local_input[EXTENT], local_n, MPI_DOUBLE,
                output, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // =========================
    // OUTPUT
    // =========================
    if (rank == 0) {

        printf("%f\n", max_time);
        write_output(output_name, output, num_values);

        free(input);
        free(output);
        free(recvcounts);
        free(displs);
    }

    // =========================
    // CLEANUP
    // =========================
    free(local_input);
    free(local_output);
    free(left_ghost);
    free(right_ghost);

    MPI_Finalize();
    return 0;
}


int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i=0; i<num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)){
		perror("Warning: couldn't close input file");
	}
	return num_values;
}


int write_output(char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}
