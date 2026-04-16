#include "stencil.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: stencil input_file output_file number_of_applications\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *input_name = argv[1];
    char *output_name = argv[2];
    int num_steps = atoi(argv[3]);

    int num_values = 0;
    double *input = NULL;

    // -------------------------
    // Rank 0 reads input only
    // -------------------------
    if (rank == 0) {
        num_values = read_input(input_name, &input);
        if (num_values < 0) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    // Broadcast number of values
    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = num_values / size;

    // -------------------------
    // Allocate local arrays with ghost cells
    // [0,1] = left ghost
    // [2..local_n+1] = real data
    // [local_n+2, local_n+3] = right ghost
    // -------------------------
    double *local_in  = calloc(local_n + 4, sizeof(double));
    double *local_out = calloc(local_n + 4, sizeof(double));

    // -------------------------
    // Scatter data (only rank 0 uses input buffer)
    // -------------------------
    MPI_Scatter(input, local_n, MPI_DOUBLE,
                &local_in[2], local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    printf("rank %d first local value = %f\n", rank, local_in[2]);
    free(input);

    // Stencil setup
    double h = 2.0 * PI / num_values;
    const int W = 5;
    const int EXT = W / 2;

    const double stencil[5] = {
        1.0 / (12 * h),
        -8.0 / (12 * h),
        0.0,
        8.0 / (12 * h),
        -1.0 / (12 * h)
    };

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // -------------------------
    // Stencil iterations
    // -------------------------
    for (int s = 0; s < num_steps; s++) {

        // Neighbour exchange
        MPI_Sendrecv(&local_in[2], 2, MPI_DOUBLE, left, 0,
             &local_in[local_n + 2], 2, MPI_DOUBLE, right, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&local_in[local_n], 2, MPI_DOUBLE, right, 1,
             &local_in[0], 2, MPI_DOUBLE, left, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Stencil computation
        for (int i = 2; i < local_n + 2; i++) {
            double sum = 0.0;

            for (int j = 0; j < W; j++) {
                sum += stencil[j] * local_in[i - EXT + j];
            }

        local_out[i] = sum;
    }

        // Swap buffers
        double *tmp = local_in;
        local_in = local_out;
        local_out = tmp;
    }

    double local_time = MPI_Wtime() - start;

    // -------------------------
    // Gather results
    // -------------------------
    double *output = NULL;
    if (rank == 0) {
        output = malloc(num_values * sizeof(double));
    }

    MPI_Gather(&local_in[2], local_n, MPI_DOUBLE,
               output, local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // -------------------------
    // Reduce timing (MAX time)
    // -------------------------
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // -------------------------
    // Output (rank 0 only)
    // -------------------------
    if (rank == 0) {
        printf("%f\n", max_time);

#ifdef PRODUCE_OUTPUT_FILE
        write_output(output_name, output, num_values);
#endif

        free(output);
    }

    free(local_in);
    free(local_out);

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
