#include "stencil.h"

int main(int argc, char **argv) {

    MPI_Init(&argc , &argv);

    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (4 != argc) {
        if (rank == 0) {
            printf("Usage: stencil input_file output_file number_of_applications\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *input_name = argv[1];
    char *output_name = argv[2];
    int num_steps = atoi(argv[3]);

    // read input only on rank 0
    double *input = NULL;
    int num_values;

    if (rank == 0) {
        if (0 > (num_values = read_input(input_name, &input))) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = num_values / size;

    double *local_input  = malloc((local_n + 4) * sizeof(double));
    double *local_output = malloc((local_n + 4) * sizeof(double));

    MPI_Scatter(input, local_n, MPI_DOUBLE,
                &local_input[2], local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double h = 2.0 * PI / num_values;
    const int STENCIL_WIDTH = 5;
    const int EXTENT = STENCIL_WIDTH / 2;
    const double STENCIL[] =
        {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

    int left_neigh  = (rank - 1 + size) % size;
    int right_neigh = (rank + 1) % size;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int s = 0; s < num_steps; s++) {

        // exchange ghost cells
        MPI_Sendrecv(&local_input[2], 2, MPI_DOUBLE, left_neigh, 0,
                     &local_input[local_n + 2], 2, MPI_DOUBLE, right_neigh, 0,
                     MPI_COMM_WORLD, &status);

        MPI_Sendrecv(&local_input[local_n], 2, MPI_DOUBLE, right_neigh, 1,
                     &local_input[0], 2, MPI_DOUBLE, left_neigh, 1,
                     MPI_COMM_WORLD, &status);

        // stencil computation
        for (int i = 2; i < local_n + 2; i++) {
            double result = 0;
            for (int j = 0; j < STENCIL_WIDTH; j++) {
                int index = i - EXTENT + j;
                result += STENCIL[j] * local_input[index];
            }
            local_output[i] = result;
        }

        // swap buffers
        double *tmp = local_input;
        local_input = local_output;
        local_output = tmp;
    }

    double end = MPI_Wtime();
    double local_time = end - start;

    // compute max time
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // gather final result
    double *output = NULL;
    if (rank == 0) {
        output = malloc(num_values * sizeof(double));
    }

    MPI_Gather(&local_input[2], local_n, MPI_DOUBLE,
               output, local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f\n", max_time);
        write_output(output_name, output, num_values);
    }

    free(local_input);
    free(local_output);

    if (rank == 0) {
        free(input);
        free(output);
    }

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
