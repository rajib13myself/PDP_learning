#include "stencil.h"

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

    // Read input
    if (rank == 0) {
        if ((num_values = read_input(input_name, &input)) < 0) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double h = 2.0 * PI / num_values;
    const int STENCIL_WIDTH = 5;
    const int EXTENT = STENCIL_WIDTH / 2;

    const double STENCIL[5] = {
        1.0/(12*h), -8.0/(12*h), 0.0,
        8.0/(12*h), -1.0/(12*h)
    };

    int local_n = num_values / size;

    double *local_input  = malloc((local_n + 2*EXTENT) * sizeof(double));
    double *local_output = malloc((local_n + 2*EXTENT) * sizeof(double));

    double *left_ghost  = malloc(EXTENT * sizeof(double));
    double *right_ghost = malloc(EXTENT * sizeof(double));

    if (!local_input || !local_output || !left_ghost || !right_ghost) {
        perror("malloc failed");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // Scatter
    MPI_Scatter(input, local_n, MPI_DOUBLE,
                &local_input[EXTENT], local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    
    for (int i = 0; i < local_n + 2*EXTENT; i++) {
       local_input[i] = 0.0;
       local_output[i] = 0.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int s = 0; s < num_steps; s++) {

        // =========================
        // HALO EXCHANGE (FIXED)
        // =========================

        // send left, recv right ghost
        MPI_Sendrecv(&local_input[EXTENT], EXTENT, MPI_DOUBLE,
             left, 0,
             right_ghost, EXTENT, MPI_DOUBLE,
             right, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send right, recv left ghost
        MPI_Sendrecv(&local_input[EXTENT + local_n - EXTENT], EXTENT, MPI_DOUBLE,
             right, 1,
             left_ghost, EXTENT, MPI_DOUBLE,
             left, 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int i = 0; i < EXTENT; i++) {
           local_input[i] = left_ghost[i];
           local_input[EXTENT + local_n + i] = right_ghost[i];
        }
        
        // =========================
        // STENCIL COMPUTATION
        // =========================
        for (int i = EXTENT; i < EXTENT + local_n; i++) {

            double result = 0.0;

            for (int j = 0; j < STENCIL_WIDTH; j++) {
                result += STENCIL[j] *
                      local_input[i - EXTENT + j];
            }

            local_output[i] = result;
        }

        // swap buffers
        double *tmp = local_input;
        local_input = local_output;
        local_output = tmp;
    }

    double end_time = MPI_Wtime() - start_time;

    double max_time;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *output = NULL;

    if (rank == 0) {
        output = malloc(num_values * sizeof(double));
    }

    MPI_Gather(&local_input[EXTENT], local_n, MPI_DOUBLE,
               output, local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {

        printf("%f\n", max_time);

        // ALWAYS write output (FIXED)
        write_output(output_name, output, num_values);

        free(input);
        free(output);
    }

    free(local_input);
    free(local_output);

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
