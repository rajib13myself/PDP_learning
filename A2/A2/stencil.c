#include "stencil.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
	MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (4 != argc) {
        if (rank == 0)
            printf("Usage: stencil input_file output_file number_of_applications\n");
        MPI_Finalize();
        return 1;
    }

    char *input_name = argv[1];
    char *output_name = argv[2];
    int num_steps = atoi(argv[3]);

    double *input = NULL;
    int num_values;

    // =========================
    // Read input on rank 0
    // =========================
    if (rank == 0) {
        if (0 > (num_values = read_input(input_name, &input))) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    // =========================
    // Share num_values for all rank
    // =========================
    //MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0) {
    	for (int p = 1; p < size; p++) {
        	MPI_Send(&num_values, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
    	}
	} else {
    	MPI_Recv(&num_values, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	}

    // =========================
    // Allocate input on others
    // =========================
    if (rank != 0) {
        input = malloc(num_values * sizeof(double));
    }

    // =========================
    // Manual broadcast
    // =========================
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Send(input, num_values, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(input, num_values, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // =========================
    // Stencil setup
    // =========================
    double h = 2.0 * PI / num_values;
    const int STENCIL_WIDTH = 5;
    const int EXTENT = STENCIL_WIDTH / 2;
    const double STENCIL[] = {
        1.0/(12*h), -8.0/(12*h), 0.0,
        8.0/(12*h), -1.0/(12*h)
    };

    double *output = malloc(num_values * sizeof(double));

    // =========================
    // Work division
    // =========================
    int chunk = num_values / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? num_values : start + chunk;

    double start_time = MPI_Wtime();

    for (int s = 0; s < num_steps; s++) {

        // --- Left Loop ---
        for (int i = 0; i < EXTENT; i++) {
            if (i < start || i >= end) continue;

            double result = 0;
            for (int j = 0; j < STENCIL_WIDTH; j++) {
                int index = (i - EXTENT + j + num_values) % num_values;
                result += STENCIL[j] * input[index];
            }
            output[i] = result;
        }

        // --- Middle Loop ---
        for (int i = EXTENT; i < num_values - EXTENT; i++) {
            if (i < start || i >= end) continue;

            double result = 0;
            for (int j = 0; j < STENCIL_WIDTH; j++) {
                int index = i - EXTENT + j + EXTENT;
                result += STENCIL[j] * input[index];
            }
            output[i] = result;
        }

        // --- Right Loop ---
        for (int i = num_values - EXTENT; i < num_values; i++) {
            if (i < start || i >= end) continue;

            double result = 0;
            for (int j = 0; j < STENCIL_WIDTH; j++) {
                int index = (i - EXTENT + j) % num_values;
                result += STENCIL[j] * input[index];
            }
            output[i] = result;
        }

        // =========================
        // Gather results for output
        // =========================
        if (rank == 0) {
            for (int p = 1; p < size; p++) {
                int p_start = p * chunk;
                int p_end = (p == size - 1) ? num_values : p_start + chunk;
                MPI_Recv(&output[p_start], p_end - p_start, MPI_DOUBLE,
                         p, 1, MPI_COMM_WORLD, &status);
            }
        } else {
            MPI_Send(&output[start], end - start, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        // Broadcast updated array again
        if (rank == 0) {
            for (int p = 1; p < size; p++) {
                MPI_Send(output, num_values, MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(input, num_values, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        }

        //Swap
            double *tmp = input;
            input = output;
            output = tmp;
    }

    double my_execution_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        printf("%f\n", my_execution_time);

		#ifdef PRODUCE_OUTPUT_FILE
        	write_output(output_name, input, num_values);
		#endif
    }

    free(input);
    free(output);

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
