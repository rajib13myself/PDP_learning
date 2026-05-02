// Manager code

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
int myrank, size, numworkers, tag = 3, sum = 0, K = 1;
MPI_Status status;
MPI_Comm workercomm;
pid_t pid;

numworkers = atoi(argv[1]);

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
pid = getpid();
    printf("I am %d of %d, pid %ld: \n", myrank, size, pid);fflush(stdout);
printf("Parent [pid %ld] about to spawn!\n", (long)pid);
MPI_Comm_spawn("spawn_recursive_worker",
        MPI_ARGV_NULL, numworkers, MPI_INFO_NULL, 0, MPI_COMM_SELF,
        &workercomm, MPI_ERRCODES_IGNORE);
printf("Parent %ld done with spawn: %d workers\n", pid, numworkers);

for (int i = 0; i < numworkers; i++) { //sends 1,2,3... to workers
    K=i+1;
    MPI_Send(&K, 1, MPI_INT, i, tag, workercomm);
}

for (int i = 0; i < numworkers; i++) { //receives an int from worker
    MPI_Recv(&K, 1, MPI_INT, i, tag, workercomm, &status);
    sum += K;
}
printf("%d \n", sum);

MPI_Comm_free(&workercomm);
MPI_Finalize();
return 0;
}

