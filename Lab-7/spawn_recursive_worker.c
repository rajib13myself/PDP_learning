// Worker code

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
int K, myrank, size, tag = 3;
MPI_Status status;
MPI_Comm parentcomm;
pid_t pid;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_get_parent(&parentcomm);

MPI_Recv(&K, 1, MPI_INT, 0, tag, parentcomm, &status); //recv K
printf("child %d: k=%d\n", myrank, K);
K++;
if (K<5) { // && myrank==0) {
    MPI_Comm childParentComm;
    pid = getpid();
    printf("Recursive:  I am %d of %d, pid %ld: \n", myrank, size,  pid);fflush(stdout);
    MPI_Comm_spawn("spawn_recursive_worker",
            MPI_ARGV_NULL, 2, MPI_INFO_NULL, 0, MPI_COMM_SELF,
            &childParentComm, MPI_ERRCODES_IGNORE);
     printf("Recursive: Parent %ld done with spawn: 2 workers\n", pid);

    //sends K to first worker_child
    MPI_Send(&K, 1, MPI_INT, 0, tag, childParentComm);

    K++;
    //sends K+1 to second worker_child
    MPI_Send(&K, 1, MPI_INT, 1, tag, childParentComm);

    int K1, K2;
    MPI_Recv(&K1, 1, MPI_INT, 0, tag, childParentComm, &status);
    MPI_Recv(&K2, 1, MPI_INT, 1, tag, childParentComm, &status);
    printf("Recursive: Parent %ld received K1=%d and K2= %d\n", pid, K1, K2);
    K = K1 + K2;
    MPI_Comm_free(&childParentComm);
}
MPI_Send(&K, 1, MPI_INT, 0, tag, parentcomm);
MPI_Comm_free(&parentcomm);
MPI_Finalize();
return 0;
}
