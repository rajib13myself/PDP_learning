#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int msg;
    MPI_Comm parent, child;
    int rank, size, len, no_child = 3;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    pid_t pid;
    int i;
    char *cmds[] = {"Demo_spawn", "Demo_spawn", "Demo_spawn" };
    char *argv0[] = { "foo", NULL };
    char *argv1[] = { "bar", NULL };
    char *argv2[] = { "foo", NULL };
    char **spawn_argv[3];
    int maxprocs[] = { 1, 1, 1};
    MPI_Info info[] = { MPI_INFO_NULL, MPI_INFO_NULL, MPI_INFO_NULL };

    spawn_argv[0] = argv0;
    spawn_argv[1] = argv1;
    spawn_argv[2] = argv2;
	    
    MPI_Init(&argc,&argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Comm_get_parent(&parent);  
    MPI_Get_processor_name(hostname, &len);
    pid = getpid();
    printf("Hello, world.  I am %d of %d on %s, pid %ld: \n", rank, size, hostname, pid);fflush(stdout);

 
    /* If we get COMM_NULL back, then we're the parent */
      if (MPI_COMM_NULL == parent) {
        pid = getpid();
        printf("Parent [pid %ld] about to spawn!\n", (long)pid);
        MPI_Comm_spawn_multiple(no_child, cmds, MPI_ARGVS_NULL, maxprocs,
                                info, 0, MPI_COMM_WORLD,
                                &child, MPI_ERRCODES_IGNORE);
        printf("Parent %ld done with spawn\n",pid);
        if (0 == rank) {
            msg = 55;
            printf("Parent  [pid %ld] sending message '%d' to children\n",pid,msg);
	    for (i==0;i<no_child; ++i)
	    {
               MPI_Send(&msg, 1, MPI_INT, i, 1, child);
	    }
	    }
/*	else {
           msg = 56;
            printf("Parent  [pid %ld] sending message '%d' to children\n",pid,msg);
	    for (i==0;i<no_child; ++i)
	    {
               MPI_Send(&msg, 1, MPI_INT, i, 1, child);
	    }	
        }*/
        MPI_Comm_disconnect(&child);
        printf("Parent %ld disconnected\n", pid);
    }
    /* Otherwise, we're the child */
  else {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Get_processor_name(hostname, &len);
        pid = getpid();
        printf("Hello from child no. %d of %d on host %s pid %ld: argv[1] = %s\n", rank, size, hostname, (long)pid, argv[1]);
        MPI_Recv(&msg, 1, MPI_INT, 0, 1, parent, MPI_STATUS_IGNORE);
        printf("Child %d received msg: %d\n", rank, msg);
        MPI_Comm_disconnect(&parent);
        printf("Child %d disconnected\n", rank);
    }

    MPI_Finalize();
    return 0;
}
