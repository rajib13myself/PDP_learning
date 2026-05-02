########################################################################
# Makefile for MPI-advanced
#
########################################################################

CC         =  mpicc
CCFLAGS    =  -g -O3 -std=gnu99
LIBS       =  -lmpi -lm

all: main work

array:		array_sect.c
	$(CC) $(CCFLAGS) -o array_sect array_sect.c $(LIBS)

oneside:	oneside_com.c
	$(CC) $(CCFLAGS) -o oneside oneside_com.c $(LIBS)

main:         spawn_recursive_main.c
	$(CC) $(CCFLAGS) -o Demo_spawn_rec spawn_recursive_main.c $(LIBS)

work:         spawn_recursive_worker.c
	$(CC) $(CCFLAGS) -o spawn_recursive_worker spawn_recursive_worker.c $(LIBS)


clean:
	$(RM)  array_sect oneside Demo_spawn_rec spawn_recursive_worker

