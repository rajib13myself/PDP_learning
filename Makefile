############################
# Makefile for Mandatory lab 4
############################

CC = mpicc
CCFLAGS = -g -std=c99 -O3
LIBS = -lm

BINS = pi

all: $(BINS)

pi: pi.c
	$(CC) $(CCFLAGS) -o pi pi.c $(LIBS)
matrix: matrix-vector.c
	$(CC) $(CCFLAGS) -o matrix-vector matrix-vector.c $(LIBS)
clean:
	$(RM) $(BINS)

