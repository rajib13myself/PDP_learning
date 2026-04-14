############################
# Makefile for Mandatory lab 4
############################

CC = mpicc
CCFLAGS = -g -std=c99 -O3
LIBS = -lm

BINS = pi

all: $(BINS)

%: %.c
	$(CC) $(CCFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) $(BINS)

