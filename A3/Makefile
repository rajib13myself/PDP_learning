CC = mpicc
CFLAGS = -Wall -O3 -g

BINS = quicksort
OBJS = quicksort.o pivot.o

all: $(BINS)

quicksort: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

quicksort.o: quicksort.c quicksort.h pivot.h
	$(CC) $(CFLAGS) -c quicksort.c

pivot.o: pivot.c pivot.h
	$(CC) $(CFLAGS) -c pivot.c

clean:
	rm -f $(BINS) $(OBJS)