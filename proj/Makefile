CC = mpicc

CFLAGS = -Wall -O3 -g

TARGET = power

OBJS = main.o matrix.o power.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

main.o: main.c matrix.h power.h
	$(CC) $(CFLAGS) -c main.c

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c matrix.c

power.o: power.c power.h
	$(CC) $(CFLAGS) -c power.c

clean:
	rm -f $(TARGET) $(OBJS)