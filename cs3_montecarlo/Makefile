CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -Wextra -std=c99
OMPFLAGS = -fopenmp
LDFLAGS = -lm

all: serial parallel parallel_mpi

serial: serial.c
	$(CC) $(CFLAGS) serial.c -o serial $(LDFLAGS)

parallel: parallel.c
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel.c -o parallel $(LDFLAGS)

parallel_mpi: parallel_mpi.c
	$(MPICC) $(CFLAGS) parallel_mpi.c -o parallel_mpi $(LDFLAGS)

clean:
	rm -f serial parallel parallel_mpi *.o

.PHONY: all clean
