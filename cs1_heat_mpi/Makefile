CC = mpicc
CFLAGS = -O3 -Wall -Wextra -std=c99
OMPFLAGS = -fopenmp
LDFLAGS = -lm

all: serial parallel parallel_hybrid

serial: serial.c
	gcc -O3 -Wall -Wextra -std=c99 serial.c -o serial $(LDFLAGS)

parallel: parallel.c
	$(CC) $(CFLAGS) parallel.c -o parallel $(LDFLAGS)

parallel_hybrid: parallel_hybrid.c
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel_hybrid.c -o parallel_hybrid $(LDFLAGS)

clean:
	rm -f serial parallel parallel_hybrid *.o

.PHONY: all clean
