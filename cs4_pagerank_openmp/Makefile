CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
OMPFLAGS = -fopenmp
LDFLAGS = -lm

all: serial parallel bandwidth

serial: serial.c
	$(CC) $(CFLAGS) serial.c -o serial $(LDFLAGS)

parallel: parallel.c
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel.c -o parallel $(LDFLAGS)

bandwidth: bandwidth.c
	$(CC) $(CFLAGS) $(OMPFLAGS) bandwidth.c -o bandwidth

clean:
	rm -f serial parallel bandwidth *.o degree_distribution.txt

.PHONY: all clean
