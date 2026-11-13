CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
OMPFLAGS = -fopenmp

all: serial parallel

serial: serial.c
	$(CC) $(CFLAGS) serial.c -o serial

parallel: parallel.c
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel.c -o parallel

clean:
	rm -f serial parallel *.o

.PHONY: all clean
