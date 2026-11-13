CC = gcc
CXX ?= g++
MPICC = mpicc
CFLAGS = -O3 -Wall -Wextra -std=c99
LDFLAGS = -lm

all: serial parallel pingpong std_sort

serial: serial.c
	$(CC) $(CFLAGS) serial.c -o serial $(LDFLAGS)

std_sort: serial.c
	$(CXX) -O3 -Wall -Wextra -std=c++11 -x c++ serial.c -o std_sort

parallel: parallel.c
	$(MPICC) $(CFLAGS) parallel.c -o parallel $(LDFLAGS)

pingpong: pingpong.c
	$(MPICC) $(CFLAGS) pingpong.c -o pingpong $(LDFLAGS)

clean:
	rm -f serial parallel pingpong std_sort *.o

.PHONY: all clean
