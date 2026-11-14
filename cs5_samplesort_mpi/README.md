# Case Study 5: Distributed Sample Sort (MPI)
Group 07 | Armaghan Ahmed (23L-2555) and Mishaal Rafique (23L-3044)

## What this program does
The strong-scaling benchmark sorts 100,000,000 32-bit integer keys (400 MB). Each rank generates and sorts its own shard. Load-balance tests use 10,000,000 keys.
Uses the 4-phase sample sort algorithm:
1. Local quicksort on N/p keys.
2. Regular sampling with oversampling factor s, root sorts samples and broadcasts p-1 splitters.
3. Binary search local partitioning, `MPI_Alltoall` for counts, and `MPI_Alltoallv` for data.
4. Local quicksort of received buckets.

Verifies that each rank is sorted, `max_i <= min_{i+1}` across ranks, and total element count equals N.

## Files
- `serial.c`: Serial quicksort baseline on N keys.
- `parallel.c`: Distributed sample sort in MPI.
- `pingpong.c`: Ping-pong latency and bandwidth benchmark for the alpha-beta communication cost model.
- `Makefile`: Compiles serial, parallel, and pingpong.

## How to build
```bash
make
```

## How to run
Serial:
```bash
./serial 100000000 0
```

Parallel Sample Sort (Uniform distribution):
```bash
mpirun -np 4 ./parallel -n 100000000 --uniform -s 16
```

Parallel Sample Sort (Skewed distribution):
```bash
mpirun -np 4 ./parallel -n 10000000 --skewed -s 16
```

Ping-pong communication test:
```bash
mpirun -np 2 ./pingpong
```

Use `./std_sort 100000000 0` for the single-process C++ comparison. For a key-by-key check, append a filename to the serial or std::sort arguments, then pass `--verify FILE` to MPI. Both distributions use seed 42. Each rank skips to its own part of the stream.
