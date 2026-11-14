# Case Study 3: Monte Carlo Option Pricing (OpenMP & MPI)
Group 07 | Armaghan Ahmed (23L-2555) and Mishaal Rafique (23L-3044)

## What this program does
Prices a European call option using Geometric Brownian Motion (GBM) over M >= 10^8 random sample paths. Compares the simulated price and standard error against the exact closed-form Black-Scholes formula.

Default parameters: S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0.

## Files
- `serial.c`: Serial baseline using SplitMix64 and Box-Muller transforms.
- `parallel.c`: OpenMP implementation with disjoint blocks of a reproducible random sequence. Also includes:
  - `--shared-rng`: Demonstrates severe lock contention when threads share one PRNG.
  - `--same-seed`: Demonstrates why using the same seed produces duplicate paths.
- `parallel_mpi.c`: Distributed memory MPI implementation (Bonus +3 marks).
- `Makefile`: Compiles both OpenMP and MPI executables.

## How to build
```bash
make
```

## How to run
Serial:
```bash
./serial 100000000 123456789
```

Parallel OpenMP (recommended independent streams):
```bash
./parallel -m 100000000 -p 4 -s 123456789
```

Lock contention test:
```bash
./parallel -m 10000000 -p 4 --shared-rng
```

Same seed test:
```bash
./parallel -m 100000000 -p 4 --same-seed
```

MPI version (Bonus):
```bash
mpirun -np 4 ./parallel_mpi -m 100000000 -s 123456789
```

Weak scaling starts at 12.5 million paths and adds 12.5 million per worker. The correctness script repeats the same seed and compares prices and errors to the serial result for 1, 2, 4 and 8 workers.
