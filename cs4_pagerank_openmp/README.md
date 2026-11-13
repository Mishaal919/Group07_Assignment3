# Case Study 4: Parallel PageRank (OpenMP)
Group 07 | Armaghan Ahmed (23L-2555) and Mishaal Rafique (23L-3044)

## What this program does
Generates a synthetic power-law social network graph (|V| = 150,000, |E| = 2,000,000) using the R-MAT model (a=0.57, b=0.19, c=0.19, d=0.05) and stores it in Compressed Sparse Row (CSR) format. Iterates PageRank with damping factor d = 0.85 until convergence. Handles dangling nodes by redistributing their probability mass evenly.

## Files
- `serial.c`: Serial baseline using CSR pull-style iteration.
- `parallel.c`: OpenMP implementation comparing:
  - `--pull`: Race-free pull-style destination partitioning (recommended).
  - `--push`: Push-style updates using `#pragma omp atomic`.
  - `--sched static|dynamic|guided`: Explores scheduling clauses.
  - `-c <chunk>`: Sets the OpenMP chunk size.
  - `--dump-degrees`: Exports degree distributions for plotting.
- `Makefile`: Compiles with `gcc -O3 -fopenmp`.

## How to build
```bash
make
```

## How to run
Serial:
```bash
./serial 150000 2000000 30
```

Parallel Pull (Static):
```bash
./parallel -n 150000 -m 2000000 -p 4 -i 30 --pull --sched static
```

Parallel Pull (Dynamic with chunk size 1024):
```bash
./parallel -n 150000 -m 2000000 -p 4 -i 30 --pull --sched dynamic -c 1024
```

Parallel Push (Atomic):
```bash
./parallel -n 150000 -m 2000000 -p 4 -i 30 --push
```

Strong scaling and schedule tests used a cap of 30 iterations. The accuracy tests use a higher cap of 100, save the serial vector with a fourth argument and use `--dump FILE` for either parallel method. The per-vertex tolerance is 1e-10. Run `./bandwidth` for the four-thread streaming copy measurement.
