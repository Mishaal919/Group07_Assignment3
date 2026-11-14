# Case Study 1: Heat Diffusion Simulation (MPI)
Group 07 | Armaghan Ahmed (23L-2555) and Mishaal Rafique (23L-3044)

## What this program does
Simulates 2D heat spreading over an N x N grid using the standard 5-point stencil across T time steps:
`u(i,j, t+1) = u(i,j, t) + alpha * [ u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) - 4*u(i,j) ]`
Boundary conditions are fixed: Top = 100, Bottom = 0, Left = 75, Right = 50. Alpha is set to 0.20 for numerical stability.

## Files
- `serial.c`: Baseline serial version using double-buffered pointers.
- `parallel.c`: MPI version supporting:
  - 1D row slicing (`--1d`) or 2D Cartesian grid (`--2d`, default)
  - Blocking communication (`--blocking`) or non-blocking overlap (`--nonblocking`, default)
  - Periodic convergence checks every K steps using `MPI_Allreduce(MPI_MAX)`
- `parallel_hybrid.c`: Hybrid MPI + OpenMP solver.
- `Makefile`: Compiles with `mpicc -O3`.

## How to build
```bash
make
```

## How to run
Serial:
```bash
./serial 1024 500 50
```

Parallel (2D non-blocking, recommended):
```bash
mpirun -np 4 ./parallel -n 1024 -t 500 -k 50 --2d --nonblocking
```

Parallel (1D blocking):
```bash
mpirun -np 4 ./parallel -n 1024 -t 500 -k 50 --1d --blocking
```

Bonus Hybrid (2 ranks x 2 threads):
```bash
mpirun -np 2 ./parallel_hybrid -n 1024 -t 500 -k 50 -omp 2
```

Use N = 1024 or 4096 for strong scaling and N = 1024, 1448, 2048 and 2896 for weak scaling. For a cell-by-cell verification, pass an output filename as the fourth serial argument and pass `--verify FILE` to the MPI version. The grid tolerance is 1e-10.
