# Case Study 2: Genomics k-mer Counting (OpenMP)
Group 07 | Armaghan Ahmed (23L-2555) and Mishaal Rafique (23L-3044)

## What this program does
Counts the frequency of all substrings of length k (k-mers) in a random synthetic DNA sequence of length L = 10^8 characters over {A, C, G, T}. Uses 2-bit encoding per base (A=00, C=01, G=10, T=11), resulting in 4^k histogram bins.

## Files
- `serial.c`: Serial sliding window with 2-bit bitmask shifts.
- `parallel.c`: OpenMP implementation with:
  - `-v A`: Version A (data race on purpose with unprotected parallel loop)
  - `-v B`: Version B (uses `#pragma omp atomic`)
  - `-v C`: Version C (private histograms merged at the end)
  - `--falsesharing`: Diagnostic benchmark showing cache-line false sharing vs 64-byte padding
- `Makefile`: Compiles with `gcc -O3 -fopenmp`.

## How to build
```bash
make
```

## How to run
Serial:
```bash
./serial 100000000 8 42
```

Version A (see the race condition and lost counts):
```bash
./parallel -l 100000000 -k 8 -p 4 -v A
```

Version B (atomic updates):
```bash
./parallel -l 100000000 -k 8 -p 4 -v B
./parallel -l 100000000 -k 12 -p 4 -v B
```

Version C (private histograms):
```bash
./parallel -l 100000000 -k 8 -p 4 -v C
./parallel -l 100000000 -k 12 -p 4 -v C
```

False sharing experiment:
```bash
./parallel -p 4 --falsesharing
```

The sequence seed is 42. The four-thread A/B/C comparison uses either k = 8 or k = 12. Strong scaling uses 100 million bases and the B/C comparison uses 50 million bases. For an exact comparison, save the serial bins with a fourth argument and save the parallel bins with `--dump FILE`. Compare the binary files with `cmp`.
