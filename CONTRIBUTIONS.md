# Group 07 — Assignment 3 Contributions & Work Log

**Course:** CS3006 – Parallel and Distributed Computing  
**Group:** Group 07  
**Members:**  
- Armaghan Ahmed (Roll No: 23L-2555)  
- Mishaal Rafique (Roll No: 23L-3044)  

---

## 1. Division of Work and Roles

In accordance with the guidelines in Section 7 of the assignment specification, ownership of the five case studies was divided so that each member served as the Lead on at least two case studies and as the Reviewer on the remaining ones:

| Case Study | Domain / Topic | Model | Lead | Reviewer |
| :--- | :--- | :--- | :--- | :--- |
| **Case Study 1** | Weather & Heat Diffusion Simulation | MPI | Armaghan Ahmed | Mishaal Rafique |
| **Case Study 2** | Genomics Parallel $k$-mer Counting | OpenMP | Mishaal Rafique | Armaghan Ahmed |
| **Case Study 3** | Financial Risk: Monte Carlo Option Pricing | OpenMP / MPI | Armaghan Ahmed | Mishaal Rafique |
| **Case Study 4** | Social Network Analytics: Parallel PageRank | OpenMP | Mishaal Rafique | Armaghan Ahmed |
| **Case Study 5** | Big-Data Sorting: Distributed Sample Sort | MPI | Armaghan Ahmed | Mishaal Rafique |
| **Joint Deliverables** | Benchmarking, Report, Slides, Integration | OpenMP + MPI | Both | Both |

---

## 2. Detailed Log of Individual Contributions

### Armaghan Ahmed (23L-2555)
- **Role:** Lead for Case Studies 1, 3, 5; Reviewer for Case Studies 2, 4.
- **Estimated Effort:** ~16 hours across the two-week schedule.
- **Tasks & Authored Deliverables:**
  - **Case Study 1 (`cs1_heat_mpi/`):**
    - Authored `serial.c` with 5-point stencil and double-buffered grid pointer swapping.
    - Implemented MPI 1D row-block domain decomposition and 2D Cartesian block decomposition via `MPI_Cart_create`.
    - Developed both blocking halo exchange (`MPI_Sendrecv`) and non-blocking halo exchange (`MPI_Isend` / `MPI_Irecv`) to overlap interior domain stencil computations with border data transfers.
    - Added global convergence checking every $K$ steps with `MPI_Allreduce(MPI_MAX)`.
    - Developed the hybrid MPI + OpenMP implementation (`parallel_hybrid.c`) for the bonus item.
  - **Case Study 3 (`cs3_montecarlo/`):**
    - Authored `serial.c` baseline using SplitMix64 uniform PRNG and Box-Muller transforms for Geometric Brownian Motion option pricing.
    - Implemented OpenMP parallel estimator (`parallel.c`) with reproducible disjoint random streams using SplitMix64 block skip-ahead offsets.
    - Built diagnostic benchmarks demonstrating lock contention in a shared PRNG (`--shared-rng`) and statistical correlation issues when using identical seeds across threads (`--same-seed`).
    - Verified output against Black-Scholes analytical prices and calculated standard errors.
    - Authored the distributed-memory MPI option pricer (`parallel_mpi.c`) for the alternative model bonus item.
  - **Case Study 5 (`cs5_samplesort_mpi/`):**
    - Implemented 4-phase distributed sample sort in MPI (`parallel.c`): local quicksort, regular sample extraction, root splitter broadcast, `MPI_Alltoall` for partition counts, `MPI_Alltoallv` for key redistribution, and local bucket sort.
    - Implemented uniform random and skewed key generators.
    - Developed verification routines confirming local ordering on all ranks, global boundary ordering ($\max_i \le \min_{i+1}$), and total element conservation.
    - Authored ping-pong benchmark (`pingpong.c`) to empirically determine latency $\alpha$ and bandwidth $\beta$ for the analytical communication model.
  - **Reviewing & Joint Work:**
    - Reviewed, built, and validated Mishaal's k-mer counting and PageRank code on test hardware.
    - Authored report sections for Cases 1, 3, 5; contributed to the introduction, cross-case analysis, and slides for Cases 1, 3, 5.

---

### Mishaal Rafique (23L-3044)
- **Role:** Lead for Case Studies 2, 4; Reviewer for Case Studies 1, 3, 5.
- **Estimated Effort:** ~15.5 hours across the two-week schedule.
- **Tasks & Authored Deliverables:**
  - **Case Study 2 (`cs2_kmer_openmp/`):**
    - Authored `serial.c` baseline sliding window algorithm with compact 2-bit nucleotide encoding (A=00, C=01, G=10, T=11) mapping to $4^k$ bins.
    - Implemented Version A (unprotected OpenMP parallel loop demonstrating data races and lost increments).
    - Implemented Version B (race-free with `#pragma omp atomic`).
    - Implemented Version C (per-thread private histograms with parallel bin-wise reduction at loop completion).
    - Handled boundary chunk logic so $k$-mers spanning adjacent thread chunks are counted exactly once without duplicates or omissions.
    - Built diagnostic false-sharing benchmark (`--falsesharing`) contrasting packed thread counters against 64-byte cache-line padded counters.
  - **Case Study 4 (`cs4_pagerank_openmp/`):**
    - Implemented synthetic power-law graph generation using the R-MAT algorithm ($a=0.57, b=0.19, c=0.19, d=0.05$) generating $2 \times 10^6$ directed edges stored in Compressed Sparse Row (CSR) format.
    - Authored serial CSR PageRank baseline (`serial.c`) with dangling node probability redistribution.
    - Implemented parallel pull-style iteration (destination vertex partitioning, race-free reads) and parallel push-style iteration (`#pragma omp atomic` updates on destination ranks).
    - Benchmarked OpenMP scheduling clauses (`static`, `dynamic`, `guided`) across chunk sizes and computed the load imbalance ratio $R = T_{\text{slowest}} / T_{\text{mean}}$.
    - Authored streaming memory bandwidth benchmark (`bandwidth.c`) to compare achieved throughput against the memory bus limit.
  - **Reviewing & Joint Work:**
    - Reviewed, built, and validated Armaghan's heat diffusion, Monte Carlo, and sample sort code.
    - Authored report sections for Cases 2 and 4; contributed to the cross-case comparison table, roofline discussion, and slides for Cases 2 and 4.

---

## 3. Shared Responsibilities & Integration

- **Benchmarking & Timing:** Conducted joint sequential timing sweeps on an idle machine (AMD Ryzen 5 PRO 3500U, 4 cores / 8 threads, 16 GiB RAM). Timings were recorded as the median of $\ge 3$ consecutive runs.
- **Report & Slides:** Jointly synthesized the 13-page technical report (`report/Group07_report.pdf`) and the 6 presentation slides (`slides/Group07_slides.pdf`).
- **Code Cleanliness & Verification:** Ensured all source files build with `make` under `-O3 -Wall -Wextra`, produce zero compiler warnings, and verify outputs against the serial baselines.
