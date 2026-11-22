/*
 * MPI Ping-Pong Benchmark to measure latency (alpha) and per-byte transfer time (beta)
 * for the communication cost model T_comm = alpha + beta * m.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Ping-pong requires at least 2 MPI ranks.\n");
        MPI_Finalize();
        return 0;
    }

    int warmup = 100;
    int trials = 1000;

    /* 1. Zero-byte message to measure latency alpha */
    char dummy = 0;
    for (int i = 0; i < warmup; i++) {
        if (rank == 0) {
            MPI_Send(&dummy, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&dummy, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(&dummy, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < trials; i++) {
        if (rank == 0) {
            MPI_Send(&dummy, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&dummy, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(&dummy, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&dummy, 0, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    double t1 = MPI_Wtime();
    double rtt_0 = (t1 - t0) / trials;
    double alpha = rtt_0 / 2.0; /* one-way latency */

    /* 2. Large message (1 MB) to measure bandwidth and per-byte time beta */
    int num_bytes = 1024 * 1024; /* 1MB */
    char *buf = (char *)malloc(num_bytes);
    int large_trials = 200;

    for (int i = 0; i < 20; i++) {
        if (rank == 0) {
            MPI_Send(buf, num_bytes, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(buf, num_bytes, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(buf, num_bytes, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, num_bytes, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    for (int i = 0; i < large_trials; i++) {
        if (rank == 0) {
            MPI_Send(buf, num_bytes, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(buf, num_bytes, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(buf, num_bytes, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, num_bytes, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }
    double t3 = MPI_Wtime();
    double rtt_large = (t3 - t2) / large_trials;
    double one_way_large = rtt_large / 2.0;
    double beta = (one_way_large - alpha) / (double)num_bytes;
    if (beta < 0) beta = one_way_large / (double)num_bytes;
    double bandwidth_gb_s = (1.0 / beta) / 1e9;

    if (rank == 0) {
        printf("MPI PING-PONG RESULTS:\n");
        printf("  Latency alpha    : %.4e seconds (%.2f microseconds)\n", alpha, alpha * 1e6);
        printf("  Per-byte beta    : %.4e seconds/byte\n", beta);
        printf("  Effective B/W    : %.2f GB/s\n", bandwidth_gb_s);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
