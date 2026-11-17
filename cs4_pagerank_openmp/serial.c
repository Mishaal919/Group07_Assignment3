#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// CS3006 PDC Assignment 3 - Case Study 4: Parallel PageRank (Serial Baseline)

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    int src;
    int dst;
} GraphEdge;

// Recursive Matrix (R-MAT) generator for realistic power-law social network graphs
void make_rmat_graph(int n, int m, double a, double b, double c, GraphEdge *edges, unsigned int seed) {
    srand(seed);
    for (int e = 0; e < m; e++) {
        int u = 0, v = 0;
        int step = n / 2;
        while (step > 0) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < a) {
                // Top-left quadrant
            } else if (r < a + b) {
                // Top-right
                v += step;
            } else if (r < a + b + c) {
                // Bottom-left
                u += step;
            } else {
                // Bottom-right
                u += step;
                v += step;
            }
            step /= 2;
        }
        if (u >= n) u = n - 1;
        if (v >= n) v = n - 1;
        edges[e].src = u;
        edges[e].dst = v;
    }
}

int main(int argc, char **argv) {
    int num_nodes = 150000;
    int num_edges = 2000000;
    double damping = 0.85;
    double tolerance = 1e-6;
    int max_iterations = 50;
    unsigned int seed = 42;

    if (argc > 1) num_nodes = atoi(argv[1]);
    if (argc > 2) num_edges = atoi(argv[2]);
    if (argc > 3) max_iterations = atoi(argv[3]);

    printf("Building R-MAT power-law graph (|V|=%d, |E|=%d)...\n", num_nodes, num_edges);
    GraphEdge *edge_list = (GraphEdge *)malloc((size_t)num_edges * sizeof(GraphEdge));
    if (!edge_list) {
        fprintf(stderr, "Error: out of memory allocating edge list\n");
        return 1;
    }
    make_rmat_graph(num_nodes, num_edges, 0.57, 0.19, 0.19, edge_list, seed);

    // Compute degrees
    int *out_degrees = (int *)calloc(num_nodes, sizeof(int));
    int *in_degrees  = (int *)calloc(num_nodes, sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        out_degrees[edge_list[e].src]++;
        in_degrees[edge_list[e].dst]++;
    }

    // Build CSR for incoming neighbors (allows race-free pull-style PageRank)
    int *in_row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));
    in_row_ptr[0] = 0;
    for (int i = 0; i < num_nodes; i++) {
        in_row_ptr[i + 1] = in_row_ptr[i] + in_degrees[i];
    }

    int *in_col_idx = (int *)malloc(num_edges * sizeof(int));
    int *fill_offsets = (int *)calloc(num_nodes, sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        int dst = edge_list[e].dst;
        int idx = in_row_ptr[dst] + fill_offsets[dst]++;
        in_col_idx[idx] = edge_list[e].src;
    }
    free(fill_offsets);
    free(edge_list);

    // Initial ranks = 1 / n
    double *ranks = (double *)malloc(num_nodes * sizeof(double));
    double *ranks_next = (double *)malloc(num_nodes * sizeof(double));
    double initial_score = 1.0 / (double)num_nodes;
    for (int i = 0; i < num_nodes; i++) ranks[i] = initial_score;

    double t0 = get_time_sec();
    int iters = 0;
    double l1_diff = 0.0;

    for (iters = 0; iters < max_iterations; iters++) {
        // Collect probability mass from dangling nodes (out_deg == 0)
        double dangling_mass = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            if (out_degrees[i] == 0) {
                dangling_mass += ranks[i];
            }
        }

        // Base score for all vertices (teleportation + redistributed dangling mass)
        double base_score = (1.0 - damping + damping * dangling_mass) / (double)num_nodes;
        l1_diff = 0.0;

        // Pull-style PageRank update
        for (int v = 0; v < num_nodes; v++) {
            double incoming_sum = 0.0;
            int start = in_row_ptr[v];
            int end = in_row_ptr[v + 1];
            for (int e = start; e < end; e++) {
                int u = in_col_idx[e];
                if (out_degrees[u] > 0) {
                    incoming_sum += ranks[u] / (double)out_degrees[u];
                }
            }
            double new_rank = base_score + damping * incoming_sum;
            ranks_next[v] = new_rank;
            l1_diff += fabs(new_rank - ranks[v]);
        }

        // Swap vectors
        double *tmp = ranks;
        ranks = ranks_next;
        ranks_next = tmp;

        if (l1_diff < tolerance) {
            iters++;
            break;
        }
    }

    double elapsed = get_time_sec() - t0;

    double sum_rank = 0.0;
    double max_rank = 0.0;
    int max_vertex = 0;
    for (int i = 0; i < num_nodes; i++) {
        sum_rank += ranks[i];
        if (ranks[i] > max_rank) {
            max_rank = ranks[i];
            max_vertex = i;
        }
    }

    double edge_rate = ((double)num_edges * iters) / (elapsed * 1e6);
    if (argc > 4) {
        FILE *fp = fopen(argv[4], "wb");
        if (!fp) return 1;
        size_t written = fwrite(ranks, sizeof(*ranks), num_nodes, fp);
        if (fclose(fp) != 0 || written != (size_t)(num_nodes)) return 1;
    }

    printf("SERIAL PAGERANK: |V|=%d, |E|=%d, Iters=%d, Diff=%.6e, Time=%.6f s (%.2f M edges/s), SumRank=%.6f, TopRank=%.8e (node %d)\n",
           num_nodes, num_edges, iters, l1_diff, elapsed, edge_rate, sum_rank, max_rank, max_vertex);

    free(out_degrees);
    free(in_degrees);
    free(in_row_ptr);
    free(in_col_idx);
    free(ranks);
    free(ranks_next);
    return 0;
}
