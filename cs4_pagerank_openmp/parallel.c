#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// CS3006 PDC Assignment 3 - Case Study 4: Parallel PageRank with OpenMP
// Implements:
// - Pull-style kernel (vertices read in-neighbors; zero write conflicts)
// - Push-style kernel (vertices push to out-neighbors; uses atomic updates)
// - OpenMP schedule tuning: static, dynamic, guided with variable chunk sizes
// - Measures thread load imbalance: R = max_thread_time / mean_thread_time

typedef struct {
    int src;
    int dst;
} GraphEdge;

void make_rmat_graph(int n, int m, double a, double b, double c, GraphEdge *edges, unsigned int seed) {
    srand(seed);
    for (int e = 0; e < m; e++) {
        int u = 0, v = 0;
        int step = n / 2;
        while (step > 0) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < a) {}
            else if (r < a + b) v += step;
            else if (r < a + b + c) u += step;
            else { u += step; v += step; }
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
    int num_threads = 4;
    double damping = 0.85;
    double tolerance = 1e-6;
    int max_iterations = 50;
    unsigned int seed = 42;
    int use_push_mode = 0; // 0 = pull, 1 = push
    int sched_choice = 0;  // 0 = static, 1 = dynamic, 2 = guided
    int chunk_size = 0;
    int export_degrees = 0;

    const char *dump_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) num_nodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) num_edges = atoi(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) max_iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--push") == 0) use_push_mode = 1;
        else if (strcmp(argv[i], "--pull") == 0) use_push_mode = 0;
        else if (strcmp(argv[i], "--sched") == 0 && i + 1 < argc) {
            char *s = argv[++i];
            if (strcmp(s, "static") == 0) sched_choice = 0;
            else if (strcmp(s, "dynamic") == 0) sched_choice = 1;
            else if (strcmp(s, "guided") == 0) sched_choice = 2;
        }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) chunk_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-degrees") == 0) export_degrees = 1;
    }

    if (num_nodes < 1 || num_edges < 1 || num_threads < 1 || max_iterations < 1 || chunk_size < 0)
        return 1;
    if (sched_choice != 0 && chunk_size == 0) chunk_size = 256;
    omp_sched_t schedule = sched_choice == 0 ? omp_sched_static :
        (sched_choice == 1 ? omp_sched_dynamic : omp_sched_guided);
    omp_set_schedule(schedule, chunk_size);
    omp_set_num_threads(num_threads);

    GraphEdge *edge_list = (GraphEdge *)malloc((size_t)num_edges * sizeof(GraphEdge));
    make_rmat_graph(num_nodes, num_edges, 0.57, 0.19, 0.19, edge_list, seed);

    int *out_degrees = (int *)calloc(num_nodes, sizeof(int));
    int *in_degrees  = (int *)calloc(num_nodes, sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        out_degrees[edge_list[e].src]++;
        in_degrees[edge_list[e].dst]++;
    }

    if (export_degrees) {
        FILE *fp = fopen("degree_distribution.txt", "w");
        if (fp) {
            for (int i = 0; i < num_nodes; i++) {
                fprintf(fp, "%d %d %d\n", i, in_degrees[i], out_degrees[i]);
            }
            fclose(fp);
        }
    }

    // Build CSR for In-neighbors (Pull-style)
    int *in_row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));
    in_row_ptr[0] = 0;
    for (int i = 0; i < num_nodes; i++) in_row_ptr[i + 1] = in_row_ptr[i] + in_degrees[i];
    int *in_col_idx = (int *)malloc(num_edges * sizeof(int));
    int *in_cursor = (int *)calloc(num_nodes, sizeof(int));
    for (int e = 0; e < num_edges; e++) {
        int dst = edge_list[e].dst;
        in_col_idx[in_row_ptr[dst] + in_cursor[dst]++] = edge_list[e].src;
    }
    free(in_cursor);

    // Build CSR for Out-neighbors (Push-style) if needed
    int *out_row_ptr = NULL;
    int *out_col_idx = NULL;
    if (use_push_mode) {
        out_row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));
        out_row_ptr[0] = 0;
        for (int i = 0; i < num_nodes; i++) out_row_ptr[i + 1] = out_row_ptr[i] + out_degrees[i];
        out_col_idx = (int *)malloc(num_edges * sizeof(int));
        int *out_cursor = (int *)calloc(num_nodes, sizeof(int));
        for (int e = 0; e < num_edges; e++) {
            int src = edge_list[e].src;
            out_col_idx[out_row_ptr[src] + out_cursor[src]++] = edge_list[e].dst;
        }
        free(out_cursor);
    }
    free(edge_list);

    double *ranks = (double *)malloc(num_nodes * sizeof(double));
    double *ranks_next = (double *)malloc(num_nodes * sizeof(double));
    double init_score = 1.0 / (double)num_nodes;
    for (int i = 0; i < num_nodes; i++) ranks[i] = init_score;

    double *per_thread_times = (double *)calloc(num_threads, sizeof(double));

    double t0 = omp_get_wtime();
    int iters = 0;
    double l1_diff = 0.0;

    for (iters = 0; iters < max_iterations; iters++) {
        // Parallel sum of dangling node mass
        double dangling_mass = 0.0;
        #pragma omp parallel for reduction(+:dangling_mass) schedule(static)
        for (int i = 0; i < num_nodes; i++) {
            if (out_degrees[i] == 0) dangling_mass += ranks[i];
        }

        double base_score = (1.0 - damping + damping * dangling_mass) / (double)num_nodes;
        l1_diff = 0.0;

        if (!use_push_mode) {
            // PULL STYLE: Each thread updates disjoint destination nodes (race-free)
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double t_thr = omp_get_wtime();
                double local_diff = 0.0;

                #pragma omp for schedule(runtime) nowait
                for (int v = 0; v < num_nodes; v++) {
                    double sum = 0.0;
                    for (int e = in_row_ptr[v]; e < in_row_ptr[v + 1]; e++) {
                        int u = in_col_idx[e];
                        sum += ranks[u] / (double)out_degrees[u];
                    }
                    double val = base_score + damping * sum;
                    ranks_next[v] = val;
                    local_diff += fabs(val - ranks[v]);
                }

                // Stop the worker timer before the barrier at the end of the region.
                per_thread_times[tid] += omp_get_wtime() - t_thr;
                #pragma omp atomic
                l1_diff += local_diff;

            }
        } else {
            // PUSH STYLE: Each thread loops over source nodes, pushes to destination with atomics
            #pragma omp parallel for schedule(static)
            for (int v = 0; v < num_nodes; v++) ranks_next[v] = base_score;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double t_thr = omp_get_wtime();

                #pragma omp for schedule(static)
                for (int u = 0; u < num_nodes; u++) {
                    if (out_degrees[u] == 0) continue;
                    double contrib = damping * (ranks[u] / (double)out_degrees[u]);
                    int start = out_row_ptr[u];
                    int end = out_row_ptr[u + 1];
                    for (int e = start; e < end; e++) {
                        int v = out_col_idx[e];
                        // High contention on popular destination hubs
                        #pragma omp atomic
                        ranks_next[v] += contrib;
                    }
                }
                per_thread_times[tid] += (omp_get_wtime() - t_thr);
            }

            #pragma omp parallel for reduction(+:l1_diff) schedule(static)
            for (int v = 0; v < num_nodes; v++) l1_diff += fabs(ranks_next[v] - ranks[v]);
        }

        double *tmp = ranks;
        ranks = ranks_next;
        ranks_next = tmp;

        if (l1_diff < tolerance) {
            iters++;
            break;
        }
    }

    double elapsed = omp_get_wtime() - t0;

    // Load imbalance ratio R = max_time / mean_time
    double max_t = 0.0, total_t = 0.0;
    for (int t = 0; t < num_threads; t++) {
        total_t += per_thread_times[t];
        if (per_thread_times[t] > max_t) max_t = per_thread_times[t];
    }
    double mean_t = total_t / num_threads;
    double imbalance_r = max_t / (mean_t > 1e-9 ? mean_t : 1e-9);

    double sum_rank = 0.0, top_val = 0.0;
    int top_node = 0;
    for (int i = 0; i < num_nodes; i++) {
        sum_rank += ranks[i];
        if (ranks[i] > top_val) {
            top_val = ranks[i];
            top_node = i;
        }
    }

    double edge_rate = ((double)num_edges * iters) / (elapsed * 1e6);
    const char *sched_str[] = {"static", "dynamic", "guided"};

    if (dump_path) {
        FILE *fp = fopen(dump_path, "wb");
        if (!fp) return 1;
        size_t written = fwrite(ranks, sizeof(*ranks), num_nodes, fp);
        if (fclose(fp) != 0 || written != (size_t)(num_nodes)) return 1;
    }

    printf("OPENMP PAGERANK: Mode=%s, Sched=%s(c=%d), P=%d, Iters=%d, Diff=%.6e, Time=%.6f s, Rate=%.2f M edges/s, ImbalanceR=%.3f, TopRank=%.8e (node %d), SumRank=%.6f\n",
           (use_push_mode ? "PUSH" : "PULL"), sched_str[sched_choice], chunk_size, num_threads,
           iters, l1_diff, elapsed, edge_rate, imbalance_r, top_val, top_node, sum_rank);

    free(out_degrees);
    free(in_degrees);
    free(in_row_ptr);
    free(in_col_idx);
    if (out_row_ptr) free(out_row_ptr);
    if (out_col_idx) free(out_col_idx);
    free(ranks);
    free(ranks_next);
    free(per_thread_times);
    return 0;
}
