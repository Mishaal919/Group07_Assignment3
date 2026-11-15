#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Simple timer helper using clock_gettime
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Set up boundary conditions as specified in assignment sheet
// Top = 100, Bottom = 0, Left = 75, Right = 50, Interior = 0
void init_heat_grid(double *grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                grid[i * n + j] = 100.0;
            } else if (i == n - 1) {
                grid[i * n + j] = 0.0;
            } else if (j == 0) {
                grid[i * n + j] = 75.0;
            } else if (j == n - 1) {
                grid[i * n + j] = 50.0;
            } else {
                grid[i * n + j] = 0.0;
            }
        }
    }
}

int main(int argc, char **argv) {
    int n = 1024;
    int steps = 500;
    int k_check = 50;
    double alpha = 0.20; // safe stencil weight (must be <= 0.25 for stability)

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (argc > 3) k_check = atoi(argv[3]);

    size_t total_bytes = (size_t)n * n * sizeof(double);
    double *curr = (double *)malloc(total_bytes);
    double *next = (double *)malloc(total_bytes);

    if (!curr || !next) {
        fprintf(stderr, "Error: out of memory allocating %d x %d grid\n", n, n);
        return 1;
    }

    init_heat_grid(curr, n);
    memcpy(next, curr, total_bytes);

    double t0 = get_time_sec();
    double last_max_diff = 0.0;

    for (int step = 0; step < steps; step++) {
        double max_change = 0.0;
        int need_check = ((step + 1) % k_check == 0) || (step == steps - 1);

        for (int i = 1; i < n - 1; i++) {
            int row_idx = i * n;
            int up_idx  = (i - 1) * n;
            int dn_idx  = (i + 1) * n;

            for (int j = 1; j < n - 1; j++) {
                double u_old = curr[row_idx + j];
                double laplacian = curr[up_idx + j] + curr[dn_idx + j] + 
                                   curr[row_idx + j - 1] + curr[row_idx + j + 1] - 4.0 * u_old;
                double u_new = u_old + alpha * laplacian;
                next[row_idx + j] = u_new;

                if (need_check) {
                    double diff = fabs(u_new - u_old);
                    if (diff > max_change) max_change = diff;
                }
            }
        }

        if (need_check) {
            last_max_diff = max_change;
        }

        // Swap buffer pointers instead of copying memory
        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    double elapsed = get_time_sec() - t0;

    if (argc > 4) {
        FILE *fp = fopen(argv[4], "wb");
        if (!fp) return 1;
        size_t written = fwrite(curr, sizeof(*curr), (size_t)n * n, fp);
        if (fclose(fp) != 0 || written != (size_t)((size_t)n * n)) return 1;
    }

    // Checksum and L2 norm for verifying with parallel version
    double sum = 0.0;
    double l2 = 0.0;
    for (int i = 0; i < n * n; i++) {
        sum += curr[i];
        l2 += curr[i] * curr[i];
    }
    l2 = sqrt(l2);

    printf("SERIAL: N=%d, T=%d, K=%d, Time=%.6f s, Sum=%.8e, L2=%.8e, FinalMaxDiff=%.8e\n",
           n, steps, k_check, elapsed, sum, l2, last_max_diff);

    free(curr);
    free(next);
    return 0;
}
