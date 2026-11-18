#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// CS3006 PDC Assignment 3 - Case Study 1: Heat Diffusion with MPI
// Supports:
// 1. 1D row decomposition or 2D Cartesian grid (MPI_Cart_create)
// 2. Blocking (Sendrecv) or Non-blocking (Isend/Irecv with interior/boundary overlap)
// 3. Periodic convergence check via MPI_Allreduce every K steps

#define COMM_BLOCKING 0
#define COMM_NONBLOCKING 1
#define DECOMP_1D 1
#define DECOMP_2D 2

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Default problem parameters
    int n = 1024;
    int steps = 500;
    int k_check = 50;
    int decomp = DECOMP_2D;
    int comm_mode = COMM_NONBLOCKING;
    double alpha = 0.20;
    const char *reference_path = NULL;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) k_check = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) reference_path = argv[++i];
        else if (strcmp(argv[i], "--1d") == 0) decomp = DECOMP_1D;
        else if (strcmp(argv[i], "--2d") == 0) decomp = DECOMP_2D;
        else if (strcmp(argv[i], "--blocking") == 0) comm_mode = COMM_BLOCKING;
        else if (strcmp(argv[i], "--nonblocking") == 0) comm_mode = COMM_NONBLOCKING;
    }

    // Set up Cartesian grid topology
    int dims[2] = {0, 0};
    int periods[2] = {0, 0}; // Non-periodic grid (fixed boundary temperatures)
    MPI_Comm cart_comm;

    if (decomp == DECOMP_1D) {
        dims[0] = num_ranks; // All ranks slice along rows
        dims[1] = 1;
    } else {
        MPI_Dims_create(num_ranks, 2, dims); // Automatically pick best balanced grid (e.g. 2x2)
    }

    if (n < 3 || n < dims[0] || n < dims[1] || steps < 1 || k_check < 1) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, my_rank, 2, coords);

    // Find our neighbors: up/down along dim 0, left/right along dim 1
    int nbr_up, nbr_down, nbr_left, nbr_right;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right);

    // Calculate our local subgrid size (handling remainder if n doesn't divide cleanly)
    int rows_per_rank = n / dims[0];
    int rem_rows = n % dims[0];
    int row_start = coords[0] * rows_per_rank + (coords[0] < rem_rows ? coords[0] : rem_rows);
    int local_rows = rows_per_rank + (coords[0] < rem_rows ? 1 : 0);

    int cols_per_rank = n / dims[1];
    int rem_cols = n % dims[1];
    int col_start = coords[1] * cols_per_rank + (coords[1] < rem_cols ? coords[1] : rem_cols);
    int local_cols = cols_per_rank + (coords[1] < rem_cols ? 1 : 0);

    // Allocate local grid with 1-cell halo padding around all edges
    int stride = local_cols + 2;
    int local_total_cells = (local_rows + 2) * stride;
    double *curr = (double *)calloc(local_total_cells, sizeof(double));
    double *next = (double *)calloc(local_total_cells, sizeof(double));

    // Initialize physical boundary values if this rank touches the outer boundary
    for (int i = 1; i <= local_rows; i++) {
        int global_i = row_start + i - 1;
        for (int j = 1; j <= local_cols; j++) {
            int global_j = col_start + j - 1;
            double val = 0.0;
            if (global_i == 0) val = 100.0;
            else if (global_i == n - 1) val = 0.0;
            else if (global_j == 0) val = 75.0;
            else if (global_j == n - 1) val = 50.0;

            curr[i * stride + j] = val;
            next[i * stride + j] = val;
        }
    }

    // Buffers for packing/unpacking column ghost cells (since columns are strided in memory)
    double *send_left_col  = (double *)malloc(local_rows * sizeof(double));
    double *recv_left_col  = (double *)malloc(local_rows * sizeof(double));
    double *send_right_col = (double *)malloc(local_rows * sizeof(double));
    double *recv_right_col = (double *)malloc(local_rows * sizeof(double));

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();
    double global_max_diff = 0.0;
    double wait_time_acc = 0.0;

    for (int step = 0; step < steps; step++) {
        double step_diff = 0.0;
        int need_check = ((step + 1) % k_check == 0) || (step == steps - 1);
        double t_comm_start = MPI_Wtime();

        if (comm_mode == COMM_BLOCKING) {
            // Blocking halo exchange: send top/bottom rows
            MPI_Sendrecv(&curr[1 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 101,
                         &curr[(local_rows + 1) * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 101,
                         cart_comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&curr[local_rows * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 102,
                         &curr[0 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 102,
                         cart_comm, MPI_STATUS_IGNORE);

            // Send left/right columns if decomposed across 2D
            if (dims[1] > 1) {
                for (int i = 1; i <= local_rows; i++) {
                    send_left_col[i - 1]  = curr[i * stride + 1];
                    send_right_col[i - 1] = curr[i * stride + local_cols];
                }
                MPI_Sendrecv(send_left_col, local_rows, MPI_DOUBLE, nbr_left, 201,
                             recv_right_col, local_rows, MPI_DOUBLE, nbr_right, 201,
                             cart_comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(send_right_col, local_rows, MPI_DOUBLE, nbr_right, 202,
                             recv_left_col, local_rows, MPI_DOUBLE, nbr_left, 202,
                             cart_comm, MPI_STATUS_IGNORE);
                if (nbr_right != MPI_PROC_NULL) {
                    for (int i = 1; i <= local_rows; i++) curr[i * stride + local_cols + 1] = recv_right_col[i - 1];
                }
                if (nbr_left != MPI_PROC_NULL) {
                    for (int i = 1; i <= local_rows; i++) curr[i * stride + 0] = recv_left_col[i - 1];
                }
            }
            wait_time_acc += (MPI_Wtime() - t_comm_start);

            // Compute entire local grid (interior + boundaries)
            for (int i = 1; i <= local_rows; i++) {
                int gi = row_start + i - 1;
                if (gi == 0 || gi == n - 1) continue; // Dirichlet boundary
                for (int j = 1; j <= local_cols; j++) {
                    int gj = col_start + j - 1;
                    if (gj == 0 || gj == n - 1) continue;

                    double u_old = curr[i * stride + j];
                    double lap = curr[(i - 1) * stride + j] + curr[(i + 1) * stride + j] +
                                 curr[i * stride + j - 1] + curr[i * stride + j + 1] - 4.0 * u_old;
                    double u_new = u_old + alpha * lap;
                    next[i * stride + j] = u_new;

                    if (need_check) {
                        double diff = fabs(u_new - u_old);
                        if (diff > step_diff) step_diff = diff;
                    }
                }
            }
        } else {
            // Non-blocking halo exchange: overlap interior computation with communication
            MPI_Request reqs[8];
            int nreq = 0;

            // Post receives first to avoid buffering
            MPI_Irecv(&curr[(local_rows + 1) * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 101, cart_comm, &reqs[nreq++]);
            MPI_Irecv(&curr[0 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 102, cart_comm, &reqs[nreq++]);

            // Post sends
            MPI_Isend(&curr[1 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 101, cart_comm, &reqs[nreq++]);
            MPI_Isend(&curr[local_rows * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 102, cart_comm, &reqs[nreq++]);

            if (dims[1] > 1) {
                for (int i = 1; i <= local_rows; i++) {
                    send_left_col[i - 1]  = curr[i * stride + 1];
                    send_right_col[i - 1] = curr[i * stride + local_cols];
                }
                MPI_Irecv(recv_right_col, local_rows, MPI_DOUBLE, nbr_right, 201, cart_comm, &reqs[nreq++]);
                MPI_Irecv(recv_left_col, local_rows, MPI_DOUBLE, nbr_left, 202, cart_comm, &reqs[nreq++]);
                MPI_Isend(send_left_col, local_rows, MPI_DOUBLE, nbr_left, 201, cart_comm, &reqs[nreq++]);
                MPI_Isend(send_right_col, local_rows, MPI_DOUBLE, nbr_right, 202, cart_comm, &reqs[nreq++]);
            }

            wait_time_acc += MPI_Wtime() - t_comm_start;

            // OVERLAP COMPUTATION: Calculate strictly interior cells (rows 2..local_rows-1, cols 2..local_cols-1)
            // These points only look at neighboring cells within this rank, so they don't need halos!
            for (int i = 2; i <= local_rows - 1; i++) {
                int gi = row_start + i - 1;
                if (gi <= 0 || gi >= n - 1) continue;
                for (int j = 2; j <= local_cols - 1; j++) {
                    int gj = col_start + j - 1;
                    if (gj <= 0 || gj >= n - 1) continue;

                    double u_old = curr[i * stride + j];
                    double lap = curr[(i - 1) * stride + j] + curr[(i + 1) * stride + j] +
                                 curr[i * stride + j - 1] + curr[i * stride + j + 1] - 4.0 * u_old;
                    double u_new = u_old + alpha * lap;
                    next[i * stride + j] = u_new;

                    if (need_check) {
                        double diff = fabs(u_new - u_old);
                        if (diff > step_diff) step_diff = diff;
                    }
                }
            }

            // Now wait for halos to finish transferring
            double t_wait_start = MPI_Wtime();
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);


            if (dims[1] > 1) {
                if (nbr_right != MPI_PROC_NULL) {
                    for (int i = 1; i <= local_rows; i++) curr[i * stride + local_cols + 1] = recv_right_col[i - 1];
                }
                if (nbr_left != MPI_PROC_NULL) {
                    for (int i = 1; i <= local_rows; i++) curr[i * stride + 0] = recv_left_col[i - 1];
                }
            }

            wait_time_acc += MPI_Wtime() - t_wait_start;

            // Now compute boundary rows (row 1 and row local_rows)
            int boundary_rows[2] = {1, local_rows};
            for (int r = 0; r < 2; r++) {
                int i = boundary_rows[r];
                if (i < 1 || i > local_rows) continue;
                int gi = row_start + i - 1;
                if (gi <= 0 || gi >= n - 1) continue;
                for (int j = 1; j <= local_cols; j++) {
                    int gj = col_start + j - 1;
                    if (gj <= 0 || gj >= n - 1) continue;

                    double u_old = curr[i * stride + j];
                    double lap = curr[(i - 1) * stride + j] + curr[(i + 1) * stride + j] +
                                 curr[i * stride + j - 1] + curr[i * stride + j + 1] - 4.0 * u_old;
                    double u_new = u_old + alpha * lap;
                    next[i * stride + j] = u_new;

                    if (need_check) {
                        double diff = fabs(u_new - u_old);
                        if (diff > step_diff) step_diff = diff;
                    }
                }
            }

            // And compute boundary columns (col 1 and col local_cols, skipping already computed corner rows)
            for (int i = 2; i <= local_rows - 1; i++) {
                int gi = row_start + i - 1;
                if (gi <= 0 || gi >= n - 1) continue;
                int boundary_cols[2] = {1, local_cols};
                for (int c = 0; c < 2; c++) {
                    int j = boundary_cols[c];
                    if (j < 1 || j > local_cols) continue;
                    int gj = col_start + j - 1;
                    if (gj <= 0 || gj >= n - 1) continue;

                    double u_old = curr[i * stride + j];
                    double lap = curr[(i - 1) * stride + j] + curr[(i + 1) * stride + j] +
                                 curr[i * stride + j - 1] + curr[i * stride + j + 1] - 4.0 * u_old;
                    double u_new = u_old + alpha * lap;
                    next[i * stride + j] = u_new;

                    if (need_check) {
                        double diff = fabs(u_new - u_old);
                        if (diff > step_diff) step_diff = diff;
                    }
                }
            }
        }

        // Global convergence reduction
        if (need_check) {
            MPI_Allreduce(&step_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        }

        // Pointer swap
        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    MPI_Barrier(cart_comm);
    double elapsed_local = MPI_Wtime() - t_start;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    // Compute checksums for verification against serial baseline
    double local_sum = 0.0;
    double local_l2_sq = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++) {
            double val = curr[i * stride + j];
            local_sum += val;
            local_l2_sq += val * val;
        }
    }

    double total_sum = 0.0, total_l2_sq = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_l2_sq, &total_l2_sq, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    int verification_failed = 0;
    if (reference_path) {
        FILE *fp = fopen(reference_path, "rb");
        double *row = malloc((size_t)local_cols * sizeof(*row));
        if (!fp || !row) MPI_Abort(cart_comm, 1);
        double local_error = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            long offset = ((long)(row_start + i - 1) * n + col_start) * (long)sizeof(double);
            if (fseek(fp, offset, SEEK_SET) || fread(row, sizeof(*row), local_cols, fp) != (size_t)local_cols)
                MPI_Abort(cart_comm, 1);
            for (int j = 1; j <= local_cols; j++)
                local_error = fmax(local_error, fabs(row[j - 1] - curr[i * stride + j]));
        }
        free(row);
        fclose(fp);
        double error;
        MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        verification_failed = error > 1e-10;
        if (my_rank == 0) printf("GridMaxAbsError=%.12g\n", error);
    }

    double max_wait;
    MPI_Reduce(&wait_time_acc, &max_wait, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    if (my_rank == 0) {
        printf("MPI: N=%d, T=%d, K=%d, P=%d, Grid=(%dx%d), Decomp=%s, Mode=%s, Time=%.6f s, CommWait=%.6f s, Sum=%.8e, L2=%.8e, FinalMaxDiff=%.8e\n",
               n, steps, k_check, num_ranks, dims[0], dims[1],
               (decomp == DECOMP_1D ? "1D" : "2D"),
               (comm_mode == COMM_BLOCKING ? "Blocking" : "Non-blocking"),
               elapsed_max, max_wait, total_sum, sqrt(total_l2_sq), global_max_diff);
    }

    free(send_left_col);
    free(recv_left_col);
    free(send_right_col);
    free(recv_right_col);
    free(curr);
    free(next);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return verification_failed;
}
