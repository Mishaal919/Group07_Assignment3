/*
 * Case Study 1 Bonus: Hybrid MPI + OpenMP Heat Diffusion Solver
 * Uses MPI across distributed processes and OpenMP multi-threading within each node.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    int T = 500;
    int K = 50;
    double alpha = 0.20;
    int num_threads = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-omp") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
    }

    omp_set_num_threads(num_threads);

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int nbr_up, nbr_down, nbr_left, nbr_right;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right);

    int local_rows = N / dims[0];
    int rem_rows = N % dims[0];
    int row_start = coords[0] * local_rows + (coords[0] < rem_rows ? coords[0] : rem_rows);
    if (coords[0] < rem_rows) local_rows++;

    int local_cols = N / dims[1];
    int rem_cols = N % dims[1];
    int col_start = coords[1] * local_cols + (coords[1] < rem_cols ? coords[1] : rem_cols);
    if (coords[1] < rem_cols) local_cols++;

    int stride = local_cols + 2;
    int total_cells = (local_rows + 2) * stride;
    double *u = (double *)calloc(total_cells, sizeof(double));
    double *u_next = (double *)calloc(total_cells, sizeof(double));

    for (int i = 1; i <= local_rows; i++) {
        int gi = row_start + i - 1;
        for (int j = 1; j <= local_cols; j++) {
            int gj = col_start + j - 1;
            double val = 0.0;
            if (gi == 0) val = 100.0;
            else if (gi == N - 1) val = 0.0;
            else if (gj == 0) val = 75.0;
            else if (gj == N - 1) val = 50.0;

            u[i * stride + j] = val;
            u_next[i * stride + j] = val;
        }
    }

    double *send_left  = (double *)malloc(local_rows * sizeof(double));
    double *recv_left  = (double *)malloc(local_rows * sizeof(double));
    double *send_right = (double *)malloc(local_rows * sizeof(double));
    double *recv_right = (double *)malloc(local_rows * sizeof(double));

    MPI_Barrier(cart_comm);
    double t_start = MPI_Wtime();
    double global_max_diff = 0.0;

    for (int step = 0; step < T; step++) {
        double step_max_diff = 0.0;
        int check_conv = ((step + 1) % K == 0) || (step == T - 1);

        /* Halo exchange */
        MPI_Sendrecv(&u[1 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 10,
                     &u[(local_rows + 1) * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 10,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[local_rows * stride + 1], local_cols, MPI_DOUBLE, nbr_down, 11,
                     &u[0 * stride + 1], local_cols, MPI_DOUBLE, nbr_up, 11,
                     cart_comm, MPI_STATUS_IGNORE);

        if (dims[1] > 1) {
            for (int i = 1; i <= local_rows; i++) {
                send_left[i - 1] = u[i * stride + 1];
                send_right[i - 1] = u[i * stride + local_cols];
            }
            MPI_Sendrecv(send_left, local_rows, MPI_DOUBLE, nbr_left, 20,
                         recv_right, local_rows, MPI_DOUBLE, nbr_right, 20,
                         cart_comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_right, local_rows, MPI_DOUBLE, nbr_right, 21,
                         recv_left, local_rows, MPI_DOUBLE, nbr_left, 21,
                         cart_comm, MPI_STATUS_IGNORE);
            if (nbr_right != MPI_PROC_NULL) {
                for (int i = 1; i <= local_rows; i++) u[i * stride + local_cols + 1] = recv_right[i - 1];
            }
            if (nbr_left != MPI_PROC_NULL) {
                for (int i = 1; i <= local_rows; i++) u[i * stride + 0] = recv_left[i - 1];
            }
        }

        /* Multithreaded OpenMP stencil computation */
        #pragma omp parallel for reduction(max:step_max_diff) schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            int gi = row_start + i - 1;
            if (gi == 0 || gi == N - 1) continue;
            for (int j = 1; j <= local_cols; j++) {
                int gj = col_start + j - 1;
                if (gj == 0 || gj == N - 1) continue;

                double u_curr = u[i * stride + j];
                double lap = u[(i - 1) * stride + j] + u[(i + 1) * stride + j] +
                             u[i * stride + j - 1] + u[i * stride + j + 1] - 4.0 * u_curr;
                double val = u_curr + alpha * lap;
                u_next[i * stride + j] = val;

                if (check_conv) {
                    double d = fabs(val - u_curr);
                    if (d > step_max_diff) step_max_diff = d;
                }
            }
        }

        if (check_conv) {
            MPI_Allreduce(&step_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        }

        double *tmp = u;
        u = u_next;
        u_next = tmp;
    }

    MPI_Barrier(cart_comm);
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    double local_sum = 0.0, local_l2_sq = 0.0;
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= local_cols; j++) {
            double v = u[i * stride + j];
            local_sum += v;
            local_l2_sq += v * v;
        }
    }
    double global_sum = 0.0, global_l2_sq = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_l2_sq, &global_l2_sq, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("HYBRID MPI+OpenMP: N=%d, T=%d, Ranks=%d, ThreadsPerRank=%d, TotalWorkers=%d, Time=%.6f s, Sum=%.8e, L2=%.8e, FinalMaxDiff=%.8e\n",
               N, T, size, num_threads, size * num_threads, max_elapsed, global_sum, sqrt(global_l2_sq), global_max_diff);
    }

    free(send_left);
    free(recv_left);
    free(send_right);
    free(recv_right);
    free(u);
    free(u_next);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
