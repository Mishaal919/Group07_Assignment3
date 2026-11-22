/*
 * Case Study 3 Bonus: Monte Carlo Option Pricing (MPI Version for Other-Model Comparison)
 */

#define _POSIX_C_SOURCE 199309L
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double splitmix64_to_double(uint64_t *state) {
    return (splitmix64_next(state) + 1.0) / 18446744073709551616.0;
}

static inline void box_muller(uint64_t *state, double *z0, double *z1) {
    double u1 = splitmix64_to_double(state);
    double u2 = splitmix64_to_double(state);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

static inline double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t M = 100000000ULL;
    double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    uint64_t base_seed = 123456789ULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) M = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) base_seed = (uint64_t)atoll(argv[++i]);
    }

    double bs_price = black_scholes_call(S0, K, r, sigma, T);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * sqrt(T);
    double discount = exp(-r * T);

    if (M < 2) MPI_Abort(MPI_COMM_WORLD, 1);
    uint64_t pairs = M / 2 + M % 2;
    uint64_t rem = pairs % size;
    uint64_t first = (uint64_t)rank * (pairs / size) + ((uint64_t)rank < rem ? (uint64_t)rank : rem);
    uint64_t count = pairs / size + ((uint64_t)rank < rem);
    uint64_t my_state = base_seed + 2 * first * 0x9e3779b97f4a7c15ULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double local_payoff = 0.0;
    double local_sq_payoff = 0.0;

    for (uint64_t i = 0; i < count; i++) {
        double z0, z1;
        box_muller(&my_state, &z0, &z1);

        double ST0 = S0 * exp(drift + vol * z0);
        double p0 = ST0 > K ? (ST0 - K) : 0.0;
        local_payoff += p0;
        local_sq_payoff += p0 * p0;

        if (2 * (first + i) + 1 < M) {
            double ST1 = S0 * exp(drift + vol * z1);
            double p1 = ST1 > K ? (ST1 - K) : 0.0;
            local_payoff += p1;
            local_sq_payoff += p1 * p1;
        }
    }

    double global_payoff = 0.0;
    double global_sq_payoff = 0.0;
    MPI_Reduce(&local_payoff, &global_payoff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_payoff, &global_sq_payoff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double mean_payoff = global_payoff / (double)M;
        double mc_price = discount * mean_payoff;
        double variance = (global_sq_payoff - (double)M * mean_payoff * mean_payoff) / (double)(M - 1);
        double std_dev = sqrt(variance > 0.0 ? variance : 0.0);
        double standard_error = discount * (std_dev / sqrt((double)M));
        double abs_error = fabs(mc_price - bs_price);

        printf("MPI: M=%llu, P=%d, Time=%.6f s, Throughput=%.2f M/s, Price=%.8f, SE=%.8f, BS=%.8f, AbsErr=%.8f\n",
               (unsigned long long)M, size, max_elapsed, (double)M / (max_elapsed * 1e6),
               mc_price, standard_error, bs_price, abs_error);
    }

    MPI_Finalize();
    return 0;
}
