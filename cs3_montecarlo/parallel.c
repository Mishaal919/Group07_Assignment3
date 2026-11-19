#define _POSIX_C_SOURCE 199309L
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CS3006 PDC Assignment 3 - Case Study 3: Monte Carlo Option Pricing (OpenMP)
// Features:
// - Independent SplitMix64 PRNG stream per thread (reproducible)
// - Parallel reduction across threads for payoff and squared payoff
// - Diagnostic mode for shared RNG with locks (showing contention slowdown)
// - Diagnostic mode for same seed (showing correlated identical streams)

static inline uint64_t next_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double uniform_double(uint64_t *state) {
    return (next_splitmix64(state) + 1.0) / 18446744073709551616.0;
}

static inline void get_two_normals(uint64_t *state, double *z0, double *z1) {
    double u1 = uniform_double(state);
    double u2 = uniform_double(state);
    double radius = sqrt(-2.0 * log(u1));
    double angle  = 2.0 * M_PI * u2;
    *z0 = radius * cos(angle);
    *z1 = radius * sin(angle);
}

static inline double std_normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double black_scholes_exact(double s0, double k, double r, double sigma, double t) {
    double d1 = (log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t));
    double d2 = d1 - sigma * sqrt(t);
    return s0 * std_normal_cdf(d1) - k * exp(-r * t) * std_normal_cdf(d2);
}

int main(int argc, char **argv) {
    uint64_t num_samples = 100000000ULL;
    int num_threads = 4;
    double s0 = 100.0;
    double k = 100.0;
    double r = 0.05;
    double sigma = 0.20;
    double t = 1.0;
    uint64_t base_seed = 123456789ULL;

    int test_shared_lock = 0;
    int test_same_seed = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) num_samples = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) base_seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--shared-rng") == 0) test_shared_lock = 1;
        else if (strcmp(argv[i], "--same-seed") == 0) test_same_seed = 1;
    }

    if (num_threads < 1 || num_samples < 2) return 1;
    omp_set_num_threads(num_threads);
    double *payoffs = calloc(num_threads, sizeof(*payoffs));
    double *squares = calloc(num_threads, sizeof(*squares));
    if (!payoffs || !squares) {
        free(payoffs);
        free(squares);
        return 1;
    }

    double bs_price = black_scholes_exact(s0, k, r, sigma, t);
    double drift = (r - 0.5 * sigma * sigma) * t;
    double vol = sigma * sqrt(t);
    double discount_factor = exp(-r * t);

    double total_payoff = 0.0;
    double total_sq_payoff = 0.0;

    uint64_t shared_state = base_seed;
    omp_lock_t lock;
    if (test_shared_lock) omp_init_lock(&lock);

    double t0 = omp_get_wtime();

    if (test_shared_lock) {
        // Critical Analysis test: All threads serialize through a single shared RNG protected by a lock
        #pragma omp parallel for reduction(+:total_payoff, total_sq_payoff) schedule(static)
        for (uint64_t i = 0; i < num_samples; i += 2) {
            double z0, z1;
            omp_set_lock(&lock);
            get_two_normals(&shared_state, &z0, &z1);
            omp_unset_lock(&lock);

            double st0 = s0 * exp(drift + vol * z0);
            double p0 = st0 > k ? (st0 - k) : 0.0;
            total_payoff += p0;
            total_sq_payoff += p0 * p0;

            if (i + 1 < num_samples) {
                double st1 = s0 * exp(drift + vol * z1);
                double p1 = st1 > k ? (st1 - k) : 0.0;
                total_payoff += p1;
                total_sq_payoff += p1 * p1;
            }
        }
    } else {
        #pragma omp parallel
        {
            uint64_t tid = omp_get_thread_num();
            uint64_t workers = omp_get_num_threads();
            uint64_t pairs = num_samples / 2 + num_samples % 2;
            uint64_t first = tid * (pairs / workers) + (tid < pairs % workers ? tid : pairs % workers);
            uint64_t count = pairs / workers + (tid < pairs % workers);
            uint64_t offset = test_same_seed ? 0 : 2 * first;
            uint64_t my_state = base_seed + offset * 0x9e3779b97f4a7c15ULL;

            double local_payoff = 0.0;
            double local_sq = 0.0;
            for (uint64_t i = 0; i < count; i++) {
                double z0, z1;
                get_two_normals(&my_state, &z0, &z1);
                double st0 = s0 * exp(drift + vol * z0);
                double p0 = st0 > k ? st0 - k : 0.0;
                local_payoff += p0;
                local_sq += p0 * p0;
                if (2 * (first + i) + 1 < num_samples) {
                    double st1 = s0 * exp(drift + vol * z1);
                    double p1 = st1 > k ? st1 - k : 0.0;
                    local_payoff += p1;
                    local_sq += p1 * p1;
                }
            }
            payoffs[tid] = local_payoff;
            squares[tid] = local_sq;
        }
        // Reduce in worker order so the same team size gives the same result.
        for (int i = 0; i < num_threads; i++) {
            total_payoff += payoffs[i];
            total_sq_payoff += squares[i];
        }
    }

    double elapsed = omp_get_wtime() - t0;
    if (test_shared_lock) omp_destroy_lock(&lock);
    free(payoffs);
    free(squares);

    double mean_payoff = total_payoff / (double)num_samples;
    double mc_price = discount_factor * mean_payoff;
    double variance = (total_sq_payoff - (double)num_samples * mean_payoff * mean_payoff) / (double)(num_samples - 1);
    double std_dev = sqrt(variance > 0.0 ? variance : 0.0);
    double std_err = discount_factor * (std_dev / sqrt((double)num_samples));
    double abs_error = fabs(mc_price - bs_price);

    printf("OPENMP: Mode=%s, M=%llu, P=%d, Time=%.6f s, Throughput=%.2f M/s, Price=%.8f, SE=%.8f, BS=%.8f, AbsErr=%.8f\n",
           (test_shared_lock ? "SharedLockRNG" : (test_same_seed ? "SameCorrelatedSeed" : "IndependentStreams")),
           (unsigned long long)num_samples, num_threads, elapsed, (double)num_samples / (elapsed * 1e6),
           mc_price, std_err, bs_price, abs_error);

    return 0;
}
