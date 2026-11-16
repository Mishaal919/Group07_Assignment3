#define _POSIX_C_SOURCE 199309L
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CS3006 PDC Assignment 3 - Case Study 3: Monte Carlo Option Pricing (Serial Baseline)

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 64-bit SplitMix generator: fast, simple, passes BigCrush statistical tests
static inline uint64_t next_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Convert 64-bit integer to uniform double in (0, 1)
static inline double uniform_double(uint64_t *state) {
    return (next_splitmix64(state) + 1.0) / 18446744073709551616.0;
}

// Box-Muller transform: converts 2 uniform numbers into 2 standard normals Z0, Z1 ~ N(0, 1)
static inline void get_two_normals(uint64_t *state, double *z0, double *z1) {
    double u1 = uniform_double(state);
    double u2 = uniform_double(state);
    double radius = sqrt(-2.0 * log(u1));
    double angle  = 2.0 * M_PI * u2;
    *z0 = radius * cos(angle);
    *z1 = radius * sin(angle);
}

// Cumulative standard normal distribution function
static inline double std_normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Analytical Black-Scholes formula for European call option (our ground truth)
double black_scholes_exact(double s0, double k, double r, double sigma, double t) {
    double d1 = (log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t));
    double d2 = d1 - sigma * sqrt(t);
    return s0 * std_normal_cdf(d1) - k * exp(-r * t) * std_normal_cdf(d2);
}

int main(int argc, char **argv) {
    uint64_t num_samples = 100000000ULL; // 10^8 samples
    double s0 = 100.0;
    double k = 100.0;
    double r = 0.05;
    double sigma = 0.20;
    double t = 1.0;
    uint64_t seed = 123456789ULL;

    if (argc > 1) num_samples = (uint64_t)atoll(argv[1]);
    if (argc > 2) seed = (uint64_t)atoll(argv[2]);

    double bs_price = black_scholes_exact(s0, k, r, sigma, t);
    double drift = (r - 0.5 * sigma * sigma) * t;
    double vol = sigma * sqrt(t);
    double discount_factor = exp(-r * t);

    printf("MONTE CARLO OPTION PRICING (SERIAL BASELINE)\n");
    printf("Params: S0=%.2f, K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, M=%llu\n",
           s0, k, r, sigma, t, (unsigned long long)num_samples);
    printf("Black-Scholes Closed-Form Price: %.8f\n", bs_price);

    uint64_t rng_state = seed;
    double sum_payoff = 0.0;
    double sum_sq_payoff = 0.0;

    double t0 = get_time_sec();

    // Generate paths in pairs of 2 using Box-Muller
    for (uint64_t i = 0; i < num_samples; i += 2) {
        double z0, z1;
        get_two_normals(&rng_state, &z0, &z1);

        double st0 = s0 * exp(drift + vol * z0);
        double payoff0 = st0 > k ? (st0 - k) : 0.0;
        sum_payoff += payoff0;
        sum_sq_payoff += payoff0 * payoff0;

        if (i + 1 < num_samples) {
            double st1 = s0 * exp(drift + vol * z1);
            double payoff1 = st1 > k ? (st1 - k) : 0.0;
            sum_payoff += payoff1;
            sum_sq_payoff += payoff1 * payoff1;
        }
    }

    double elapsed = get_time_sec() - t0;

    // Calculate sample mean, variance, and standard error of estimator
    double mean_payoff = sum_payoff / (double)num_samples;
    double mc_price = discount_factor * mean_payoff;
    double variance = (sum_sq_payoff - (double)num_samples * mean_payoff * mean_payoff) / (double)(num_samples - 1);
    double std_dev = sqrt(variance > 0.0 ? variance : 0.0);
    double std_err = discount_factor * (std_dev / sqrt((double)num_samples));
    double abs_error = fabs(mc_price - bs_price);

    printf("SERIAL RESULTS:\n");
    printf("  Time taken       : %.6f s (%.2f M samples/s)\n", elapsed, (double)num_samples / (elapsed * 1e6));
    printf("  Estimated Price  : %.8f\n", mc_price);
    printf("  Standard Error   : %.8f\n", std_err);
    printf("  Exact BS Price   : %.8f\n", bs_price);
    printf("  Abs Error vs BS  : %.8f (%.2f standard errors)\n", abs_error, abs_error / std_err);

    return 0;
}
