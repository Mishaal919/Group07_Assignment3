#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// CS3006 PDC Assignment 3 - Case Study 2: Genomics k-mer Counting (Serial Baseline)

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Map characters to 2 bits: A=00, C=01, G=10, T=11
static inline uint8_t base_to_bits(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0;
    }
}

// Generate random DNA sequence with deterministic seed
char *make_random_dna(size_t len, unsigned int seed) {
    char *seq = (char *)malloc(len + 1);
    if (!seq) return NULL;
    const char bases[4] = {'A', 'C', 'G', 'T'};
    srand(seed);
    for (size_t i = 0; i < len; i++) {
        seq[i] = bases[rand() % 4];
    }
    seq[len] = '\0';
    return seq;
}

int main(int argc, char **argv) {
    size_t seq_len = 100000000; // 10^8 characters default
    int k = 8;                  // k=8 or k=12
    unsigned int seed = 42;

    if (argc > 1) seq_len = (size_t)atoll(argv[1]);
    if (argc > 2) k = atoi(argv[2]);
    if (argc > 3) seed = (unsigned int)atoi(argv[3]);

    if (k < 1 || k > 15 || seq_len > UINT32_MAX) return 1;
    size_t num_bins = (size_t)1 << (2 * k); // 4^k bins
    size_t num_kmers = (seq_len >= (size_t)k) ? (seq_len - k + 1) : 0;

    printf("Generating synthetic sequence (L=%zu, k=%d, 4^k bins=%zu)...\n", seq_len, k, num_bins);
    char *dna = make_random_dna(seq_len, seed);
    if (!dna) {
        fprintf(stderr, "Error: out of memory allocating sequence\n");
        return 1;
    }

    uint32_t *histogram = (uint32_t *)calloc(num_bins, sizeof(uint32_t));
    if (!histogram) {
        fprintf(stderr, "Error: out of memory allocating histogram\n");
        free(dna);
        return 1;
    }

    uint32_t bitmask = (1U << (2 * k)) - 1U;
    double t_start = get_time_sec();

    // Prime the rolling window with the first k-1 characters
    uint32_t current_kmer = 0;
    for (int j = 0; j < k - 1; j++) {
        current_kmer = ((current_kmer << 2) | base_to_bits(dna[j])) & bitmask;
    }

    // Slide window across sequence: shift in 2 bits and mask top bits
    for (size_t i = 0; i < num_kmers; i++) {
        current_kmer = ((current_kmer << 2) | base_to_bits(dna[i + k - 1])) & bitmask;
        histogram[current_kmer]++;
    }

    double elapsed = get_time_sec() - t_start;

    // Check that total sum equals expected L - k + 1
    uint64_t total_counted = 0;
    uint32_t max_bin_val = 0;
    for (size_t b = 0; b < num_bins; b++) {
        total_counted += histogram[b];
        if (histogram[b] > max_bin_val) max_bin_val = histogram[b];
    }

    double rate = (double)num_kmers / (elapsed * 1e6);
    if (argc > 4) {
        FILE *fp = fopen(argv[4], "wb");
        if (!fp) return 1;
        size_t written = fwrite(histogram, sizeof(*histogram), num_bins, fp);
        if (fclose(fp) != 0 || written != (size_t)(num_bins)) return 1;
    }

    printf("SERIAL: L=%zu, k=%d, Time=%.6f s, Throughput=%.2f M k-mers/s, TotalCount=%llu (Match=%s), MaxBin=%u\n",
           seq_len, k, elapsed, rate,
           (unsigned long long)total_counted,
           (total_counted == num_kmers ? "YES" : "NO"),
           max_bin_val);

    free(dna);
    free(histogram);
    return 0;
}
