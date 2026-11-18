#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

// CS3006 PDC Assignment 3 - Case Study 2: Genomics k-mer Counting with OpenMP
// Implements:
// - Version A: Intentional data race with omp parallel for and unprotected updates
// - Version B: Fixed using #pragma omp atomic
// - Version C: Fixed using private thread-local histograms and a parallel merge
// - False sharing test: adjacent unpadded counters vs 64-byte padded counters

static inline uint8_t base_to_bits(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0;
    }
}

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

// Demonstrates false sharing: tightly packed counters in the same 64-byte cache line
struct PackedCounter {
    volatile uint64_t count;
};

// Padded to 64 bytes so each thread gets its own independent cache line
struct CachePaddedCounter {
    volatile uint64_t count;
    char padding[56];
};

void run_false_sharing_experiment(int threads, size_t loops) {
    printf("--- Running False Sharing Test (%d threads, %zu increments) ---\n", threads, loops);
    struct PackedCounter *unpadded;
    struct CachePaddedCounter *padded;
    if (posix_memalign((void **)&unpadded, 64, threads * sizeof(*unpadded)) ||
        posix_memalign((void **)&padded, 64, threads * sizeof(*padded))) exit(1);
    memset(unpadded, 0, threads * sizeof(*unpadded));
    memset(padded, 0, threads * sizeof(*padded));

    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        for (size_t i = 0; i < loops; i++) {
            unpadded[tid].count++;
        }
    }
    double t_unpadded = omp_get_wtime() - t0;

    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        for (size_t i = 0; i < loops; i++) {
            padded[tid].count++;
        }
    }
    double t_padded = omp_get_wtime() - t1;

    printf("UNPADDED time: %.6f s\n", t_unpadded);
    printf("PADDED time: %.6f s\n", t_padded);
    printf("Padding speedup factor        : %.2fx\n\n", t_unpadded / (t_padded > 1e-9 ? t_padded : 1e-9));

    free(unpadded);
    free(padded);
}

int main(int argc, char **argv) {
    size_t seq_len = 100000000; // 10^8
    int k = 8;
    int num_threads = 4;
    char version = 'C'; // 'A', 'B', or 'C'
    int check_false_sharing = 0;
    unsigned int seed = 42;

    const char *dump_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump_path = argv[++i];
        else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) seq_len = (size_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) version = argv[++i][0];
        else if (strcmp(argv[i], "--falsesharing") == 0) check_false_sharing = 1;
    }

    // The fixed-width histogram uses two bits per base and 32-bit counters.
    if (k < 1 || k > 15 || num_threads < 1 || seq_len > UINT32_MAX ||
        (version != 'A' && version != 'B' && version != 'C')) return 1;
    omp_set_num_threads(num_threads);

    if (check_false_sharing) {
        run_false_sharing_experiment(num_threads, 100000000ULL);
        return 0;
    }

    size_t num_bins = (size_t)1 << (2 * k);
    size_t num_kmers = (seq_len >= (size_t)k) ? (seq_len - k + 1) : 0;
    uint32_t bitmask = (1U << (2 * k)) - 1U;

    char *dna = make_random_dna(seq_len, seed);
    if (!dna) {
        fprintf(stderr, "Error: out of memory allocating sequence\n");
        return 1;
    }

    uint32_t *global_histogram = (uint32_t *)calloc(num_bins, sizeof(uint32_t));
    if (!global_histogram) {
        fprintf(stderr, "Error: out of memory allocating global histogram\n");
        free(dna);
        return 1;
    }

    double t_start = omp_get_wtime();

    if (version == 'A') {
        // VERSION A: Deliberate data race!
        // No lock, no atomic, threads race on the same bins.
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (size_t t = 0; t < (size_t)num_threads; t++) {
            size_t chunk_start = t * num_kmers / num_threads;
            size_t chunk_end   = (t + 1) * num_kmers / num_threads;
            if (chunk_start >= chunk_end) continue;

            uint32_t kmer = 0;
            for (int j = 0; j < k - 1; j++) {
                kmer = ((kmer << 2) | base_to_bits(dna[chunk_start + j])) & bitmask;
            }
            for (size_t i = chunk_start; i < chunk_end; i++) {
                kmer = ((kmer << 2) | base_to_bits(dna[i + k - 1])) & bitmask;
                // BUG ON PURPOSE: Unsynchronized concurrent write
                global_histogram[kmer]++;
            }
        }
    } else if (version == 'B') {
        // VERSION B: Atomic updates
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (size_t t = 0; t < (size_t)num_threads; t++) {
            size_t chunk_start = t * num_kmers / num_threads;
            size_t chunk_end   = (t + 1) * num_kmers / num_threads;
            if (chunk_start >= chunk_end) continue;

            uint32_t kmer = 0;
            for (int j = 0; j < k - 1; j++) {
                kmer = ((kmer << 2) | base_to_bits(dna[chunk_start + j])) & bitmask;
            }
            for (size_t i = chunk_start; i < chunk_end; i++) {
                kmer = ((kmer << 2) | base_to_bits(dna[i + k - 1])) & bitmask;
                #pragma omp atomic
                global_histogram[kmer]++;
            }
        }
    } else {
        // VERSION C: Private histograms per thread
        uint32_t **thread_histograms = (uint32_t **)malloc(num_threads * sizeof(uint32_t *));
        for (int t = 0; t < num_threads; t++) {
            thread_histograms[t] = (uint32_t *)calloc(num_bins, sizeof(uint32_t));
            if (!thread_histograms[t]) {
                fprintf(stderr, "Error: out of memory allocating private histogram for thread %d\n", t);
                return 1;
            }
        }

        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int tid = 0; tid < num_threads; tid++) {
            size_t chunk_start = (size_t)tid * num_kmers / num_threads;
            size_t chunk_end   = (size_t)(tid + 1) * num_kmers / num_threads;
            uint32_t *my_hist  = thread_histograms[tid];

            if (chunk_start < chunk_end) {
                // Initialize rolling window starting at chunk_start
                uint32_t kmer = 0;
                for (int j = 0; j < k - 1; j++) {
                    kmer = ((kmer << 2) | base_to_bits(dna[chunk_start + j])) & bitmask;
                }
                for (size_t i = chunk_start; i < chunk_end; i++) {
                    kmer = ((kmer << 2) | base_to_bits(dna[i + k - 1])) & bitmask;
                    my_hist[kmer]++; // Thread writes to its own private memory
                }
            }
        }

        // Parallel merge of all thread histograms into global histogram
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (size_t b = 0; b < num_bins; b++) {
            uint32_t sum = 0;
            for (int t = 0; t < num_threads; t++) {
                sum += thread_histograms[t][b];
            }
            global_histogram[b] = sum;
        }

        for (int t = 0; t < num_threads; t++) free(thread_histograms[t]);
        free(thread_histograms);
    }

    double elapsed = omp_get_wtime() - t_start;

    if (dump_path) {
        FILE *fp = fopen(dump_path, "wb");
        if (!fp) return 1;
        size_t written = fwrite(global_histogram, sizeof(*global_histogram), num_bins, fp);
        if (fclose(fp) != 0 || written != (size_t)(num_bins)) return 1;
    }

    // Check results
    uint64_t total_count = 0;
    uint32_t max_bin = 0;
    for (size_t b = 0; b < num_bins; b++) {
        total_count += global_histogram[b];
        if (global_histogram[b] > max_bin) max_bin = global_histogram[b];
    }

    int64_t lost_kmers = (int64_t)num_kmers - (int64_t)total_count;
    double throughput = (double)num_kmers / (elapsed * 1e6);

    printf("OPENMP: Ver=%c, L=%zu, k=%d, P=%d, Time=%.6f s, Throughput=%.2f M k-mers/s, TotalCount=%llu, Expected=%zu, LostCount=%lld (%s), MaxBin=%u\n",
           version, seq_len, k, num_threads, elapsed, throughput,
           (unsigned long long)total_count, num_kmers, (long long)lost_kmers,
           (lost_kmers == 0 ? "EXACT MATCH" : "DATA RACE ERROR"),
           max_bin);

    free(dna);
    free(global_histogram);
    return 0;
}
