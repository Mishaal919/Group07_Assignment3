#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifdef __cplusplus
#include <algorithm>
#endif

// CS3006 PDC Assignment 3 - Case Study 5: Distributed Sample Sort (Serial Baseline)

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline uint32_t next_splitmix32(uint32_t *state) {
    uint32_t z = (*state += 0x9e3779b9U);
    z = (z ^ (z >> 16)) * 0x85ebca6bU;
    z = (z ^ (z >> 13)) * 0xc2b2ae35U;
    return z ^ (z >> 16);
}

int compare_uint32(const void *a, const void *b) {
    uint32_t val_a = *(const uint32_t *)a;
    uint32_t val_b = *(const uint32_t *)b;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

int main(int argc, char **argv) {
    size_t num_keys = 50000000ULL; // 50M keys (200 MB)
    int dist_mode = 0; // 0 = uniform, 1 = skewed
    uint32_t seed = 42;

    if (argc > 1) num_keys = (size_t)atoll(argv[1]);
    if (argc > 2) dist_mode = atoi(argv[2]);

    if (num_keys == 0) return 1;
    printf("Serial Baseline (N=%zu, %s distribution)...\n",
           num_keys, (dist_mode == 0 ? "Uniform" : "Skewed"));

    uint32_t *data = (uint32_t *)malloc(num_keys * sizeof(uint32_t));
    if (!data) {
        fprintf(stderr, "Error: out of memory allocating %zu keys\n", num_keys);
        return 1;
    }

    uint32_t rng = seed;
    if (dist_mode == 0) {
        for (size_t i = 0; i < num_keys; i++) data[i] = next_splitmix32(&rng);
    } else {
        // Skewed: clustered towards lower numbers with duplicates
        for (size_t i = 0; i < num_keys; i++) {
            uint32_t r = next_splitmix32(&rng);
            uint64_t sq = (uint64_t)(r >> 16) * (uint64_t)(r >> 16);
            data[i] = (uint32_t)(sq & 0xFFFFFFFF);
        }
    }

    double t0 = get_time_sec();
#ifdef __cplusplus
    std::sort(data, data + num_keys);
#else
    qsort(data, num_keys, sizeof(uint32_t), compare_uint32);
#endif
    double elapsed = get_time_sec() - t0;

    // Verify sort
    int is_sorted = 1;
    for (size_t i = 1; i < num_keys; i++) {
        if (data[i] < data[i - 1]) {
            is_sorted = 0;
            break;
        }
    }

    double throughput = (double)num_keys / (elapsed * 1e6);
    printf("SERIAL RESULTS: Time=%.6f s, Throughput=%.2f M keys/s, Sorted=%s, Min=%u, Max=%u\n",
           elapsed, throughput, (is_sorted ? "PASSED" : "FAILED"), data[0], data[num_keys - 1]);

    if (argc > 3) {
        FILE *fp = fopen(argv[3], "wb");
        if (!fp) return 1;
        size_t written = fwrite(data, sizeof(*data), num_keys, fp);
        if (fclose(fp) != 0 || written != num_keys) return 1;
    }
    free(data);
    return !is_sorted;
}
