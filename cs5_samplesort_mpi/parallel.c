#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

// CS3006 PDC Assignment 3 - Case Study 5: Distributed Sample Sort with MPI
// Pipeline phases:
// Phase 1: Local quicksort on local shard N/p
// Phase 2: Regular sampling (with oversample factor s), gather to root, choose splitters, broadcast
// Phase 3: Binary search multi-way partitioning, Alltoall for counts, Alltoallv for data
// Phase 4: Final local quicksort on received buckets
// Verification: local sortedness, global sortedness (max_i <= min_{i+1}), total count preserved

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

// Binary search to find partition split points in sorted array
static inline int find_bucket_split(const uint32_t *arr, int len, uint32_t splitter_key) {
    int low = 0, high = len;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] <= splitter_key) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    uint64_t total_keys = 50000000ULL;
    int dist_mode = 0; // 0 = uniform, 1 = skewed
    int oversample = 16;
    uint32_t base_seed = 42;
    const char *reference_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) total_keys = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) reference_path = argv[++i];
        else if (strcmp(argv[i], "--uniform") == 0) dist_mode = 0;
        else if (strcmp(argv[i], "--skewed") == 0) dist_mode = 1;
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) oversample = atoi(argv[++i]);
    }

    // MPI_Alltoallv uses int counts and offsets.
    if (total_keys < (uint64_t)num_ranks || total_keys > INT_MAX || oversample < 1 ||
        (uint64_t)num_ranks * (num_ranks - 1) * oversample > INT_MAX)
        MPI_Abort(MPI_COMM_WORLD, 1);
    uint64_t local_count = total_keys / num_ranks;
    uint64_t remainder = total_keys % num_ranks;
    if ((uint64_t)my_rank < remainder) local_count++;

    uint32_t *local_keys = (uint32_t *)malloc(local_count * sizeof(uint32_t));
    if (!local_keys) {
        fprintf(stderr, "Rank %d: out of memory allocating %llu keys\n", my_rank, (unsigned long long)local_count);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Generate the same set as the serial program, with one disjoint range per rank.
    uint64_t first = (uint64_t)my_rank * (total_keys / num_ranks) +
        ((uint64_t)my_rank < remainder ? (uint64_t)my_rank : remainder);
    uint32_t rng_state = base_seed + (uint32_t)first * 0x9e3779b9U;
    if (dist_mode == 0) {
        for (uint64_t i = 0; i < local_count; i++) local_keys[i] = next_splitmix32(&rng_state);
    } else {
        for (uint64_t i = 0; i < local_count; i++) {
            uint32_t r = next_splitmix32(&rng_state);
            uint64_t sq = (uint64_t)(r >> 16) * (uint64_t)(r >> 16);
            local_keys[i] = (uint32_t)(sq & 0xFFFFFFFF);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    // --- PHASE 1: LOCAL SORT ---
    double t_p1 = MPI_Wtime();
    qsort(local_keys, local_count, sizeof(uint32_t), compare_uint32);
    double time_local_sort = MPI_Wtime() - t_p1;

    // --- PHASE 2: REGULAR SAMPLING & SPLITTER SELECTION ---
    double t_p2 = MPI_Wtime();
    int samples_per_rank = (num_ranks > 1) ? (num_ranks - 1) * oversample : 1;
    uint32_t *local_samples = (uint32_t *)malloc(samples_per_rank * sizeof(uint32_t));
    for (int i = 0; i < samples_per_rank; i++) {
        uint64_t idx = (uint64_t)(i + 1) * local_count / (samples_per_rank + 1);
        if (idx >= local_count) idx = local_count - 1;
        local_samples[i] = local_keys[idx];
    }

    uint32_t *collected_samples = NULL;
    if (my_rank == 0) {
        collected_samples = (uint32_t *)malloc(num_ranks * samples_per_rank * sizeof(uint32_t));
    }

    MPI_Gather(local_samples, samples_per_rank, MPI_UINT32_T,
               collected_samples, samples_per_rank, MPI_UINT32_T,
               0, MPI_COMM_WORLD);

    uint32_t *splitters = (uint32_t *)malloc((num_ranks > 1 ? num_ranks - 1 : 1) * sizeof(uint32_t));
    if (my_rank == 0) {
        int total_sample_count = num_ranks * samples_per_rank;
        qsort(collected_samples, total_sample_count, sizeof(uint32_t), compare_uint32);
        for (int i = 0; i < num_ranks - 1; i++) {
            int idx = (int)((int64_t)(i + 1) * total_sample_count / num_ranks);
            splitters[i] = collected_samples[idx];
        }
        free(collected_samples);
    }

    if (num_ranks > 1) {
        MPI_Bcast(splitters, num_ranks - 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }
    double time_sampling = MPI_Wtime() - t_p2;
    free(local_samples);

    // --- PHASE 3: MULTI-WAY PARTITIONING AND ALLTOALLV ---
    double t_p3 = MPI_Wtime();
    int *send_counts = (int *)malloc(num_ranks * sizeof(int));
    int *send_displs = (int *)malloc(num_ranks * sizeof(int));
    int *recv_counts = (int *)malloc(num_ranks * sizeof(int));
    int *recv_displs = (int *)malloc(num_ranks * sizeof(int));

    int current_idx = 0;
    for (int i = 0; i < num_ranks - 1; i++) {
        int split_idx = find_bucket_split(local_keys, local_count, splitters[i]);
        send_counts[i] = split_idx - current_idx;
        send_displs[i] = current_idx;
        current_idx = split_idx;
    }
    send_counts[num_ranks - 1] = local_count - current_idx;
    send_displs[num_ranks - 1] = current_idx;

    // Exchange bucket sizes across all ranks
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_received = 0;
    for (int i = 0; i < num_ranks; i++) {
        recv_displs[i] = total_received;
        total_received += recv_counts[i];
    }

    uint32_t *received_keys = (uint32_t *)malloc((size_t)total_received * sizeof(uint32_t));
    if (!received_keys && total_received > 0) {
        fprintf(stderr, "Rank %d: out of memory allocating receive buffer (%d keys)\n", my_rank, total_received);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t_exchange = MPI_Wtime();
    // Exchange key payloads
    MPI_Alltoallv(local_keys, send_counts, send_displs, MPI_UINT32_T,
                  received_keys, recv_counts, recv_displs, MPI_UINT32_T,
                  MPI_COMM_WORLD);
    double time_comm = MPI_Wtime() - t_exchange;
    double time_partition = t_exchange - t_p3;

    // Free buffers no longer needed
    free(local_keys);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    free(splitters);

    // --- PHASE 4: FINAL LOCAL MERGE / SORT ---
    double t_p4 = MPI_Wtime();
    if (total_received > 1) {
        qsort(received_keys, total_received, sizeof(uint32_t), compare_uint32);
    }
    double time_merge = MPI_Wtime() - t_p4;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_elapsed = MPI_Wtime() - t_total_start;

    // --- VERIFICATION ---
    int local_ok = 1;
    for (int i = 1; i < total_received; i++) {
        if (received_keys[i] < received_keys[i - 1]) {
            local_ok = 0;
            break;
        }
    }

    if (reference_path) {
        uint64_t offset = 0;
        uint64_t count = total_received;
        MPI_Exscan(&count, &offset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank == 0) offset = 0;
        FILE *fp = fopen(reference_path, "rb");
        if (!fp || fseek(fp, (long)(offset * sizeof(uint32_t)), SEEK_SET))
            MPI_Abort(MPI_COMM_WORLD, 1);
        uint32_t buffer[1024];
        for (int i = 0; i < total_received; ) {
            size_t n = total_received - i < 1024 ? (size_t)(total_received - i) : 1024;
            if (fread(buffer, sizeof(*buffer), n, fp) != n ||
                memcmp(buffer, received_keys + i, n * sizeof(*buffer)) != 0) {
                local_ok = 0;
                break;
            }
            i += (int)n;
        }
        fclose(fp);
    }
    int all_local_ok = 0;
    MPI_Allreduce(&local_ok, &all_local_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    uint32_t my_min = (total_received > 0) ? received_keys[0] : 0;
    uint32_t my_max = (total_received > 0) ? received_keys[total_received - 1] : 0;

    uint32_t *all_mins = NULL;
    uint32_t *all_maxs = NULL;
    int *all_bucket_sizes = NULL;
    if (my_rank == 0) {
        all_mins = (uint32_t *)malloc(num_ranks * sizeof(uint32_t));
        all_maxs = (uint32_t *)malloc(num_ranks * sizeof(uint32_t));
        all_bucket_sizes = (int *)malloc(num_ranks * sizeof(int));
    }

    MPI_Gather(&my_min, 1, MPI_UINT32_T, all_mins, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_max, 1, MPI_UINT32_T, all_maxs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&total_received, 1, MPI_INT, all_bucket_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int global_order_ok = 1;
    uint64_t total_keys_counted = 0;
    int max_b = 0, min_b = 2147483647;

    if (my_rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            total_keys_counted += all_bucket_sizes[i];
            if (all_bucket_sizes[i] > max_b) max_b = all_bucket_sizes[i];
            if (all_bucket_sizes[i] < min_b) min_b = all_bucket_sizes[i];
        }
        int previous = -1;
        for (int i = 0; i < num_ranks; i++) {
            if (all_bucket_sizes[i] == 0) continue;
            if (previous >= 0 && all_maxs[previous] > all_mins[i]) global_order_ok = 0;
            previous = i;
        }
    }

    double max_t_total, max_t_local, max_t_samp, max_t_comm, max_t_merge, max_t_partition;
    MPI_Reduce(&total_elapsed, &max_t_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_local_sort, &max_t_local, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_sampling, &max_t_samp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_partition, &max_t_partition, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_comm, &max_t_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_merge, &max_t_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int passed = 0;
    if (my_rank == 0) {
        double mean_b = (double)total_keys / num_ranks;
        double imbalance_ratio = (double)max_b / (mean_b > 1e-9 ? mean_b : 1e-9);
        passed = (global_order_ok && (total_keys_counted == total_keys) && all_local_ok);

        printf("MPI SAMPLESORT: N=%llu, P=%d, Dist=%s, Oversample=%d, TotalTime=%.6f s (%.2f M keys/s)\n",
               (unsigned long long)total_keys, num_ranks, (dist_mode == 0 ? "Uniform" : "Skewed"),
               oversample, max_t_total, (double)total_keys / (max_t_total * 1e6));
        printf("  Phases: LocalSort=%.6f s (%.1f%%), Sampling=%.6f s (%.1f%%), Alltoallv=%.6f s (%.1f%%), Merge=%.6f s (%.1f%%)\n",
               max_t_local, (max_t_local / max_t_total) * 100.0,
               max_t_samp,  (max_t_samp / max_t_total) * 100.0,
               max_t_comm,  (max_t_comm / max_t_total) * 100.0,
               max_t_merge, (max_t_merge / max_t_total) * 100.0);
        printf("  Partition=%.6f s (includes count exchange and buffer allocation)\n", max_t_partition);
        printf("  LoadBalance: MinBucket=%d, MeanBucket=%.1f, MaxBucket=%d, Max/Mean=%.3f\n",
               min_b, mean_b, max_b, imbalance_ratio);
        printf("  BucketCounts:");
        for (int i = 0; i < num_ranks; i++) printf(" %d", all_bucket_sizes[i]);
        putchar('\n');
        if (reference_path) printf("  SerialMatch=%s\n", all_local_ok ? "YES" : "NO");
        printf("  Verification: GlobalSorted=%s, CountPreserved=%s (Collected=%llu vs Expected=%llu) -> %s\n",
               (global_order_ok && all_local_ok ? "YES" : "NO"),
               (total_keys_counted == total_keys ? "YES" : "NO"),
               (unsigned long long)total_keys_counted, (unsigned long long)total_keys,
               (passed ? "PASSED" : "FAILED"));

        free(all_mins);
        free(all_maxs);
        free(all_bucket_sizes);
    }

    if (received_keys) free(received_keys);
    MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return !passed;
}
