// STREAM-style copy test. The two arrays together are much larger than cache.
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t n = 16 * 1024 * 1024;
    double *src = malloc(n * sizeof(*src));
    double *dst = malloc(n * sizeof(*dst));
    if (!src || !dst) {
        free(src);
        free(dst);
        return 1;
    }
    double times[5];
    #pragma omp parallel for num_threads(4) schedule(static)
    for (size_t i = 0; i < n; i++) src[i] = (double)i;
    for (int trial = -1; trial < 5; trial++) {
        double start = omp_get_wtime();
        #pragma omp parallel for num_threads(4) schedule(static)
        for (size_t i = 0; i < n; i++) dst[i] = src[i];
        if (trial >= 0) times[trial] = omp_get_wtime() - start;
        double *tmp = src;
        src = dst;
        dst = tmp;
    }
    for (int i = 1; i < 5; i++) {
        for (int j = i; j > 0 && times[j] < times[j - 1]; j--) {
            double tmp = times[j];
            times[j] = times[j - 1];
            times[j - 1] = tmp;
        }
    }
    printf("Copy GB/s=%.3f (median of 5, 4 threads), check=%.0f\n",
           2.0 * n * sizeof(*src) / (times[2] * 1e9), src[n - 1]);
    free(src);
    free(dst);
    return 0;
}
