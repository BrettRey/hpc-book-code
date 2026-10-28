#include "furniture_basin.c"
#include "sampler.h"
int main(int argc, char **argv) {
    long target = argc > 1 ? atol(argv[1]) : 100000;
    long tested = 0, drawn = 0, failed = 0, by_kind[3] = {0}, above = 0, below = 0;
    while (tested < target) {
        int64_t h[NB]; int kind = draw(h); drawn++;
        if (!admissible(h, &PRE05)) continue;
        tested++; by_kind[kind]++;
        if (mean_g(h) > PRE05.eq_mean) above++; else below++;
        if (!property_preemption_returns(h)) { failed++; if (failed <= 3) { printf("  FAIL:"); for (int i = 0; i < NB; i++) if (h[i]) printf(" [%d]=%lld", i, (long long)h[i]); printf("\n"); } }
    }
    printf("furniture, preemption 0.5 (property_preemption_returns): %ld admissible states tested (two-bin %ld, sparse %ld, broad %ld; %ld pushed toward count, %ld toward mass; %ld drawn), %ld failed\n",
           tested, by_kind[0], by_kind[1], by_kind[2], above, below, drawn, failed);
    return 0;
}
