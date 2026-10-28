#include "book_basin.c"
#include "sampler.h"
int main(int argc, char **argv) {
    long target = argc > 1 ? atol(argv[1]) : 100000;
    struct { const char *name; int (*prop)(const int64_t *); const Params *pr; } P[2] = {
        {"book, no coupling  (property_semantic_k0)", property_semantic_k0, &SEMANTIC_K0},
        {"book, coupling 0.01 (property_semantic_k001)", property_semantic_k001, &SEMANTIC_K001}};
    for (int p = 0; p < 2; p++) {
        long tested = 0, drawn = 0, failed = 0, by_kind[3] = {0};
        while (tested < target) {
            int64_t h[NB]; int kind = draw(h); drawn++;
            if (!admissible(h, P[p].pr)) continue;
            tested++; by_kind[kind]++;
            if (!P[p].prop(h)) { failed++; if (failed <= 3) { printf("  FAIL:"); for (int i = 0; i < NB; i++) if (h[i]) printf(" [%d]=%lld", i, (long long)h[i]); printf("\n"); } }
        }
        printf("%s: %ld admissible states tested (two-bin %ld, sparse %ld, broad %ld; %ld drawn), %ld failed\n",
               P[p].name, tested, by_kind[0], by_kind[1], by_kind[2], drawn, failed);
    }
    return 0;
}
