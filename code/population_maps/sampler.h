/* Random admissible-state generator shared by the two sampling harnesses.
 * Include after the model file. Draws a mix of shapes: two-bin mixtures, sparse
 * states over 3-7 random bins, and broad lumpy states, with masses summing
 * exactly to ONE; callers keep only states their `admissible` accepts. */
#include <stdlib.h>
#include <math.h>
static uint64_t rs = 88172645463325252ull;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double ur(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }
static int draw(int64_t h[NB]) {
    double w[NB] = {0};
    int kind = (int)(xr() % 3);
    /* v2: no single bins (40 states, already covered exhaustively by the
     * two-point-extreme checks); kind 0 is a random two-bin mixture. Broad
     * states are tilted by a random power of bin position, so that broad
     * spreads with high means (book's input set) are reachable. */
    if (kind == 0) { w[xr() % NB] = ur(); w[xr() % NB] += ur(); }
    else if (kind == 1) { int k = 3 + (int)(xr() % 5); for (int n = 0; n < k; n++) w[xr() % NB] += ur(); }
    else {
        double tilt = 12.0 * ur(), dir = ur() < 0.5 ? 1.0 : -1.0;
        for (int i = 0; i < NB; i++) {
            double u = ur(), x = (i + 0.5) / NB;
            w[i] = u * u * u * u * pow(dir > 0 ? x : 1.0 - x, tilt);
        }
    }
    double s = 0; for (int i = 0; i < NB; i++) s += w[i];
    if (s <= 0) return 0;
    int64_t acc = 0; int big = 0;
    for (int i = 0; i < NB; i++) { h[i] = (int64_t)(w[i] / s * ONE); acc += h[i]; if (h[i] > h[big]) big = i; }
    h[big] += ONE - acc;
    return kind;
}
