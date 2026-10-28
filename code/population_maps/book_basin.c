/*
 * book_basin.c -- population-level map of one count noun's grammatical status.
 *
 * A didactic model from a linguistics book about how grammatical categories
 * are maintained. Each speaker holds a grammatical count status g in [0,1]
 * for the noun ("book"); constructions are licensed by thresholds on g:
 *   a(n) g >= 0.90, three g >= 0.85, many g >= 0.50, agreement g >= 0.30,
 *   much g <= 0.40.
 * The population is a histogram over NB = 40 equal bins of g (mass in Q24
 * fixed point, total mass ONE). One phase:
 *   1. R = 3 observation rounds. In a round, the probability that a hearer
 *      hears each construction is fixed by the current histogram (speakers'
 *      intents: one item 0.38, a small exact number 0.25, a vague many 0.22,
 *      existence 0.13, a mass quantity 0.02; each speaker uses the first
 *      construction its g licenses, else falls back to "much" if licensed,
 *      else says nothing). A hearer at g who hears a count construction with
 *      threshold t moves to g + LR*(t - g) if g < t (and stays otherwise); on
 *      "much" it moves to g - LR*(g - 0.2)/2. Then bidirectional inference pulls g toward the
 *      noun's semantic individuation SEM = 0.95 at rate KAPPA. Mass at a bin
 *      centre moves to the updated value, split linearly between the two
 *      nearest bin centres.
 *   2. Turnover: a fraction TAU = 0.04 of the population is replaced by
 *      learners. A learner's starting histogram is LEARNER0 (see below); it
 *      then hears LR_ROUNDS = 17 rounds from the current population.
 * Learner priors (the mechanism under test):
 *   semantic:   learners expect grammar to follow meaning; g starts ~ clamped
 *               Normal(0.95, sd 0.1414), i.e. their sense of the referents.
 *   uninformed: g starts ~ clamped Normal(0.50, sd 0.10).
 * LEARNER0 tables are those distributions binned (computed with erf, then
 * rounded to Q24 so each sums exactly to ONE).
 *
 * The map was checked against an agent-based simulation of 100 speakers
 * (mean g within 0.03 at every checked phase in four settings).
 */

#include <stdint.h>
#include <stdio.h>

#define NB 40
#define Q 24
#define ONE ((int64_t)1 << Q)
#define R 3
#define LR_ROUNDS 17
#define H 100   /* phases before the return check */
#define K 20    /* further phases that must also pass */

typedef struct {
    int64_t kappa;           /* coupling rate, Q24 */
    const int64_t *learner0; /* learner starting histogram, Q24, sums to ONE */
    int64_t eq_mean;         /* the map's own equilibrium mean, Q24 */
    int64_t eq_frac[5];      /* equilibrium licensing fractions, Q24 */
} Params;

static const int64_t LR = 1006633;     /* 0.06  */
static const int64_t TAU = 671089;     /* 0.04  */
static const int64_t SEM = 15938355;   /* 0.95  */
static const int64_t TH[4] = {15099494, 14260634, 8388608, 5033165}; /* 0.90 0.85 0.50 0.30 */
static const int64_t MUCH_MAX = 6710886;     /* 0.40 */
static const int64_t MUCH_TARGET = 3355443;  /* 0.20 */
static const int64_t W_SINGLE = 6375342, W_EXACT = 4194304, W_VAGUE = 3690988,
                     W_EXIST = 2181038, W_MASS = 335544; /* 0.38 0.25 0.22 0.13 0.02 */
static const int64_t DELTA = 436208;     /* 0.026 */
static const int64_t FRAC_TOL = 2013266; /* 0.12  */
static const int64_t MEAN_FLOOR = 3355443; /* inputs: mean >= eq_mean - 0.20 */

#include "learner_tables.h"  /* LEARNER0_SEMANTIC[NB], LEARNER0_UNINFORMED[NB] */

/* Q24 product, rounded to nearest (ties away from zero). Every call in this
 * file has |a|, |b| <= 2^25, so the 64-bit product cannot overflow. */
static int64_t mulq(int64_t a, int64_t b) {
    int64_t p = a * b;
    int64_t half = (int64_t)1 << (Q - 1);
    return p >= 0 ? (p + half) >> Q : -((-p + half) >> Q);
}

static int64_t centre(int i) { return ((int64_t)(2 * i + 1) * ONE) / (2 * NB); }

/* construction index: 0 a(n), 1 three, 2 many, 3 agreement, 4 much */
static int64_t updated(int64_t g, int c, int64_t kappa) {
    if (c == 4) {
        g -= mulq(LR, g - MUCH_TARGET) / 2;
    } else if (g < TH[c]) {
        g += mulq(LR, TH[c] - g);
    }
    g += mulq(kappa, SEM - g);
    if (g < 0) g = 0;
    if (g > ONE) g = ONE;
    return g;
}

/* out += weight * (M_c applied to h); mass-conserving split */
static void apply(const int64_t h[NB], int c, int64_t weight, int64_t kappa, int64_t out[NB]) {
    for (int i = 0; i < NB; i++) {
        if (h[i] == 0) continue;
        int64_t m = mulq(weight, h[i]);
        int64_t x = updated(centre(i), c, kappa) * NB - ONE / 2; /* position in bin units, Q24 */
        int lo;
        int64_t frac;
        if (x <= 0) { lo = 0; frac = 0; }
        else {
            lo = (int)(x >> Q);
            frac = x - ((int64_t)lo << Q);
            if (lo >= NB - 1) { lo = NB - 1; frac = 0; }
        }
        int hi = lo + 1 < NB ? lo + 1 : NB - 1;
        int64_t up = mulq(m, frac);
        out[lo] += m - up;
        out[hi] += up;
    }
}

static int64_t total(const int64_t h[NB]) {
    int64_t s = 0;
    for (int i = 0; i < NB; i++) s += h[i];
    return s;
}

static int64_t frac_where(const int64_t h[NB], int64_t t, int below_or_equal, int strict_below) {
    int64_t s = 0, tot = total(h);
    for (int i = 0; i < NB; i++) {
        int64_t c = centre(i);
        int in = strict_below ? (c < t) : (below_or_equal ? (c <= t) : (c >= t));
        if (in) s += h[i];
    }
    return (s << Q) / tot;
}

static void production(const int64_t h[NB], int64_t p[5]) {
    int64_t f90 = frac_where(h, TH[0], 0, 0), f85 = frac_where(h, TH[1], 0, 0);
    int64_t f50 = frac_where(h, TH[2], 0, 0), f30 = frac_where(h, TH[3], 0, 0);
    int64_t f40m = frac_where(h, MUCH_MAX, 1, 0), f30b = frac_where(h, TH[3], 0, 1);
    p[0] = mulq(W_SINGLE, f90);
    p[1] = mulq(W_EXACT, f85);
    p[2] = mulq(W_VAGUE, f50);
    p[3] = mulq(W_EXIST, f30);
    p[4] = mulq(W_SINGLE + W_EXACT + W_VAGUE + W_MASS, f40m) + mulq(W_EXIST, f30b);
}

static void one_round(int64_t h[NB], const int64_t p[5], int64_t kappa) {
    int64_t out[NB] = {0};
    int64_t silent = ONE - (p[0] + p[1] + p[2] + p[3] + p[4]);
    for (int i = 0; i < NB; i++) out[i] += mulq(silent, h[i]);
    for (int c = 0; c < 5; c++)
        if (p[c] > 0) apply(h, c, p[c], kappa, out);
    for (int i = 0; i < NB; i++) h[i] = out[i];
}

void phase_step(int64_t h[NB], const Params *pr) {
    int64_t p[5];
    for (int r = 0; r < R; r++) {
        production(h, p);
        one_round(h, p, pr->kappa);
    }
    production(h, p);
    int64_t L[NB];
    for (int i = 0; i < NB; i++) L[i] = pr->learner0[i];
    for (int r = 0; r < LR_ROUNDS; r++) one_round(L, p, pr->kappa);
    for (int i = 0; i < NB; i++) h[i] = mulq(ONE - TAU, h[i]) + mulq(TAU, L[i]);
}

int64_t mean_g(const int64_t h[NB]) {
    int64_t s = 0;
    for (int i = 0; i < NB; i++) s += h[i] * centre(i);
    return s / total(h);
}

void fractions(const int64_t h[NB], int64_t f[5]) {
    f[0] = frac_where(h, TH[0], 0, 0);
    f[1] = frac_where(h, TH[1], 0, 0);
    f[2] = frac_where(h, TH[2], 0, 0);
    f[3] = frac_where(h, TH[3], 0, 0);
    f[4] = frac_where(h, MUCH_MAX, 1, 0);
}

/* Admissible input: non-negative masses summing exactly to ONE, mean at least
 * eq_mean - 0.20. */
int admissible(const int64_t h0[NB], const Params *pr) {
    for (int i = 0; i < NB; i++) if (h0[i] < 0 || h0[i] > ONE) return 0; /* also rules out overflow */
    if (total(h0) != ONE) return 0;
    return mean_g(h0) >= pr->eq_mean - MEAN_FLOOR;
}

static int64_t absq(int64_t x) { return x < 0 ? -x : x; }

static int near_eq(const int64_t h[NB], const Params *pr) {
    if (absq(mean_g(h) - pr->eq_mean) > DELTA) return 0;
    int64_t f[5];
    fractions(h, f);
    for (int k = 0; k < 5; k++) if (absq(f[k] - pr->eq_frac[k]) > FRAC_TOL) return 0;
    return 1;
}

/* Return property: from h0, after H phases and for K further phases, the mean
 * is within DELTA of equilibrium and every licensing fraction within FRAC_TOL. */
int returns_to_basin(const int64_t h0[NB], const Params *pr) {
    int64_t h[NB];
    for (int i = 0; i < NB; i++) h[i] = h0[i];
    for (int t = 1; t <= H + K; t++) {
        phase_step(h, pr);
        if (t >= H && !near_eq(h, pr)) return 0;
    }
    return 1;
}

/* Tight-frame property used as the contrast: a(n) licensing never falls below
 * half of the population during H + K phases. */
int keeps_tight_frames(const int64_t h0[NB], const Params *pr) {
    int64_t h[NB];
    for (int i = 0; i < NB; i++) h[i] = h0[i];
    for (int t = 1; t <= H + K; t++) {
        phase_step(h, pr);
        int64_t f[5];
        fractions(h, f);
        if (f[0] < ONE / 2) return 0;
    }
    return 1;
}

#include "equilibria.h"  /* EQ_* constants computed by this program (see --equilibria) */

const Params SEMANTIC_K0 = {0, LEARNER0_SEMANTIC, EQ_MEAN_SEM_K0, EQ_FRAC_SEM_K0_INIT};
const Params SEMANTIC_K001 = {167772, LEARNER0_SEMANTIC, EQ_MEAN_SEM_K001, EQ_FRAC_SEM_K001_INIT};
/* The contrast keeps the semantic-learner equilibrium as its reference, so its
 * admissible inputs are the same spreads around a count-core equilibrium; only
 * the learners' prior changes. */
const Params UNINFORMED_K001 = {167772, LEARNER0_UNINFORMED, EQ_MEAN_SEM_K001, EQ_FRAC_SEM_K001_INIT};

/* The three properties to verify. */
int property_semantic_k0(const int64_t h0[NB]) {
    return !admissible(h0, &SEMANTIC_K0) || returns_to_basin(h0, &SEMANTIC_K0);
}
int property_semantic_k001(const int64_t h0[NB]) {
    return !admissible(h0, &SEMANTIC_K001) || returns_to_basin(h0, &SEMANTIC_K001);
}
/* Contrast: uninformed learners, starting from any admissible spread around
 * the semantic equilibrium. Expected to fail. */
int property_uninformed_keeps_tight_frames(const int64_t h0[NB]) {
    return !admissible(h0, &UNINFORMED_K001) || keeps_tight_frames(h0, &UNINFORMED_K001);
}

#ifdef BOOK_BASIN_MAIN
#include <stdlib.h>
#include <string.h>
static void equilibrium(int64_t h[NB], const Params *pr) {
    for (int i = 0; i < NB; i++) h[i] = pr->learner0[i];
    for (int t = 0; t < 3000; t++) phase_step(h, pr);
}
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--equilibria") == 0) {
        const Params *ps[2] = {&SEMANTIC_K0, &SEMANTIC_K001};
        const char *nm[2] = {"SEM_K0", "SEM_K001"};
        for (int k = 0; k < 2; k++) {
            int64_t h[NB], f[5];
            equilibrium(h, ps[k]);
            fractions(h, f);
            printf("#define EQ_MEAN_%s %lld\n", nm[k], (long long)mean_g(h));
            printf("#define EQ_FRAC_%s_INIT {%lld, %lld, %lld, %lld, %lld}\n", nm[k],
                   (long long)f[0], (long long)f[1], (long long)f[2], (long long)f[3], (long long)f[4]);
        }
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--trace") == 0) {
        /* trace k prior: mean after each of 80 phases from learner0, for self-test against Python */
        const Params *pr = argc > 2 && strcmp(argv[2], "uninformed") == 0 ? &UNINFORMED_K001
                         : (argc > 3 && strcmp(argv[3], "k001") == 0 ? &SEMANTIC_K001 : &SEMANTIC_K0);
        int64_t h[NB];
        for (int i = 0; i < NB; i++) h[i] = pr->learner0[i];
        for (int t = 0; t < 80; t++) { phase_step(h, pr); printf("%.9f\n", (double)mean_g(h) / ONE); }
        return 0;
    }
    /* Witnesses: the property is not vacuous. A 0.1 drop of equilibrium is
     * admissible and leaves the basin at phase 1, and still returns. */
    const Params *ps[2] = {&SEMANTIC_K0, &SEMANTIC_K001};
    for (int k = 0; k < 2; k++) {
        int64_t eq[NB], d[NB] = {0};
        equilibrium(eq, ps[k]);
        for (int i = 0; i < NB; i++) d[i - 4 < 0 ? 0 : i - 4] += eq[i];
        /* rounding leaves the total a few units off ONE; correct the largest bin */
        int big = 0;
        for (int i = 1; i < NB; i++) if (d[i] > d[big]) big = i;
        d[big] += ONE - total(d);
        int64_t h[NB];
        memcpy(h, d, sizeof h);
        phase_step(h, ps[k]);
        printf("params %d: drop admissible %d, out of basin after 1 phase %d, returns %d\n",
               k, admissible(d, ps[k]), !near_eq(h, ps[k]), returns_to_basin(d, ps[k]));
    }
    int64_t eq[NB];
    equilibrium(eq, &SEMANTIC_K0);
    int64_t s = total(eq);
    eq[NB - 1] += ONE - s;
    printf("contrast: semantic equilibrium admissible for uninformed params %d, keeps tight frames %d\n",
           admissible(eq, &UNINFORMED_K001), keeps_tight_frames(eq, &UNINFORMED_K001));
    return 0;
}
#endif
