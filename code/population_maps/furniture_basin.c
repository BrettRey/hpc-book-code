/*
 * furniture_basin.c -- population-level map of an object-mass noun's
 * grammatical count status ("furniture": individuated referents, non-count
 * grammar). A didactic model from a linguistics book; not a model of data.
 *
 * Each speaker holds a grammatical count status g in [0,1]; constructions are
 * licensed by thresholds: a(n) g >= 0.90, three g >= 0.85, many g >= 0.50,
 * agreement g >= 0.30, much g <= 0.40. The population is a histogram over
 * NB = 40 bins of g (Q24 fixed point, total mass ONE). One phase:
 *   1. R = 3 observation rounds. A speaker's intent (one item, a small exact
 *      number, a vague many, existence, a mass quantity) has weights W. A
 *      one-item or exact-number intent the speaker can't license comes out
 *      as a measure phrase ("a piece of furniture"), heard as "much"; a
 *      vague-many intent below "many" falls to "much" if licensed. Those
 *      "much" uses come where counting was called for, so hearers also apply
 *      statistical preemption (rate PRE). "Much" from existence or mass
 *      intents carries no preemption. A hearer at g who hears a count
 *      construction with threshold t moves to g + LR*(t - g) if g < t; on
 *      "much" it moves to g - LR*(g - 0.2)*(1/2 + PRE if preempting).
 *      Bidirectional inference then pulls g toward the noun's meaning-side
 *      value SEM at rate KAPPA. Mass at a bin centre moves to the updated
 *      value, split linearly between the two nearest bin centres.
 *   2. Turnover: TAU = 0.04 of the population is replaced by learners whose
 *      g starts at LEARNER0 (they expect grammar to follow meaning, so g
 *      starts near 0.9) and who then hear LR_ROUNDS = 17 rounds.
 * The float version of this map matched an agent-based simulation of 100
 * speakers (mean g within 0.03 at every checked phase, with and without
 * preemption, out to 300 phases).
 */

#include <stdint.h>
#include <stdio.h>

#define NB 40
#define Q 24
#define ONE ((int64_t)1 << Q)
#define R 3
#define LR_ROUNDS 17
#define H 40   /* phases before the return check */
#define K 20   /* further phases that must also pass */

#include "furniture_params.h"  /* LEARNER0[NB], W_PRE05[5], W_PRE0[5], SEM_PRE05, SEM_PRE0 */

typedef struct {
    int64_t kappa;       /* coupling rate */
    int64_t sem;         /* meaning-side value coupling pulls toward */
    int64_t pre;         /* preemption rate */
    const int64_t *w;    /* intent weights */
    int64_t eq_mean;     /* reference equilibrium (preemption 0.5) */
    int64_t eq_frac[5];
} Params;

static const int64_t LR = 1006633;     /* 0.06 */
static const int64_t TAU = 671089;     /* 0.04 */
static const int64_t TH[4] = {15099494, 14260634, 8388608, 5033165}; /* 0.90 0.85 0.50 0.30 */
static const int64_t MUCH_MAX = 6710886;     /* 0.40 */
static const int64_t MUCH_TARGET = 3355443;  /* 0.20 */
static const int64_t DELTA = 436208;         /* 0.026 */
static const int64_t FRAC_TOL = 2013266;     /* 0.12 */
static const int64_t RADIUS = 3355443;       /* inputs: |mean - eq_mean| <= 0.20 */

/* Q24 product, rounded to nearest (ties away from zero). Every call has
 * |a|, |b| <= 2^25, so the 64-bit product cannot overflow. */
static int64_t mulq(int64_t a, int64_t b) {
    int64_t p = a * b;
    int64_t half = (int64_t)1 << (Q - 1);
    return p >= 0 ? (p + half) >> Q : -((-p + half) >> Q);
}

static int64_t centre(int i) { return ((int64_t)(2 * i + 1) * ONE) / (2 * NB); }

/* construction: 0 a(n), 1 three, 2 many, 3 agreement, 4 much, 5 much with preemption */
static int64_t updated(int64_t g, int c, const Params *pr) {
    if (c == 4) {
        g -= mulq(LR, g - MUCH_TARGET) / 2;
    } else if (c == 5) {
        g -= mulq(LR, g - MUCH_TARGET) / 2 + mulq(mulq(LR, pr->pre), g - MUCH_TARGET);
    } else if (g < TH[c]) {
        g += mulq(LR, TH[c] - g);
    }
    g += mulq(pr->kappa, pr->sem - g);
    if (g < 0) g = 0;
    if (g > ONE) g = ONE;
    return g;
}

static void apply(const int64_t h[NB], int c, int64_t weight, const Params *pr, int64_t out[NB]) {
    for (int i = 0; i < NB; i++) {
        if (h[i] == 0) continue;
        int64_t m = mulq(weight, h[i]);
        int64_t x = updated(centre(i), c, pr) * NB - ONE / 2;
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

/* share of mass with centre >= t (mode 0), <= t (mode 1), or < t (mode 2) */
static int64_t share(const int64_t h[NB], int64_t t, int mode) {
    int64_t s = 0, tot = total(h);
    for (int i = 0; i < NB; i++) {
        int64_t c = centre(i);
        int in = mode == 0 ? c >= t : (mode == 1 ? c <= t : c < t);
        if (in) s += h[i];
    }
    return (s << Q) / tot;
}

static void production(const int64_t h[NB], const Params *pr, int64_t p[6]) {
    const int64_t *w = pr->w;
    int64_t f90 = share(h, TH[0], 0), f85 = share(h, TH[1], 0), f50 = share(h, TH[2], 0);
    int64_t f30 = share(h, TH[3], 0), f40m = share(h, MUCH_MAX, 1), f30b = share(h, TH[3], 2);
    p[0] = mulq(w[0], f90);
    p[1] = mulq(w[1], f85);
    p[2] = mulq(w[2], f50);
    p[3] = mulq(w[3], f30);
    p[4] = mulq(w[3], f30b) + mulq(w[4], f40m);
    p[5] = mulq(w[0], ONE - f90) + mulq(w[1], ONE - f85) + mulq(w[2], f40m);
}

static void one_round(int64_t h[NB], const int64_t p[6], const Params *pr) {
    int64_t out[NB] = {0};
    int64_t silent = ONE - (p[0] + p[1] + p[2] + p[3] + p[4] + p[5]);
    for (int i = 0; i < NB; i++) out[i] += mulq(silent, h[i]);
    for (int c = 0; c < 6; c++)
        if (p[c] > 0) apply(h, c, p[c], pr, out);
    for (int i = 0; i < NB; i++) h[i] = out[i];
}

void phase_step(int64_t h[NB], const Params *pr) {
    int64_t p[6];
    for (int r = 0; r < R; r++) {
        production(h, pr, p);
        one_round(h, p, pr);
    }
    production(h, pr, p);
    int64_t L[NB];
    for (int i = 0; i < NB; i++) L[i] = LEARNER0[i];
    for (int r = 0; r < LR_ROUNDS; r++) one_round(L, p, pr);
    for (int i = 0; i < NB; i++) h[i] = mulq(ONE - TAU, h[i]) + mulq(TAU, L[i]);
}

int64_t mean_g(const int64_t h[NB]) {
    int64_t s = 0;
    for (int i = 0; i < NB; i++) s += h[i] * centre(i);
    return s / total(h);
}

void fractions(const int64_t h[NB], int64_t f[5]) {
    f[0] = share(h, TH[0], 0);
    f[1] = share(h, TH[1], 0);
    f[2] = share(h, TH[2], 0);
    f[3] = share(h, TH[3], 0);
    f[4] = share(h, MUCH_MAX, 1);
}

static int64_t absq(int64_t x) { return x < 0 ? -x : x; }

/* Admissible input: masses in [0, ONE] summing exactly to ONE, mean within
 * 0.20 of the reference equilibrium (pushed toward count or toward mass). */
int admissible(const int64_t h0[NB], const Params *pr) {
    for (int i = 0; i < NB; i++) if (h0[i] < 0 || h0[i] > ONE) return 0;
    if (total(h0) != ONE) return 0;
    return absq(mean_g(h0) - pr->eq_mean) <= RADIUS;
}

static int near_eq(const int64_t h[NB], const Params *pr) {
    if (absq(mean_g(h) - pr->eq_mean) > DELTA) return 0;
    int64_t f[5];
    fractions(h, f);
    for (int k = 0; k < 5; k++) if (absq(f[k] - pr->eq_frac[k]) > FRAC_TOL) return 0;
    return 1;
}

int returns_to_basin(const int64_t h0[NB], const Params *pr) {
    int64_t h[NB];
    for (int i = 0; i < NB; i++) h[i] = h0[i];
    for (int t = 1; t <= H + K; t++) {
        phase_step(h, pr);
        if (t >= H && !near_eq(h, pr)) return 0;
    }
    return 1;
}

/* Contrast: the mean never rises above the "much" threshold (0.40). */
int stays_noncount(const int64_t h0[NB], const Params *pr) {
    int64_t h[NB];
    for (int i = 0; i < NB; i++) h[i] = h0[i];
    for (int t = 1; t <= H + K; t++) {
        phase_step(h, pr);
        if (mean_g(h) > MUCH_MAX) return 0;
    }
    return 1;
}

#include "furniture_equilibrium.h"  /* EQ_MEAN_PRE05, EQ_FRAC_PRE05_INIT (see --equilibrium) */

const Params PRE05 = {167772, SEM_PRE05, ONE / 2, W_PRE05, EQ_MEAN_PRE05, EQ_FRAC_PRE05_INIT};
/* Contrast: preemption off. The reference equilibrium (and so the admissible
 * set) stays the preemption one; only the dynamics change. */
const Params PRE0 = {167772, SEM_PRE0, 0, W_PRE0, EQ_MEAN_PRE05, EQ_FRAC_PRE05_INIT};

int property_preemption_returns(const int64_t h0[NB]) {
    return !admissible(h0, &PRE05) || returns_to_basin(h0, &PRE05);
}
int property_no_preemption_stays_noncount(const int64_t h0[NB]) {
    return !admissible(h0, &PRE0) || stays_noncount(h0, &PRE0);
}

#ifdef FURNITURE_BASIN_MAIN
#include <string.h>
static void equilibrium(int64_t h[NB], const Params *pr) {
    for (int i = 0; i < NB; i++) h[i] = LEARNER0[i];
    for (int t = 0; t < 3000; t++) phase_step(h, pr);
}
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--equilibrium") == 0) {
        int64_t h[NB], f[5];
        equilibrium(h, &PRE05);
        fractions(h, f);
        printf("#define EQ_MEAN_PRE05 %lld\n", (long long)mean_g(h));
        printf("#define EQ_FRAC_PRE05_INIT {%lld, %lld, %lld, %lld, %lld}\n",
               (long long)f[0], (long long)f[1], (long long)f[2], (long long)f[3], (long long)f[4]);
        return 0;
    }
    if (argc > 2 && strcmp(argv[1], "--trace") == 0) {
        const Params *pr = strcmp(argv[2], "pre0") == 0 ? &PRE0 : &PRE05;
        int64_t h[NB];
        for (int i = 0; i < NB; i++) h[i] = LEARNER0[i];
        for (int t = 0; t < 80; t++) { phase_step(h, pr); printf("%.9f\n", (double)mean_g(h) / ONE); }
        return 0;
    }
    /* Witnesses: not vacuous. A push of 0.2 toward count from equilibrium is
     * admissible, leaves the basin, and returns; the contrast fails from the
     * equilibrium itself. */
    int64_t eq[NB], d[NB] = {0};
    equilibrium(eq, &PRE05);
    int big = 0;
    for (int i = 1; i < NB; i++) if (eq[i] > eq[big]) big = i;
    eq[big] += ONE - total(eq);
    for (int i = 0; i < NB; i++) d[i + 8 < NB ? i + 8 : NB - 1] += eq[i];
    int64_t h[NB];
    memcpy(h, d, sizeof h);
    phase_step(h, &PRE05);
    printf("push +0.2: admissible %d, out of basin after 1 phase %d, returns %d\n",
           admissible(d, &PRE05), !near_eq(h, &PRE05), returns_to_basin(d, &PRE05));
    printf("contrast from equilibrium: admissible %d, stays non-count %d\n",
           admissible(eq, &PRE0), stays_noncount(eq, &PRE0));
    return 0;
}
#endif
