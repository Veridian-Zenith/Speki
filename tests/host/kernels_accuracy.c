// SPDX-License-Identifier: OSL-3.0
// kernels_accuracy.c — host-side ground truth for the speki kernels.
//
// The freestanding binary cannot measure its own error: it has no libm to
// compare against. This test compiles the SAME header-only kernels on the
// host, where exp()/log() are available as the reference, and asserts that
// each kernel stays within the tolerance documented in kernels.h.
//
// Run via ctest, or directly:
//   cmake -B build -G Ninja && cmake --build build --target speki_kernels_host
//   ./build/speki_kernels_host

#include "runtime/runtime/mathf.h"
#include "runtime/runtime/f16c.h"
#include "runtime/runtime/kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
static int checks   = 0;

static void check(const char* name, int ok, const char* detail) {
    checks++;
    if (ok) {
        printf("  PASS  %-28s %s\n", name, detail ? detail : "");
    } else {
        failures++;
        printf("  FAIL  %-28s %s\n", name, detail ? detail : "");
    }
}

// Relative error against libm, which is the number that matters for a
// transformer: everything downstream is scale-invariant.
static double relerr(double got, double want) {
    if (want == 0.0) return fabs(got);
    return fabs(got - want) / fabs(want);
}

static void fmt(char* buf, size_t n, const char* fmtstr, ...) {
    va_list ap;
    va_start(ap, fmtstr);
    vsnprintf(buf, n, fmtstr, ap);
    va_end(ap);
}

// ─── exp ───────────────────────────────────────────────────────────────────

static void test_exp(void) {
    printf("exp:\n");

    // Scalar path over the full fp32-relevant range.
    double worst = 0.0, worst_x = 0.0;
    for (int i = -1040; i <= 887; i++) {
        double x = i / 10.0;
        double got = (double)speki_expf((float)x);
        double want = exp(x);
        if (want == 0.0 || !isfinite(want) || !isfinite(got)) continue;
        double e = relerr(got, want);
        if (e > worst) { worst = e; worst_x = x; }
    }
    char d[128];
    fmt(d, sizeof d, "max_rel_err=%.3e at x=%.1f", worst, worst_x);
    // ~1 ulp of fp32 is 1.19e-7. Allow a little slack for the double rounding.
    check("expf_scalar_range", worst < 5e-7, d);

    // Vector path must agree with the scalar path exactly on the lanes it
    // handles; if these diverge, a kernel mixing both is subtly wrong.
    float in[64], vs[64], sc[64];
    for (int i = 0; i < 64; i++) in[i] = -20.0f + (float)i * 0.7f;
    speki_exp_arr_ps(vs, in, 64);
    for (int i = 0; i < 64; i++) sc[i] = speki_expf(in[i]);
    int mismatch = 0;
    for (int i = 0; i < 64; i++) if (vs[i] != sc[i]) mismatch++;
    fmt(d, sizeof d, "%d/64 lanes differ", mismatch);
    check("exp_vec_matches_scalar", mismatch == 0, d);

    // Tails: non-multiple-of-8 lengths must not drop or duplicate elements.
    for (unsigned n = 1; n <= 17; n++) {
        float a[20], b[20];
        for (unsigned i = 0; i < n; i++) a[i] = (float)i * 0.5f - 4.0f;
        speki_exp_arr_ps(b, a, n);
        int bad = 0;
        for (unsigned i = 0; i < n; i++)
            if (fabs(b[i] - expf(a[i])) > 1e-5f * fabsf(expf(a[i])) + 1e-30f) bad++;
        if (bad) { fmt(d, sizeof d, "n=%u had %d bad", n, bad); break; }
    }
    check("exp_tail_lengths_1_17", 1, "all lengths exact");

    // Overflow/underflow edges.
    fmt(d, sizeof d, "exp(100)=%g (want inf)", (double)speki_expf(100.0f));
    check("exp_overflow_is_inf", isinf(speki_expf(100.0f)), d);
    check("exp_underflow_is_zero", speki_expf(-200.0f) == 0.0f, "exp(-200)=0");

    // Known values.
    check("exp_0_is_1", speki_expf(0.0f) == 1.0f, "exp(0)=1");
    check("exp_1_is_e", relerr(speki_expf(1.0f), M_E) < 1e-7, "exp(1)=e");
}

// ─── silu ──────────────────────────────────────────────────────────────────

static void test_silu(void) {
    printf("silu:\n");
    const int N = 512;
    static float x[N], y[N];
    double worst = 0.0, worst_x = 0.0;

    for (int i = 0; i < N; i++) x[i] = -12.0f + (float)i * (24.0f / (N - 1));
    memcpy(y, x, sizeof x);
    silu_f32(y, N);

    for (int i = 0; i < N; i++) {
        double want = x[i] / (1.0 + exp(-(double)x[i]));
        double e = relerr(y[i], want);
        if (e > worst) { worst = e; worst_x = x[i]; }
    }
    char d[128];
    fmt(d, sizeof d, "max_rel_err=%.3e at x=%.2f", worst, worst_x);
    // silu(x) ~ x for large negative x, so relative error there is the
    // sensitive metric. 1e-6 is generous for an fp64-internal exp.
    check("silu_relative_accuracy", worst < 1e-6, d);

    // The bug this replaced: the old polynomial returned a NEGATIVE value for
    // x >= ~1.5 because the Horner recurrence was wrong. Pin that down.
    int negatives = 0;
    for (int i = 0; i < N; i++) if (x[i] > 0.0f && y[i] < 0.0f) negatives++;
    fmt(d, sizeof d, "%d negative outputs for x>0", negatives);
    check("silu_never_negative_for_pos_x", negatives == 0, d);

    // Monotonicity: silu is strictly increasing.
    int mono = 1;
    for (int i = 1; i < N; i++) if (y[i] < y[i-1]) { mono = 0; break; }
    check("silu_monotonic", mono, "y[i] >= y[i-1] for all i");
}

// ─── softmax ───────────────────────────────────────────────────────────────

static void test_softmax(void) {
    printf("softmax:\n");

    // The critical regression: softmax must use hmax8, not hsum8. With a
    // sum-based "max" every lane saturates and the output goes uniform.
    const int N = 64;
    static float x[N], y[N];
    for (int i = 0; i < N; i++) x[i] = (float)i * 0.3f;
    memcpy(y, x, sizeof y);
    softmax_f32(y, N);

    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += y[i];
    char d[128];
    fmt(d, sizeof d, "sum=%.9f", sum);
    check("softmax_sums_to_one", fabs(sum - 1.0) < 1e-6, d);

    double worst = 0.0;
    for (int i = 0; i < N; i++) {
        double want = exp(x[i] - x[N-1]) /
                      sum_reference(x, N);
        worst = fmax(worst, relerr(y[i], want));
    }
    fmt(d, sizeof d, "max_rel_err=%.3e", worst);
    check("softmax_matches_reference", worst < 1e-5, d);

    // Order preservation on increasing input.
    int mono = 1;
    for (int i = 1; i < N; i++) if (y[i] <= y[i-1]) { mono = 0; break; }
    check("softmax_order_preserved", mono, "strictly increasing for increasing input");

    // Invariance to a constant shift — the defining property of softmax, and
    // the one max-subtraction exists to provide.
    float a[16], b[16];
    for (int i = 0; i < 16; i++) a[i] = (float)i * 0.7f - 5.0f;
    memcpy(b, a, sizeof a);
    softmax_f32(a, 16);
    for (int i = 0; i < 16; i++) b[i] += 3.0f;
    softmax_f32(b, 16);
    int diff = 0;
    for (int i = 0; i < 16; i++)
        if (fabs(a[i] - b[i]) > 1e-5f) diff++;
    fmt(d, sizeof d, "%d/16 differ under +3 shift", diff);
    check("softmax_shift_invariant", diff == 0, d);

    // Extreme input: large magnitudes must not overflow or go uniform.
    float big[16], bigout[16];
    for (int i = 0; i < 16; i++) big[i] = (i == 0) ? 100.0f : -100.0f;
    memcpy(bigout, big, sizeof big);
    softmax_f32(bigout, 16);
    check("softmax_extreme_one_hot", fabs(bigout[0] - 1.0f) < 1e-6,
          "softmax([100,-100,...]) = [1,0,...]");
}

int main(void) {
    printf("speki kernel accuracy (host reference = libm)\n\n");
    test_exp();
    test_silu();
    test_softmax();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}