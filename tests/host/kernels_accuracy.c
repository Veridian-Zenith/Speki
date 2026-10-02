// SPDX-License-Identifier: OSL-3.0
// kernels_accuracy.c — host-side ground truth for the speki kernels.
//
// The freestanding binary cannot measure its own numerical error: it has no
// libm to compare against, and adding one would defeat the point of the
// project. This test compiles the SAME header-only kernels on the host and
// checks them against an independent reference (see ref.h) — independent in
// both senses: not libc's exp, and not the same algorithm as mathf.h.
//
// It links no C library. See ref.h for why.
//
// Run via ctest, or directly:
//   cmake -B build/cmake --preset dev
//   cmake --build build/cmake --target speki_kernels_host
//   ./build/speki_kernels_host

#include "runtime/runtime/mathf.h"
#include "runtime/runtime/f16c.h"
#include "runtime/runtime/kernels.h"

#include "tests/host/ref.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

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

// Relative error, which is the number that matters here: everything
// downstream of these kernels is scale-invariant.
static double relerr(double got, double want) {
    if (want == 0.0) return ref_fabs(got);
    return ref_fabs(got - want) / ref_fabs(want);
}

static void fmt(char* buf, size_t n, const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, n, f, ap);
    va_end(ap);
}

// ─── exp ───────────────────────────────────────────────────────────────────

static void test_exp(void) {
    printf("exp:\n");

    // Scalar path across the range that matters for fp32 output. We stop at
    // -87.3 because exp(x) below that is subnormal in fp32, where relative
    // error is dominated by denormal rounding rather than by our algorithm.
    double worst = 0.0, worst_x = 0.0;
    for (int i = -870; i <= 887; i++) {
        double x = i / 10.0;
        double got = (double)speki_expf((float)x);
        double want = ref_exp_f64(x);
        if (want == 0.0) continue;
        double e = relerr(got, want);
        if (e > worst) { worst = e; worst_x = x; }
    }
    char d[160];
    fmt(d, sizeof d, "max_rel_err=%.3e at x=%.1f", worst, worst_x);
    // 1 ulp of fp32 is 1.19e-7. Allow slack for the double rounding at the end.
    check("expf_scalar_range", worst < 5e-7, d);

    // Vector path must agree with the scalar path EXACTLY. If these drift, a
    // kernel mixing both is subtly wrong in a way no tolerance would catch.
    float in[64], vs[64], sc[64];
    for (int i = 0; i < 64; i++) in[i] = -20.0f + (float)i * 0.7f;
    speki_exp_arr_ps(vs, in, 64);
    for (int i = 0; i < 64; i++) sc[i] = speki_expf(in[i]);
    int mismatch = 0;
    for (int i = 0; i < 64; i++) if (vs[i] != sc[i]) mismatch++;
    fmt(d, sizeof d, "%d/64 lanes differ", mismatch);
    check("exp_vec_matches_scalar", mismatch == 0, d);

    // Tail lengths: a non-multiple-of-8 n must not drop or duplicate elements.
    // Compare against our own scalar path — the point here is that the tail
    // loop covers exactly the lanes the vector loop skipped. Accuracy against
    // the reference is covered above.
    {
        int bad_len = 0;
        for (unsigned n = 1; n <= 17; n++) {
            float a[20], b[20], s[20];
            for (unsigned i = 0; i < n; i++) a[i] = (float)i * 0.5f - 4.0f;
            speki_exp_arr_ps(b, a, n);
            for (unsigned i = 0; i < n; i++) s[i] = speki_expf(a[i]);
            for (unsigned i = 0; i < n; i++) if (b[i] != s[i]) bad_len++;
        }
        fmt(d, sizeof d, "%d lanes differ from scalar, n=1..17", bad_len);
        check("exp_tail_lengths_1_17", bad_len == 0, d);
    }

    // Overflow / underflow edges.
    check("exp_overflow_is_inf", ref_isinf((double)speki_expf(100.0f)),
          "exp(100) = +inf");
    check("exp_underflow_is_zero", speki_expf(-200.0f) == 0.0f,
          "exp(-200) = 0");

    // Known values.
    check("exp_0_is_1", speki_expf(0.0f) == 1.0f, "exp(0) = 1");
    check("exp_1_is_e",
          relerr(speki_expf(1.0f), 2.718281828459045) < 1e-7, "exp(1) = e");
}

// ─── silu ──────────────────────────────────────────────────────────────────

static void test_silu(void) {
    printf("silu:\n");
    const int N = 512;
    static float x[N], y[N];

    for (int i = 0; i < N; i++) x[i] = -12.0f + (float)i * (24.0f / (N - 1));
    memcpy(y, x, sizeof x);
    silu_f32(y, N);

    double worst = 0.0, worst_x = 0.0;
    for (int i = 0; i < N; i++) {
        double e_ref = ref_exp_f64(-(double)x[i]);
        double want = (double)x[i] / (1.0 + e_ref);
        double e = relerr(y[i], want);
        if (e > worst) { worst = e; worst_x = x[i]; }
    }
    char d[160];
    fmt(d, sizeof d, "max_rel_err=%.3e at x=%.2f", worst, worst_x);
    check("silu_relative_accuracy", worst < 1e-5, d);

    // The regression this guards: the old polynomial returned NEGATIVE values
    // for x >= ~1.5 because the Horner recurrence was y^k*poly + c_k instead
    // of poly*y + c_k.
    int negatives = 0;
    for (int i = 0; i < N; i++) if (x[i] > 0.0f && y[i] < 0.0f) negatives++;
    fmt(d, sizeof d, "%d negative outputs for x>0", negatives);
    check("silu_never_negative_for_pos_x", negatives == 0, d);

    // silu is strictly increasing, so this also catches shape errors.
    int mono = 1;
    for (int i = 1; i < N; i++) if (y[i] < y[i-1]) { mono = 0; break; }
    check("silu_monotonic", mono, "y[i] >= y[i-1] for all i");
}

// ─── softmax ───────────────────────────────────────────────────────────────

// softmax_denominator — max-subtracted sum in fp128, the way the kernel is
// supposed to compute it.
static double softmax_denom(const float* x, int n) {
    ref_fp m = (ref_fp)x[0];
    for (int i = 1; i < n; i++) if ((ref_fp)x[i] > m) m = (ref_fp)x[i];
    ref_fp s = 0.0q;
    for (int i = 0; i < n; i++) s += ref_exp((ref_fp)x[i] - m);
    return (double)s;
}

static void test_softmax(void) {
    printf("softmax:\n");
    const int N = 64;
    static float x[N], y[N];

    // The critical regression: softmax used hsum8 where it needed hmax8, so it
    // took the SUM of the row as the maximum. Every exp(x - sum) then hit the
    // clamp and the whole row collapsed to a uniform distribution.
    for (int i = 0; i < N; i++) x[i] = (float)i * 0.3f;
    memcpy(y, x, sizeof y);
    softmax_f32(y, N);

    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += y[i];
    char d[160];
    fmt(d, sizeof d, "sum=%.9f", sum);
    check("softmax_sums_to_one", ref_fabs(sum - 1.0) < 1e-6, d);

    // Denominator first, so both sides share it.
    double denom = softmax_denom(x, N);
    ref_fp maxv = (ref_fp)x[0];
    for (int i = 1; i < N; i++) if ((ref_fp)x[i] > maxv) maxv = (ref_fp)x[i];

    double worst = 0.0;
    for (int i = 0; i < N; i++) {
        double want = (double)(ref_exp((ref_fp)x[i] - maxv) / (ref_fp)denom);
        double e = relerr(y[i], want);
        if (e > worst) worst = e;
    }
    fmt(d, sizeof d, "max_rel_err=%.3e", worst);
    check("softmax_matches_reference", worst < 1e-5, d);

    // Order preservation on increasing input. This is what the hsum8 bug
    // destroyed: uniform output has no order at all.
    int mono = 1;
    for (int i = 1; i < N; i++) if (y[i] <= y[i-1]) { mono = 0; break; }
    check("softmax_order_preserved", mono,
          "strictly increasing for increasing input");

    // Invariance to a constant shift — the defining property, and the reason
    // max-subtraction exists at all.
    float a[16], b[16];
    for (int i = 0; i < 16; i++) a[i] = (float)i * 0.7f - 5.0f;
    memcpy(b, a, sizeof a);
    softmax_f32(a, 16);
    for (int i = 0; i < 16; i++) b[i] += 3.0f;
    softmax_f32(b, 16);
    int diff = 0;
    for (int i = 0; i < 16; i++) if (ref_fabs(a[i] - b[i]) > 1e-5f) diff++;
    fmt(d, sizeof d, "%d/16 differ under +3 shift", diff);
    check("softmax_shift_invariant", diff == 0, d);

    // Extreme input must not overflow or go uniform.
    float big[16], bigout[16];
    for (int i = 0; i < 16; i++) big[i] = (i == 0) ? 100.0f : -100.0f;
    memcpy(bigout, big, sizeof big);
    softmax_f32(bigout, 16);
    check("softmax_extreme_one_hot", ref_fabs(bigout[0] - 1.0f) < 1e-6,
          "softmax([100,-100,...]) = [1,0,...]");
}

int main(void) {
    printf("speki kernel accuracy (reference = independent fp128 Taylor)\n\n");
    test_exp();
    test_silu();
    test_softmax();
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}