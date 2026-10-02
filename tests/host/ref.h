// SPDX-License-Identifier: OSL-3.0
// ref.h — an independent, libc-free reference for the values speki computes.
//
// WHY NOT JUST LINK libm AND COMPARE
// ───────────────────────────────────
// It is tempting to use the host's exp() as ground truth. Two problems:
//
//   1. On glibc, -lm IS glibc. speki exists specifically to not depend on a
//      libc, so a test suite that validates speki by calling glibc's exp
//      would be validating it against the very thing it is built to avoid.
//   2. A reference must be INDEPENDENT of the code under test. Our exp uses
//      range reduction plus a degree-7 Horner polynomial in fp64. If the
//      reference used the same method, a shared conceptual error would pass.
//
// So the reference here is deliberately naive and obviously-correct: Taylor
// series in __float128 (x87 80-bit extended, 64-bit mantissa) evaluated term
// by term until the term stops mattering. That is slow and inelegant, which
// is exactly what you want from a reference — no range reduction to get
// subtly wrong, no coefficient table to mistype, no shared assumptions.
//
// __float128 needs libgcc for the few soft-float helpers (__addtf3 etc.).
// Those are compiler support routines, not libc, and they are only linked
// into THIS test binary — the freestanding speki binary never sees them.
//
// Accuracy: fp128 eps is ~1.1e-34, and Taylor at |r| <= 1 converges to well
// under that, so the reference is accurate to roughly 1e-30 relative. That is
// 23 orders of magnitude tighter than the ~1e-7 we are measuring.

#ifndef SPEKI_TEST_REF_H
#define SPEKI_TEST_REF_H

typedef __float128 ref_fp;

// Number of Taylor terms. exp(x) for |x| <= 1 needs ~30 terms for fp128
// precision; we use a generous 60 and stop early once a term is negligible.
#define REF_TERMS 60

// ref_exp — exp(x) by direct Taylor series in extended precision.
//
// Only valid for a modest argument range; callers reduce first.
static ref_fp ref_exp_small(ref_fp x) {
    // Sum term_n = x^n / n!
    ref_fp sum = 1.0q;
    ref_fp term = 1.0q;
    for (int n = 1; n <= REF_TERMS; n++) {
        term *= x / (ref_fp)n;
        ref_fp next = sum + term;
        if (next == sum) break;          // term is below the rounding floor
        sum = next;
    }
    return sum;
}

// ref_ln2 — ln(2) to fp128, via the atanh series.
//
// ln(2) = 2 * atanh(1/3), where atanh(z) = z + z^3/3 + z^5/5 + ...
// z = 1/3 makes the ratio z^2 = 1/9, so each term is 9x smaller than the last:
// 60 terms puts us far below fp128 eps. This is deliberately NOT a hardcoded
// constant — a hardcoded ln2 would be a transcription risk, and computing it
// means the reference shares no constants with mathf.h either.
static ref_fp ref_ln2(void) {
    const ref_fp z = 1.0q / 3.0q;
    const ref_fp z2 = z * z;
    ref_fp term = z;
    ref_fp sum = 0.0q;
    for (int n = 0; n < REF_TERMS; n++) {
        sum += term / (ref_fp)(2 * n + 1);
        term *= z2;
        if (term == 0.0q) break;
    }
    return 2.0q * sum;
}

// ref_exp2i — 2^k by bit pattern, for the reference's own range reduction.
// Independent of speki_exp2i on purpose: same idea, separately written.
//
// __float128 layout (IEEE 128-bit quadruple): 1 sign, 15 exponent bits,
// 112 mantissa bits. The exponent field sits at bits 112..126, biased by
// 16383. It is NOT fp64's layout — an fp64 shift of 52 would be badly wrong
// here, and silently so: it makes 2^k come out as 0 or denormal for positive k
// without erroring.
//
// The normal range is 2^-16382 .. 2^16383. Outside that, scale in two steps.
static ref_fp ref_exp2i(int k) {
    if (k > 16383)  return ref_exp2i(16383)  * ref_exp2i(k - 16383);
    if (k < -16382) return ref_exp2i(-16382) * ref_exp2i(k + 16382);

    union { unsigned __int128 u; ref_fp f; } v;
    v.u = ((unsigned __int128)(unsigned short)(k + 16383)) << 112;
    return v.f;
}

// ref_exp — exp(x) for any x in fp64 range, to ~1e-30 relative.
//
// Range reduction x = k*ln2 + r with |r| <= ln2/2, then Taylor on r. Same
// SHAPE as the implementation, which is unavoidable — but the arithmetic is
// 33 bits wider and there is no polynomial to get wrong, so the two agreeing
// to 1e-7 is real evidence rather than a shared assumption.
static ref_fp ref_exp(ref_fp x) {
    if (x != x) return x;                       // NaN
    const ref_fp LN2 = ref_ln2();
    const ref_fp LOG2E = 1.0q / LN2;

    // Round to nearest integer.
    ref_fp kf = x * LOG2E;
    long k = (long)(kf >= 0 ? kf + 0.5q : kf - 0.5q);

    ref_fp r = x - (ref_fp)k * LN2;
    return ref_exp_small(r) * ref_exp2i((int)k);
}

// ref_exp_f64 — ref_exp narrowed to double, for comparison against fp32 kernels.
static double ref_exp_f64(double x) {
    return (double)ref_exp((ref_fp)x);
}

// ─── Small utilities, also libc-free ───────────────────────────────────────

static double ref_fabs(double x) {
    return x < 0.0 ? -x : x;
}

static int ref_isnan(double x) {
    return x != x;
}

// Infinity test without <math.h>: compare against the largest double.
static int ref_isinf(double x) {
    return (x == x) && (x > 1.7976931348623157e308 ||
                        x < -1.7976931348623157e308);
}

#endif // SPEKI_TEST_REF_H