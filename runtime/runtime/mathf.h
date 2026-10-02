// SPDX-License-Identifier: OSL-3.0
// mathf.h — libm-free transcendental primitives.
//
// Everything here is hand-rolled for this runtime: no libm, no Cephes
// coefficient tables, no minimax/Remez fitting, no lookup tables. We lean on
// the FPU we already have instead of approximating harder.
//
// WHY fp64 INTERNALS
// ──────────────────
// The obvious approach for exp is Cody-Waite range reduction in fp32:
//   k = rint(x * log2(e));  r = x - k*ln2;  return poly(r) * 2^k
// In fp32 that caps out around 1e-5 relative error, because ln2 is not
// representable in 24 bits and the k*ln2_hi residual grows with |k|.
// Recovering from that means a split constant plus a correction term — more
// code, more constants, still not exact.
//
// Instead: do the reduction and the polynomial in fp64, then round to fp32
// exactly once at the end. AVX2 gives us 4-wide fp64 natively, so the "slow"
// path is 4 lanes per instruction and the result is correct to ~1 ulp of fp32.
// Shorter AND more accurate — that is the whole reason this file is small.
//
// Freestanding throughout: we never call libm, and the 2^k scaling is a
// bit-pattern construction rather than scalbn().

#ifndef SPEKI_MATHF_H
#define SPEKI_MATHF_H

#include "../crt/raw_syscalls.h"
#include "types.h"

#include <immintrin.h>

// ln(2) and log2(e) to fp64 precision. Both are irrational; these are the
// nearest fp64 values and are exact as written.
#define SPEKI_LN2   0.69314718055994530942
#define SPEKI_LOG2E 1.44269504088896340736

// exp(x) overflows fp32 above ln(FLT_MAX) and underflows below the smallest
// subnormal. Clamping here keeps k inside the range where k*ln2 is exact in
// fp64 and 2^k is a normal fp64 power.
#define SPEKI_EXP_HI 88.7
#define SPEKI_EXP_LO -104.0

// +inf. Built by the builtin so it is a compile-time constant in a
// freestanding TU with no libm.
#define SPEKI_INF (__builtin_inff())

// ─── 2^k by bit pattern ────────────────────────────────────────────────────
//
// A normal fp64 has biased exponent k + 1023 in bits 52..62. For
// -1022 <= k <= 1023 that is one shift, versus a libm call. Our clamped range
// keeps k within roughly [-150, 128], comfortably normal.
static inline fp64 speki_exp2i(i32 k) {
    union { u64 u; fp64 d; } v;
    v.u = ((u64)(u32)(k + 1023)) << 52;
    return v.d;
}

// ─── exp on a reduced argument ─────────────────────────────────────────────
//
// Input must satisfy |r| <= ln(2)/2 (~0.3466). Horner, degree 7. Taylor's
// remainder at the interval edge is r^8/8! ≈ 5.2e-9, about 40x below fp32
// eps — so the polynomial is not the error source, the fp64 reduction is.
// Horner in fp64, using explicit fma so there is exactly ONE rounding per
// step. The vector path uses _mm256_fmadd_pd, which also contracts to a
// single fma — using the builtin here and there keeps the two bit-identical.
// If this is written as p*r + c and the compiler contracts it, fine; if it
// does not, the vector and scalar paths would disagree by ~1 ulp per step and
// the host suite's exact-agreement check would fail.
static inline fp64 speki_exp_reduced(fp64 r) {
    fp64 p = 1.0 / 5040.0;              // 1/7!
    p = __builtin_fma(p, r, 1.0 / 720.0);   // 1/6!
    p = __builtin_fma(p, r, 1.0 / 120.0);   // 1/5!
    p = __builtin_fma(p, r, 1.0 / 24.0);    // 1/4!
    p = __builtin_fma(p, r, 1.0 / 6.0);     // 1/3!
    p = __builtin_fma(p, r, 1.0 / 2.0);     // 1/2!
    p = __builtin_fma(p, r, 1.0);
    p = __builtin_fma(p, r, 1.0);
    return p;
}

// ─── exp (scalar) ──────────────────────────────────────────────────────────

// speki_expd — libm-free exp in fp64.
static inline fp64 speki_expd(fp64 x) {
    if (x != x) return x;                       // NaN in, NaN out
    if (x > SPEKI_EXP_HI) return SPEKI_INF;
    if (x < SPEKI_EXP_LO) return 0.0;

    // k = round(x * log2 e), giving r = x - k*ln2 with |r| <= ln2/2.
    __m128d vx = _mm_set_sd(x);
    __m128d vf = _mm_mul_sd(vx, _mm_set_sd(SPEKI_LOG2E));
    i32     k  = _mm_cvttsd_si32(_mm_round_sd(vf, vf, _MM_FROUND_TO_NEAREST_INT));

    fp64 r = x - (fp64)k * SPEKI_LN2;
    return speki_exp_reduced(r) * speki_exp2i(k);
}

// speki_expf — libm-free exp in fp32. Rounds once, at the end.
static inline fp32 speki_expf(fp32 x) {
    return (fp32)speki_expd((fp64)x);
}

// ─── exp, 4 fp64 lanes ─────────────────────────────────────────────────────
//
// Vector form of the same reduction, same coefficients as the scalar path.
// _mm256_cvtpd_epi32 rounds with the current MXCSR mode (round-to-nearest-even
// by default). Clamping is vectorized so a lane holding +inf answers like the
// scalar path, and min/max propagate NaN so NaN inputs stay NaN.

static inline __m256d speki_expd4(__m256d x) {
    const __m256d vln2   = _mm256_set1_pd(SPEKI_LN2);
    const __m256d vlog2e = _mm256_set1_pd(SPEKI_LOG2E);
    const __m256d vhi    = _mm256_set1_pd(SPEKI_EXP_HI);
    const __m256d vlo    = _mm256_set1_pd(SPEKI_EXP_LO);

    __m256d xc = _mm256_min_pd(_mm256_max_pd(x, vlo), vhi);

    __m256d vf = _mm256_mul_pd(xc, vlog2e);
    __m128i ki = _mm256_cvtpd_epi32(vf);       // 4x i32 packed
    __m256d kd = _mm256_cvtepi32_pd(ki);

    // r = x - k*ln2; fnmadd does -(kd*ln2) + xc with a single rounding.
    __m256d r = _mm256_fnmadd_pd(kd, vln2, xc);

    __m256d p = _mm256_set1_pd(1.0 / 5040.0);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 720.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 120.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 24.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 6.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 2.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    // Exactly SEVEN steps, matching speki_exp_reduced. An eighth here would
    // make this degree 9 while the scalar path stayed degree 7, and the two
    // would then disagree by more than rounding — which is how this bug
    // happened in the first place.
    __m256d e = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));

    // Scale by 2^k: normal fp64 exponent field = k + 1023 in bits 52..62.
    //
    // cvtepi32_epi64 sign-extends each i32 into its own 64-bit lane (AVX2
    // VPMOVSXDQ). Do NOT shortcut this with _mm256_slli_epi64 on the packed
    // i32: that shifts each 64-bit lane as a unit, so the i32s sitting in the
    // HIGH halves (k1, k3) get pushed out of the register and k0/k2 end up
    // duplicated into both lanes. That silently scrambles every lane but the
    // first.
    __m128i kb  = _mm_add_epi32(ki, _mm_set1_epi32(1023));
    __m256i k64 = _mm256_slli_epi64(_mm256_cvtepi32_epi64(kb), 52);
    __m256d s   = _mm256_castsi256_pd(k64);

    __m256d out = _mm256_mul_pd(e, s);

    // Fixups compare the ORIGINAL x, so NaN inputs are left alone.
    out = _mm256_blendv_pd(out, _mm256_set1_pd(SPEKI_INF),
                           _mm256_cmp_pd(x, vhi, _CMP_GT_OQ));
    out = _mm256_andnot_pd(_mm256_cmp_pd(x, vlo, _CMP_LT_OQ), out);
    return out;
}

// ─── exp, 8 fp32 lanes ─────────────────────────────────────────────────────
//
// AVX2 fp64 is 4 lanes per register, so 8 fp32 lanes take two passes. Lanes
// stay fp64 throughout and narrow once at the end, matching the scalar path.
// _mm256_cvtpd_ps narrows only 4 lanes, so each fp32 half is narrowed on its
// own and the two results are spliced. (Narrowing the joined fp64 pair would
// silently discard half the lanes.)
static inline __m256 speki_exp8_ps(__m256 x) {
    __m128 flo = _mm256_castps256_ps128(x);          // lanes 0..3
    __m128 fhi = _mm256_extractf128_ps(x, 1);        // lanes 4..7
    __m128 nlo = _mm256_cvtpd_ps(speki_expd4(_mm256_cvtps_pd(flo)));
    __m128 nhi = _mm256_cvtpd_ps(speki_expd4(_mm256_cvtps_pd(fhi)));
    // _mm256_castps128_ps256 zero-extends; the extract-cast goes the other way.
    return _mm256_insertf128_ps(_mm256_castps128_ps256(nlo), nhi, 1);
}

// speki_exp_arr_ps — exp over an fp32 array, 8 lanes per iteration plus a
// scalar tail. This is the entry point kernels use.
__attribute__((no_stack_protector, noinline))
static void speki_exp_arr_ps(fp32* out, const fp32* in, u32 n) {
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            speki_exp8_ps(_mm256_loadu_ps(in + i)));
    }
    for (; i < n; i++) out[i] = speki_expf(in[i]);
}

#endif // SPEKI_MATHF_H