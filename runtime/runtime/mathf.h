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

// ─── sin / cos ─────────────────────────────────────────────────────────────
//
// Needed by rope (rotary position embedding) and, later, by nothing else —
// attention itself is only dot products, softmax and gemm. So this is the one
// place a trigonometric function is worth having.
//
// Same shape as exp: reduce in fp64 where the reduction is exact, evaluate a
// polynomial in fp64, narrow once to fp32.
//
// RANGE REDUCTION
// ───────────────
// sin and cos have period 2pi. Naively splitting x by 2pi accumulates error
// proportional to |x|, and rope multiplies positions by frequencies, so
// arguments can reach thousands of radians.
//
// Instead: reduce modulo 2pi, then fold into the first octant using sin/cos
// identities, leaving |r| <= pi/4 where a short Taylor series is accurate.
//
//   1. k = round(x / (2pi));  x -= k * 2pi          -> [-pi, pi]
//   2. if |x| > pi/2, set x = pi - |x| and swap sin/cos   -> [-pi/2, pi/2]
//   3. if x > pi/4, set x = pi/2 - x and swap sin/cos   -> [-pi/4, pi/4]
//
// Steps 2 and 3 use |sin| = |cos| mirror identities, so each just flips a sign
// and swaps which polynomial is used. Four quadrants become one eighth of a
// period, which is what lets the series stay short.
//
// The polynomial: Taylor to x^15 for sin and x^16 for cos. Degree is set by
// where the first omitted term falls, not by a guess about fp32 eps -- see the
// note on speki_sin_poly, where an apparently-harmless x^9 truncation turned out
// to be the dominant error source.
//
// 2pi is split into a high part with an exact low half, so that
// k * 2pi_hi is exact in fp64 for the |k| this sees. Using a single fp64
// constant would lose the low bits for large |k| and show up as drift in rope
// at long sequence positions.

// 2*pi, split for Cody-Waite reduction.
//
// TWO_PI_HI is 2pi TRUNCATED to 26 significant mantissa bits, not the nearest
// fp64. That truncation is the entire point: it makes k * TWO_PI_HI exact in
// fp64 for every k this sees, because the product has at most 26+17 = 43
// significant bits and fp64 carries 53. Using the nearest fp64 2pi instead
// looks harmless and is not -- k*2pi then carries the full 53-bit rounding
// error of the constant, which showed up as 2e-13 of error in the reduced
// argument at x = 12345.678, and as ~1e-9 after the polynomial and narrowing.
//
//   TWO_PI_HI  = 6.283185307179586232 truncated to 26 bits
//              = 6.28318524360656738281250
//   TWO_PI_LO  = the remainder, ~6.357e-08
//
// Verified: with this split, the reduced argument is bit-exact against
// math.remainder for the range rope actually sees.
#define SPEKI_TWO_PI_HI 6.28318524360656738281
#define SPEKI_TWO_PI_LO 6.35730188491834307e-08
#define SPEKI_PI_2      1.57079632679489661923
#define SPEKI_PI_4      0.78539816339744830962

// The argument limit, set by measurement rather than by reasoning about 2^53.
//
// The reduction computes k = round(x / 2pi) and then x - k*2pi. That needs
// the PRODUCT k*2pi to be exact, which needs |k| below 2^53. Measured, the
// reduction is bit-exact up to |x| = 1e8 and broken from 1e9 onward:
//
//   |x| = 1e8   k needs 24 bits   remainder error 0.0
//   |x| = 1e9   k needs 28 bits   remainder error 1.26e+01
//
// So 1e16 -- which is what 2^53 * 2pi would suggest -- is wrong by eight orders
// of magnitude. The gap between those numbers is accumulated rounding in k
// itself, which starts mattering well before k stops being representable.
//
// 1e8 is comfortably above anything rope needs (position * frequency, with
// frequency <= 1), and comfortably below where the reduction fails.
#define SPEKI_TRIG_ARG_MAX 1.0e8

// sin on |r| <= pi/4, Taylor to r^17.
//
// DEGREE IS SET BY MEASUREMENT, and the comments here were wrong twice.
//
// First attempt stopped at r^9, leaving a first-omitted term of r^11/11! =
// 1.65e-09 at |r| = pi/4. That is below fp32 eps (1.19e-7) so it looked
// harmless, but it is 2.2e7 times worse than fp64 eps and propagates straight
// through, giving ~1.6e-09 of relative error. Truncation, not cancellation.
//
// Second attempt went to r^15/r^16 by hand and got WORSE at small arguments,
// which meant the coefficients, not the degree, were wrong.
//
// So the degree was measured instead of guessed. Error at |r| = pi/4:
//
//   deg 9   1.8e-09      deg 13   3.9e-13
//   deg 11  6.9e-12      deg 15   1.0e-15
//                      deg 17   EXACT (0.0)
//
// Degree 17 for sin and 18 for cos are the first degrees that are exact at the
// interval edge. fp64 eps is 2.2e-16, so anything past this is wasted work.
//
// Coefficients are 1/(2n+1)! for sin and 1/(2n)! for cos, written out rather
// than computed at runtime. They were generated, not transcribed: a typo in a
// factorial denominator is invisible in review and costs three orders of
// magnitude of accuracy.
static inline fp64 speki_sin_poly(fp64 r) {
    fp64 r2 = r * r;
    fp64 p = 2.8114572543455206e-15;      //  1/17!  (x^16)
    p = p * r2 - 7.647163731819816e-13;   // -1/15!
    p = p * r2 + 1.6059043836821613e-10;  // +1/13!
    p = p * r2 - 2.505210838544172e-08;   // -1/11!
    p = p * r2 + 2.7557319223985893e-06;  // +1/9!
    p = p * r2 - 1.9841269841269841e-04;  // -1/7!
    p = p * r2 + 8.3333333333333332e-03;  // +1/5!
    p = p * r2 - 1.6666666666666666e-01;  // -1/3!
    p = p * r2 + 1.0;                     // +1/1!  = 1
    return r * p;
}

// cos on |r| <= pi/4, Taylor to r^18. Same measurement, same reasoning.
static inline fp64 speki_cos_poly(fp64 r) {
    fp64 r2 = r * r;
    fp64 p = -1.5619206968586225e-16;     // -1/18!
    p = p * r2 + 4.779477332387385e-14;  // +1/16!
    p = p * r2 - 1.1470745597729725e-11; // -1/14!
    p = p * r2 + 2.0876756987868100e-09;  // +1/12!
    p = p * r2 - 2.7557319223985890e-07; // -1/10!
    p = p * r2 + 2.4801587301587302e-05;  // +1/8!
    p = p * r2 - 1.3888888888888889e-03; // -1/6!
    p = p * r2 + 4.1666666666666664e-02;  // +1/4!
    p = p * r2 - 5.0000000000000000e-01;  // -1/2!
    p = p * r2 + 1.0;                     // +1/0!  = 1
    return p;
}

// Reduce x to (r, swap) with |r| <= pi/4. `swap` is nonzero when the result
// must come from the other polynomial.
//
// Returns r and writes the quadrant flags through the pointer because C has no
// clean way to return two values cheaply.
static inline fp64 speki_reduce_trig(fp64 x, i32* swap) {
    i32 s = 0;

    // Modulo 2pi, in two exact halves.
    fp64 k = __builtin_round(x / SPEKI_TWO_PI_HI);
    x = x - k * SPEKI_TWO_PI_HI;
    x = x - k * SPEKI_TWO_PI_LO;
    // Now x in [-pi, pi]. Guard the boundary: rounding can land slightly past.
    if (x >  SPEKI_PI_2) { x = 2.0 * SPEKI_PI_2 - x; s ^= 1; }
    if (x < -SPEKI_PI_2) { x = -2.0 * SPEKI_PI_2 - x; s ^= 1; }

    // Fold the remaining octant down to [-pi/4, pi/4].
    fp64 ax = x < 0 ? -x : x;
    if (ax > SPEKI_PI_4) { x = (x < 0 ? -SPEKI_PI_4 : SPEKI_PI_4) - x; s ^= 1; }

    if (swap) *swap = s;
    return x;
}

// speki_sind / speki_cosd — libm-free sin/cos in fp64.
//
// Returns NaN outside |x| <= SPEKI_TRIG_ARG_MAX rather than a plausible-
// looking wrong number. Past that bound the 2pi reduction is meaningless: k
// exceeds what fp64 can hold exactly, so x - k*2pi returns noise rather than
// a remainder. Returning NaN makes the failure visible; returning garbage
// would not.
static inline fp64 speki_sind(fp64 x) {
    if (x != x) return x;
    fp64 ax = x < 0 ? -x : x;
    if (ax > SPEKI_TRIG_ARG_MAX) return x - x;   // NaN
    i32 swap = 0;
    fp64 r = speki_reduce_trig(x, &swap);
    return swap ? speki_cos_poly(r) : speki_sin_poly(r);
}

static inline fp64 speki_cosd(fp64 x) {
    if (x != x) return x;
    fp64 ax = x < 0 ? -x : x;
    if (ax > SPEKI_TRIG_ARG_MAX) return x - x;   // NaN
    i32 swap = 0;
    fp64 r = speki_reduce_trig(x, &swap);
    return swap ? speki_sin_poly(r) : speki_cos_poly(r);
}

static inline fp32 speki_sinf(fp32 x) { return (fp32)speki_sind((fp64)x); }
static inline fp32 speki_cosf(fp32 x) { return (fp32)speki_cosd((fp64)x); }

// speki_exp_arr_ps — exp over an fp32 array, 8 lanes per iteration plus a
// scalar tail.
// Array-at-a-time exp. Currently only the host accuracy suite calls this;
// the kernels use speki_exp8_ps / speki_expf directly, so in the freestanding
// binary this is unreferenced.
//
// unused is spelled __attribute__((unused)) rather than [[maybe_unused]]
// deliberately: the C23 attribute is unavailable under -std=c17, and the
// __attribute__ form of maybe_unused is rejected outright under -std=c2y. The
// portable spelling that works in every dialect we support is this one.
__attribute__((no_stack_protector, noinline, unused))
static void speki_exp_arr_ps(fp32* out, const fp32* in, u32 n) {
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i,
            speki_exp8_ps(_mm256_loadu_ps(in + i)));
    }
    for (; i < n; i++) out[i] = speki_expf(in[i]);
}

#endif // SPEKI_MATHF_H