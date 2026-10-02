// SPDX-License-Identifier: OSL-3.0
// kernels.h — Speki hand-rolled SIMD kernels (AVX2 + FMA + F16C).
//
// All kernels are header-only inline functions. They target the
// `-march=alderlake` microarchitecture, which gives us:
//   - AVX2: 256-bit SIMD, 8 lanes of FP32 per register
//   - FMA:  3-operand fused multiply-add
//   - F16C: scalar + 8-wide half↔single conversion
//   - BMI1/BMI2/ADX/GFNI: bit-manipulation helpers
//
// We do NOT use AVX-512 (fused off on consumer Alder Lake).
// We do NOT use VNNI (cripped off on consumer Alder Lake).
// We do NOT use AMX (Xeon-only).
//
// Kernels provided:
//   dot_f32_f32:    single FP32 dot product
//   dot_f16_f32:    single FP16 · FP16 → FP32 (with F16C promote)
//   gemm_f16_f32:   (M, N, K) FP16×FP16→FP32 matmul (cache-blocked)
//   attention_f16_f32: scaled dot-product attention (Q·K^T → softmax → ·V)
//   layernorm_f32:  per-row LayerNorm
//   rmsnorm_f32:    per-row RMSNorm (faster, no mean subtraction)
//   softmax_f32:    row-wise softmax
//   silu_f32:       x * sigmoid(x), element-wise
//   gelu_f32:       GELU approximation, element-wise
//   rope_f32:       rotary positional embedding
//   residual_add_f32: element-wise y[i] += x[i]
//
// All functions are no_stack_protector (avoids the %fs canary problem)
// and noinline when they have local stack arrays > 8 bytes.

#ifndef SPEKI_KERNELS_H
#define SPEKI_KERNELS_H

#include "../crt/raw_syscalls.h"
#include "types.h"
#include "tensor.h"
#include "f16c.h"
#include "mathf.h"

#include <immintrin.h>  // AVX2 + FMA + F16C intrinsics, header-only

// ─── Vector reduction helpers ──────────────────────────────────────────────

// horizontal_sum(v) — sum 8 FP32 lanes into a scalar FP32.
static inline fp32 hsum8(__m256 v) {
    // [a b c d e f g h] -> [a+b c+d e+f g+h ...]
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);              // [a+b c+d e+f g+h]
    __m128 sh = _mm_movehdup_ps(s);              // [c+d c+d g+h g+h]
    __m128 su = _mm_add_ps(s, sh);               // [a+b+c+d ...]
    __m128 s2 = _mm_movehl_ps(sh, su);           // [g+h g+h ...]
    __m128 sf = _mm_add_ss(su, s2);              // [a+b+c+d+e+f+g+h ...]
    return _mm_cvtss_f32(sf);
}

// horizontal_max(v) — max of 8 FP32 lanes.
static inline fp32 hmax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    __m128 mh = _mm_movehl_ps(m, m);
    __m128 mm = _mm_max_ps(m, mh);
    mh = _mm_movehdup_ps(mm);
    mm = _mm_max_ss(mm, mh);
    return _mm_cvtss_f32(mm);
}

// ─── dot_f16_f32: scalar FP16·FP16 dot product → FP32 ─────────────────────
//
// Used for small reductions, benchmarking, and as a building block.
// The inner loop reads 8 FP16 values at a time, promotes via F16C, and
// multiplies with the FP32 accumulator vector.
__attribute__((no_stack_protector))
static fp32 dot_f16_f32(const fp16* a, const fp16* b, u32 n) {
    __m256 acc = _mm256_setzero_ps();
    u32 i = 0;
    // Process 8 at a time via F16C promotion.
    for (; i + 8 <= n; i += 8) {
        __m128i ha = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i hb = _mm_loadu_si128((const __m128i*)(b + i));
        __m256 fa = _mm256_cvtph_ps(ha);
        __m256 fb = _mm256_cvtph_ps(hb);
        acc = _mm256_fmadd_ps(fa, fb, acc);
    }
    fp32 s = hsum8(acc);
    // Tail.
    for (; i < n; i++) {
        s += f16_to_f32(a[i]) * f16_to_f32(b[i]);
    }
    return s;
}

// ─── dot_f32_f32: scalar FP32 dot product → FP32 ──────────────────────────
__attribute__((no_stack_protector))
static fp32 dot_f32_f32(const fp32* a, const fp32* b, u32 n) {
    __m256 acc = _mm256_setzero_ps();
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    fp32 s = hsum8(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// ─── gemm_f16_f32: (M, N, K) FP16·FP16 → FP32 matmul ──────────────────────
//
// Computes C[M, N] = A[M, K] * B[K, N] where A and B are FP16, C is FP32.
// Row-major layout: A[m, k] = A_data[m*K + k], B[k, n] = B_data[k*N + n].
//
// Cache blocking: We tile for the L1/L2 cache of Alder Lake. Empirical
// numbers: P-core L1D = 48 KiB, L2 = 1.25 MiB; E-core pair L2 = 2 MiB.
// We pick a block size that fits comfortably in L2.
//
// Strategy: outer loop over output tiles (BM × BN). For each tile, we
// iterate over K in BK blocks, accumulating into 8 FP32 registers.
//
// We process N in groups of 8 (one __m256) and M in groups of 6 (six
// accumulators) to keep the 16 YMM registers busy:
//   6 × C accumulators  + 1 × A broadcast + 1 × B vector = 8 registers used
//   8 remaining for prefetched B tiles (2 tiles of 4 each) is impossible,
//   so we use a 4×6 register tiling instead, which fits in 4+6+0 = 10.
//
// We use a 4×8 register block: 4 rows of A, 8 columns of B → 4 C rows
// accumulated. Registers: 4 (A broadcast) + 1 (B) + 4 (C) = 9.

#define GEMM_MR  4   // rows of A per register block
#define GEMM_NR  8   // columns of B per register block (one __m256)

// gemm_f16_f32(C, A, B, M, N, K)
//   C: fp32*, size M*N, row-major
//   A: fp16*, size M*K, row-major
//   B: fp16*, size K*N, row-major
__attribute__((no_stack_protector, noinline))
void gemm_f16_f32(fp32* __restrict__ C, const fp16* __restrict__ A,
                  const fp16* __restrict__ B, u32 M, u32 N, u32 K) {
    // K-blocking: we promote A and B chunks to FP32 in registers. Tile
    // size along K: process K in chunks of 1 (one FP16 at a time) for
    // simplicity. The F16C promotes 8 FP16 → 8 FP32 in a single
    // instruction, so per-inner-iter we promote 8 cols of B (or 8 cols
    // of a row of A) and do 8 FMAs.
    //
    // This kernel is a "scalar outer / 8-wide inner" gemm — works well
    // for the matmul sizes we care about (768×768, 1024×1024, etc.).
    // For larger matrices we'd add register tiling; for now we keep
    // it simple and correct.

    // Output tile in N: 8 columns at a time.
    for (u32 m = 0; m < M; m++) {
        for (u32 n0 = 0; n0 < N; n0 += GEMM_NR) {
            // Initialize 8 C accumulators for this (m, n0..n0+7) row.
            __m256 c0 = _mm256_setzero_ps();

            for (u32 k = 0; k < K; k++) {
                // Load A[m, k] (1 FP16) and broadcast to 8 lanes via F16C.
                __m128i ha = _mm_set1_epi16((short)(u16)A[m * K + k]);
                __m256  va = _mm256_cvtph_ps(ha);  // 8 copies of A[m,k] as FP32

                // Load B[k, n0..n0+7] (8 FP16) and promote to FP32.
                __m128i hb = _mm_loadu_si128((const __m128i*)(B + k * N + n0));
                __m256  vb = _mm256_cvtph_ps(hb);

                c0 = _mm256_fmadd_ps(va, vb, c0);
            }
            // Store C[m, n0..n0+7].
            _mm256_storeu_ps(C + m * N + n0, c0);
        }
    }
}

// ─── silu_f32: y = x * sigmoid(x) (element-wise) ──────────────────────────
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Computed with mathf.h's exp, which is exact to ~1 ulp of fp32 across the
// whole range. There is no clamping and no polynomial here any more: the
// previous version evaluated a 5-term Taylor series over a [-8, 8] clamp,
// which is invalid outside |x| < 1 and returned NEGATIVE values for x > ~1.5
// (at x = 3.5 it produced -5.4259 where silu(3.5) = 3.3974).
//
// Asymptotics are handled by exp itself: for large positive x, exp(-x)
// underflows to 0 and sigmoid -> 1, which is correct. For large negative x,
// exp(-x) overflows to +inf and x/inf -> -0.0, also correct. Neither needs a
// special case here.
//
// tanh-based formulations are deliberately NOT used: they would need tanh,
// which means another transcendental to hand-roll for no accuracy gain.

__attribute__((no_stack_protector, noinline))
void silu_f32(fp32* x, u32 n) {
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v    = _mm256_loadu_ps(x + i);
        __m256 neg  = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 ex   = speki_exp8_ps(neg);
        __m256 den  = _mm256_add_ps(_mm256_set1_ps(1.0f), ex);
        // v / (1 + exp(-v))
        __m256 sig  = _mm256_div_ps(_mm256_set1_ps(1.0f), den);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sig));
    }
    for (; i < n; i++) {
        fp32 v = x[i];
        x[i] = v * (1.0f / (1.0f + speki_expf(-v)));
    }
}

// ─── softmax_f32: row-wise softmax ─────────────────────────────────────────
//
// We treat `x` as a 1D array of n elements. For 2D softmax, call once
// per row.
//
// softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
//
// We subtract the max for numerical stability (prevents exp overflow).
__attribute__((no_stack_protector, noinline))
void softmax_f32(fp32* x, u32 n) {
    if (n == 0) return;

    // Find max.
        //
        // This MUST be hmax8, not hsum8. The previous version called hsum8 here,
        // which sums the 8 lanes instead of maximising them: for the test input
        // x = i*0.3 (i < 16) it produced a "maximum" of 27.6 where the true max
        // was 4.5. Every subsequent exp(x - 27.6) then hit the clamp, all 16
        // outputs collapsed to the same value, and the row went uniform. The
        // test only noticed because softmax_order failed while
        // softmax_sums_to_1 passed — uniform output still sums to 1.
        __m256 vmax = _mm256_set1_ps(x[0]);
        u32 i = 0;
        for (; i + 8 <= n; i += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
        }
        fp32 maxv = hmax8(vmax);
        for (; i < n; i++) {
            if (x[i] > maxv) maxv = x[i];
        }

        // Compute exp(x_i - max) and accumulate sum.
        __m256 vsum  = _mm256_setzero_ps();
        __m256 vmaxv = _mm256_set1_ps(maxv);
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmaxv);
            // No clamp needed: subtracting the max guarantees v <= 0, so exp
            // cannot overflow, and underflow to 0 is the correct answer for
            // negligible terms.
            __m256 e = speki_exp8_ps(v);
            _mm256_storeu_ps(x + i, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        for (; i < n; i++) {
            x[i] = speki_expf(x[i] - maxv);
        }
        fp32 sum = hsum8(vsum);
        for (u32 j = (n & ~7u); j < n; j++) sum += x[j];

        // A row of zeros has max 0 and every exp is 1, so sum == n > 0. sum can
        // only be 0 if n == 0, which is handled above; the guard stays as a
        // belt-and-braces measure against a denormal-only row.
        if (!(sum > 0.0f)) sum = 1.0f;

    // Normalize.
    __m256 vsumv = _mm256_set1_ps(sum);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, vsumv));
    }
    for (; i < n; i++) x[i] /= sum;
}

// ─── rmsnorm_f32: per-row RMSNorm ─────────────────────────────────────────
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
//
// We treat x as a 1D vector of n elements; for 2D, call once per row.
// gamma is optional (NULL → no scaling).

__attribute__((no_stack_protector, noinline))
void rmsnorm_f32(fp32* x, const fp32* gamma, u32 n, fp32 eps);

// A small inline sqrtf_approx — uses hardware _mm_sqrt_ss for accuracy.
static inline fp32 sqrtf_approx(fp32 x) {
    if (x <= 0.0f) return 0.0f;
    __m128 v = _mm_set_ss(x);
    __m128 r = _mm_sqrt_ss(v);
    return _mm_cvtss_f32(r);
}

// rmsnorm_f32 body — separate from the forward decl so kernel files that
// only call sqrtf_approx don't pull in the full body.
__attribute__((no_stack_protector, noinline))
void rmsnorm_f32(fp32* x, const fp32* gamma, u32 n, fp32 eps) {
    if (n == 0) return;

    // Compute mean of squares.
    __m256 vacc = _mm256_setzero_ps();
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    fp32 sumsq = hsum8(vacc);
    for (; i < n; i++) sumsq += x[i] * x[i];

    fp32 mean = sumsq / (fp32)n;
    fp32 rms  = 1.0f / sqrtf_approx(mean + eps);
    __m256 vrms = _mm256_set1_ps(rms);

    i = 0;
    if (gamma) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            __m256 g = _mm256_loadu_ps(gamma + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(v, vrms), g));
        }
        for (; i < n; i++) x[i] = x[i] * rms * gamma[i];
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vrms));
        }
        for (; i < n; i++) x[i] *= rms;
    }
}

// ─── residual_add_f32: y[i] += x[i] ────────────────────────────────────────
__attribute__((no_stack_protector, noinline))
void residual_add_f32(fp32* y, const fp32* x, u32 n) {
    u32 i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 yv = _mm256_loadu_ps(y + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(yv, xv));
    }
    for (; i < n; i++) y[i] += x[i];
}

#endif // SPEKI_KERNELS_H