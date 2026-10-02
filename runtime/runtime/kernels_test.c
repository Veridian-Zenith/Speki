// SPDX-License-Identifier: OSL-3.0
// kernels_test.c — exercises the kernels.h SIMD primitives.
//
// Tests:
//   dot_f32_f32     — verify sum matches scalar reference
//   dot_f16_f32     — verify sum matches scalar reference (with F16C)
//   gemm_f16_f32    — verify matmul matches scalar reference
//   silu_f32        — element-wise; compare to a couple of known values
//   softmax_f32     — verify output sums to 1 and order is preserved
//   rmsnorm_f32     — verify output RMS matches expected
//   residual_add_f32 — verify in-place add is correct

#include "../crt/raw_syscalls.h"
#include "../crt/raw_log.h"
#include "../crt/raw_alloc.h"
#include "../crt/raw_time.h"
#include "types.h"
#include "tensor.h"
#include "f16c.h"
#include "kernels.h"

static u32 n_pass = 0;
static u32 n_fail = 0;

#define KCHECK(name, cond) do { \
    if (cond) { SPEKI_LOG(LOG_INFO, "PASS " name); n_pass++; } \
    else      { SPEKI_LOG(LOG_CRIT, "FAIL " name); n_fail++; } \
} while (0)

// ─── Reference implementations (scalar, for correctness comparison) ───────

static fp32 ref_dot_f32(const fp32* a, const fp32* b, u32 n) {
    fp32 s = 0.0f;
    for (u32 i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static fp32 ref_dot_f16(const fp16* a, const fp16* b, u32 n) {
    fp32 s = 0.0f;
    for (u32 i = 0; i < n; i++) s += f16_to_f32(a[i]) * f16_to_f32(b[i]);
    return s;
}

static void ref_gemm_f16_f32(fp32* C, const fp16* A, const fp16* B,
                              u32 M, u32 N, u32 K) {
    for (u32 m = 0; m < M; m++) {
        for (u32 n = 0; n < N; n++) {
            fp32 acc = 0.0f;
            for (u32 k = 0; k < K; k++) {
                acc += f16_to_f32(A[m * K + k]) * f16_to_f32(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
}

static fp32 ref_silu(fp32 x) {
    // sigmoid(x) = 1/(1+exp(-x))
    fp32 e = 0.0f;
    fp32 v = -x;
    if (v > -8.0f && v < 0.0f) {
        e = 1.0f + v + v*v*0.5f + v*v*v*0.166666667f
            + v*v*v*v*0.041666667f + v*v*v*v*v*0.008333333f;
    } else if (v <= -8.0f) {
        e = 0.0f;  // exp(-8) ≈ 0
    } else {
        e = 1.0f;  // v clamped
    }
    return x * (1.0f / (1.0f + e));
}

static void ref_softmax(fp32* x, u32 n) {
    fp32 m = x[0];
    for (u32 i = 1; i < n; i++) if (x[i] > m) m = x[i];
    fp32 s = 0.0f;
    for (u32 i = 0; i < n; i++) {
        fp32 v = x[i] - m;
        if (v < -8.0f) v = -8.0f;
        x[i] = 1.0f + v + v*v*0.5f + v*v*v*0.166666667f
               + v*v*v*v*0.041666667f + v*v*v*v*v*0.008333333f;
        s += x[i];
    }
    for (u32 i = 0; i < n; i++) x[i] /= s;
}

static void ref_rmsnorm(fp32* x, const fp32* gamma, u32 n, fp32 eps) {
    fp32 sumsq = 0.0f;
    for (u32 i = 0; i < n; i++) sumsq += x[i] * x[i];
    fp32 rms = 1.0f / sqrtf_approx(sumsq / (fp32)n + eps);
    if (gamma) {
        for (u32 i = 0; i < n; i++) x[i] = x[i] * rms * gamma[i];
    } else {
        for (u32 i = 0; i < n; i++) x[i] *= rms;
    }
}

static int approx_eq(fp32 a, fp32 b, fp32 tol) {
    fp32 d = a - b;
    if (d < 0) d = -d;
    return d <= tol;
}

__attribute__((noinline, used, no_stack_protector))
int kernels_test_main(void) {
    SPEKI_LOG(LOG_INFO, "=== kernels_test ===");

    // ── Test dot_f32_f32 ───────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        fp32* x = (fp32*)arena_alloc(&a, 64 * sizeof(fp32), 32);
        fp32* y = (fp32*)arena_alloc(&a, 64 * sizeof(fp32), 32);
        for (u32 i = 0; i < 64; i++) { x[i] = (fp32)i * 0.1f; y[i] = (fp32)(63 - i) * 0.05f; }

        fp32 got = dot_f32_f32(x, y, 64);
        fp32 ref = ref_dot_f32(x, y, 64);
        KCHECK("dot_f32_64", approx_eq(got, ref, 1e-2f));

        // Small (n=8) test: 1*1 + 2*2 + ... + 8*8 = 204
        for (u32 i = 0; i < 8; i++) { x[i] = (fp32)(i+1); y[i] = (fp32)(i+1); }
        got = dot_f32_f32(x, y, 8);
        KCHECK("dot_f32_8_exact", approx_eq(got, 204.0f, 1e-3f));

        // Non-multiple-of-8 size
        for (u32 i = 0; i < 13; i++) { x[i] = (fp32)(i+1); y[i] = 1.0f; }
        got = dot_f32_f32(x, y, 13);
        KCHECK("dot_f32_13_tail", approx_eq(got, 91.0f, 1e-3f));

        arena_destroy(&a);
    }

    // ── Test dot_f16_f32 ───────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        fp16* x = (fp16*)arena_alloc(&a, 64 * sizeof(fp16), 32);
        fp16* y = (fp16*)arena_alloc(&a, 64 * sizeof(fp16), 32);
        for (u32 i = 0; i < 64; i++) {
            x[i] = f32_to_f16((fp32)i * 0.1f);
            y[i] = f32_to_f16((fp32)(63 - i) * 0.05f);
        }

        fp32 got = dot_f16_f32(x, y, 64);
        fp32 ref = ref_dot_f16(x, y, 64);
        KCHECK("dot_f16_64", approx_eq(got, ref, 5e-2f));

        arena_destroy(&a);
    }

    // ── Test gemm_f16_f32 (small 4×4×4) ────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        // M=4, N=8 (one n-tile), K=4
        u32 M = 4, N = 8, K = 4;
        fp16* A = (fp16*)arena_alloc(&a, M * K * sizeof(fp16), 32);
        fp16* B = (fp16*)arena_alloc(&a, K * N * sizeof(fp16), 32);
        fp32* C = (fp32*)arena_alloc(&a, M * N * sizeof(fp32), 32);
        fp32* Cref = (fp32*)arena_alloc(&a, M * N * sizeof(fp32), 32);

        for (u32 i = 0; i < M * K; i++) A[i] = f32_to_f16((fp32)(i + 1));
        for (u32 i = 0; i < K * N; i++) B[i] = f32_to_f16((fp32)((i % 5) + 1));

        gemm_f16_f32(C, A, B, M, N, K);
        ref_gemm_f16_f32(Cref, A, B, M, N, K);

        u32 errors = 0;
        for (u32 i = 0; i < M * N; i++) {
            if (!approx_eq(C[i], Cref[i], 1e-1f)) errors++;
        }
        KCHECK("gemm_4x8x4", errors == 0);

        // Larger: 32×16×16 (test the n-tile loop)
        M = 32; N = 16; K = 16;
        fp16* A2 = (fp16*)arena_alloc(&a, M * K * sizeof(fp16), 32);
        fp16* B2 = (fp16*)arena_alloc(&a, K * N * sizeof(fp16), 32);
        fp32* C2 = (fp32*)arena_alloc(&a, M * N * sizeof(fp32), 32);
        fp32* Cref2 = (fp32*)arena_alloc(&a, M * N * sizeof(fp32), 32);

        for (u32 i = 0; i < M * K; i++) A2[i] = f32_to_f16((fp32)((i * 7) % 13) * 0.1f);
        for (u32 i = 0; i < K * N; i++) B2[i] = f32_to_f16((fp32)((i * 11) % 17) * 0.1f);

        gemm_f16_f32(C2, A2, B2, M, N, K);
        ref_gemm_f16_f32(Cref2, A2, B2, M, N, K);

        errors = 0;
        fp32 max_err = 0.0f;
        for (u32 i = 0; i < M * N; i++) {
            fp32 d = C2[i] - Cref2[i];
            if (d < 0) d = -d;
            if (d > max_err) max_err = d;
            if (d > 1e-1f) errors++;
        }
        KCHECK("gemm_32x16x16", errors == 0);
        // Log max error so we can see how close we are.
        {
            char msg[64];
            u32 off = 0;
            const char* p = "gemm_max_err=";
            while (*p && off + 1 < sizeof(msg)) msg[off++] = *p++;
            // Format float as "0.NNN"
            u32 intpart = (u32)max_err;
            u32 frac    = (u32)((max_err - (fp32)intpart) * 1000.0f) % 1000;
            char buf[16];
            u32 n = 0; char tmp[8];
            if (intpart == 0) tmp[n++] = '0';
            else { while (intpart > 0) { tmp[n++] = '0' + (intpart % 10); intpart /= 10; } }
            while (n > 0) msg[off++] = tmp[--n];
            msg[off++] = '.';
            for (u32 k = 0; k < 3; k++) {
                u32 d = (frac / 100) % 10;
                msg[off++] = (char)('0' + d);
                frac = (frac * 10) % 1000;
            }
            SPEKI_LOG_BUF(LOG_INFO, msg, off);
        }

        arena_destroy(&a);
    }

    // ── Test silu_f32 ──────────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        u32 n = 16;
        fp32* x = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        fp32* xref = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) {
            x[i] = (fp32)(i - 8) * 0.5f;  // range [-4, 3.5]
            xref[i] = x[i];
        }
        silu_f32(x, n);
        for (u32 i = 0; i < n; i++) xref[i] = ref_silu(xref[i]);
        u32 errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (!approx_eq(x[i], xref[i], 5e-2f)) errs++;
        }
        KCHECK("silu_f32", errs == 0);

        arena_destroy(&a);
    }

    // ── Test softmax_f32 ──────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        u32 n = 16;
        fp32* x = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) x[i] = (fp32)i * 0.3f;  // increasing
        fp32* xref = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) xref[i] = x[i];

        // Capture snapshot before ref runs.
        for (u32 i = 0; i < n; i++) xref[i] = x[i];
        softmax_f32(x, n);
        ref_softmax(xref, n);

        // All elements should be in (0, 1).
        u32 errs = 0;
        fp32 sum = 0.0f;
        for (u32 i = 0; i < n; i++) {
            sum += x[i];
            if (!approx_eq(x[i], xref[i], 1e-2f)) errs++;
        }
        KCHECK("softmax_correctness", errs == 0);
        KCHECK("softmax_sums_to_1",   approx_eq(sum, 1.0f, 1e-3f));

        // Order is preserved: x[0] < x[1] < ... < x[n-1] since input was increasing.
        u32 order_ok = 1;
        for (u32 i = 1; i < n; i++) {
            if (x[i] <= x[i-1]) { order_ok = 0; break; }
        }
        KCHECK("softmax_order", order_ok);

        arena_destroy(&a);
    }

    // ── Test rmsnorm_f32 ──────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        u32 n = 32;
        fp32* x = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        fp32* g = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) { x[i] = (fp32)(i + 1) * 0.1f; g[i] = 1.0f; }
        fp32* xref = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) xref[i] = x[i];

        rmsnorm_f32(x, g, n, 1e-5f);
        ref_rmsnorm(xref, g, n, 1e-5f);

        u32 errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (!approx_eq(x[i], xref[i], 5e-3f)) errs++;
        }
        KCHECK("rmsnorm_f32", errs == 0);

        // Verify the post-RMSNorm RMS of x is close to 1 (when gamma=1).
        fp32 sumsq = 0.0f;
        for (u32 i = 0; i < n; i++) sumsq += x[i] * x[i];
        fp32 post_rms = sqrtf_approx(sumsq / (fp32)n);
        KCHECK("rmsnorm_post_rms_is_1", approx_eq(post_rms, 1.0f, 1e-2f));

        arena_destroy(&a);
    }

    // ── Test residual_add_f32 ──────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        u32 n = 17;
        fp32* y = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        fp32* x = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) { y[i] = (fp32)(i + 1); x[i] = 0.5f; }

        residual_add_f32(y, x, n);
        u32 errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (!approx_eq(y[i], (fp32)(i + 1) + 0.5f, 1e-5f)) errs++;
        }
        KCHECK("residual_add_f32", errs == 0);

        arena_destroy(&a);
    }

    // ── Test F16C roundtrip (FP32 → FP16 → FP32) ──────────────────────────
    {
        // Convert a known value and check error is small (< 0.5 ULP of FP16).
        fp32 orig = 0.5f;
        fp16 h = f32_to_f16(orig);
        fp32 back = f16_to_f32(h);
        KCHECK("f16c_roundtrip_half", back == orig);  // exactly representable

        orig = 3.14159f;
        h = f32_to_f16(orig);
        back = f16_to_f32(h);
        // FP16 has 11 bits of mantissa; expect error < 0.001 for this value.
        KCHECK("f16c_roundtrip_pi", approx_eq(back, orig, 1e-3f));
    }

    // ── Test F16C SIMD (8-wide conversion) ────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        fp16* h8 = (fp16*)arena_alloc(&a, 8 * sizeof(fp16), 32);
        fp32* f8 = (fp32*)arena_alloc(&a, 8 * sizeof(fp32), 32);
        for (u32 i = 0; i < 8; i++) h8[i] = f32_to_f16((fp32)(i + 1) * 0.25f);
        f16x8_to_f32x8(h8, f8);
        u32 errs = 0;
        for (u32 i = 0; i < 8; i++) {
            if (!approx_eq(f8[i], (fp32)(i + 1) * 0.25f, 1e-3f)) errs++;
        }
        KCHECK("f16x8_to_f32x8", errs == 0);

        // Round-trip back
        fp16* h8b = (fp16*)arena_alloc(&a, 8 * sizeof(fp16), 32);
        f32x8_to_f16x8(f8, h8b);
        errs = 0;
        for (u32 i = 0; i < 8; i++) {
            if (f16_to_f32(h8b[i]) != f8[i]) errs++;
        }
        KCHECK("f32x8_to_f16x8_roundtrip", errs == 0);

        arena_destroy(&a);
    }

    return 0;
}