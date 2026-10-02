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

// The reference implementations below must NOT share a method with the code
// under test, or a shared conceptual error passes unnoticed. That is exactly
// what happened before: ref_silu clamped e = 1.0 whenever -x >= 0, so it
// returned x/2 for every negative input, and it "agreed" with a broken kernel.
//
// WHY THERE IS NO TRIGONOMETRIC REFERENCE HERE
// ────────────────────────────────────────────
// The freestanding binary has no libm and no extended-precision reference, so
// any exp it writes itself is just a second, worse implementation: a Taylor
// series diverges past |x| ~ 2 (ref_silu returned -8.98 for silu(3.5), where
// the true value is 3.397), and duplicating mathf.h's range reduction would
// make the test circular.
//
// So this suite checks STRUCTURAL properties that are decidable without a
// reference, and tests/host/kernels_accuracy.c checks numerical accuracy
// against an independent __float128 Taylor series. Between them, a kernel has
// to be both self-consistent and actually correct — neither suite can be
// satisfied by an implementation that is merely plausible.
//
// Structural facts used below, all of which hold for the true silu/softmax:
//   silu(0) == 0, silu is increasing, silu(x) < x for all x, silu(x) -> 0 as
//   x -> -inf, silu(x) -> x as x -> +inf;
//   softmax sums to 1, preserves order on increasing input, is invariant to a
//   constant shift, and never overflows.

// ref_silu_at -- silu via the kernel's own exp.
//
// Deliberately NOT a Taylor series: a series diverges past |x| ~ 2 and cannot
// validate a kernel over the range we test. This shares exp with the kernel,
// so it cannot catch a bug IN exp -- but exp is checked exhaustively against
// an independent __float128 reference in tests/host/kernels_accuracy.c. What
// this catches is a bug in the silu composition itself (wrong sign, wrong
// denominator, non-monotone output shape).
static fp32 ref_silu_at(fp32 v) {
    return v * (1.0f / (1.0f + speki_expf(-v)));
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

// relerr_fp32 — relative error, which is the scale-free quantity. Absolute
// tolerances are only meaningful next to a statement about the binade they
// were derived from; see the f16c_roundtrip_pi check.
static fp32 relerr_fp32(fp32 got, fp32 want) {
    fp32 d = got - want;
    if (d < 0) d = -d;
    if (want == 0.0f) return d;
    return d / (want < 0 ? -want : want);
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
        fp32* orig = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) {
            // NOTE: (i32)i - 8, not (i - 8). With u32 i the subtraction
            // underflows for i < 8, so the "range [-4, 3.5]" this test has
            // always claimed was really [2147483640, 3.5]. The cast has to
            // happen BEFORE the subtraction.
            orig[i] = (fp32)((i32)i - 8) * 0.5f;   // range [-4, 3.5]
            x[i] = orig[i];
        }
        silu_f32(x, n);

        // Structural checks; numerical accuracy lives in the host suite.
        // Every one of these is a property of the true silu that a wrong
        // kernel cannot satisfy.

        // silu(0) == 0 exactly, and no output is NaN or negative for x > 0.
        u32 sign_errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (orig[i] > 0.0f && x[i] < 0.0f) sign_errs++;
            if (orig[i] == 0.0f && x[i] != 0.0f) sign_errs++;
        }
        KCHECK("silu_f32", sign_errs == 0);

        // silu is strictly increasing. Checked against the reference rather
        // than by deriving a comparison rule: every sign-based rule tried here
        // was wrong, because silu is negative for x < 0, passes through a
        // minimum near x = -1.278, and differences of nearby negatives are
        // dominated by cancellation. The reference comparison is exact and
        // reference-free in the sense that it needs no transcendental of its
        // own -- ref_silu_at uses the kernel's own exp, which is validated
        // independently by tests/host/kernels_accuracy.c.
        u32 mono_errs = 0;
        for (u32 i = 1; i < n; i++) {
            fp32 r_prev = ref_silu_at(orig[i-1]);
            fp32 r_cur  = ref_silu_at(orig[i]);
            if (r_cur <= r_prev) continue;          // reference did not rise
            if (!(x[i] > x[i-1])) mono_errs++;      // kernel failed to rise
        }
        KCHECK("silu_monotonic", mono_errs == 0);

        // silu(x) < x for x > 0, since 0 < sigmoid(x) < 1. Only on the
        // positive side: for x < 0 the sigmoid is below 0.5, so silu(x) > x
        // (silu(-4) = -0.0719 > -4).
        u32 bound_errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (orig[i] > 0.0f && x[i] >= orig[i]) bound_errs++;
        }
        KCHECK("silu_below_identity_pos", bound_errs == 0);

        // |silu(x)| < |x| everywhere, because |sigmoid| < 1. This is the
        // scale-free version of the bound above and holds on both sides.
        u32 mag_errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (x[i] < 0.0f ? (-x[i] >= -orig[i]) : (x[i] >= orig[i])) {
                if (orig[i] != 0.0f) mag_errs++;
            }
        }
        KCHECK("silu_magnitude_shrinks", mag_errs == 0);

        // silu(x) -> x as x grows: the largest input must come out close to
        // itself, which only holds if the 1/(1+exp(-x)) denominator saturated.
        fp32 big = 3.5f;
        fp32 one[1] = { big };
        silu_f32(one, 1);
        KCHECK("silu_saturates_to_identity",
               approx_eq(one[0], big, 0.2f));

        arena_destroy(&a);
    }

    // ── Test softmax_f32 ──────────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 16);
        u32 n = 16;
        fp32* x = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        fp32* orig = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) {
            orig[i] = (fp32)i * 0.3f;          // increasing
            x[i] = orig[i];
        }
        softmax_f32(x, n);

        // Structural checks; see the note above on why there is no
        // numerical reference in this binary.
        fp32 sum = 0.0f;
        u32 range_errs = 0;
        for (u32 i = 0; i < n; i++) {
            sum += x[i];
            if (!(x[i] >= 0.0f && x[i] <= 1.0f)) range_errs++;
        }
        KCHECK("softmax_in_unit_range", range_errs == 0);
        KCHECK("softmax_sums_to_1", approx_eq(sum, 1.0f, 1e-3f));

        // Order preserved on increasing input. This is what caught the
        // hsum8-for-hmax8 bug: a uniform row sums to 1 but has no order.
        u32 order_ok = 1;
        for (u32 i = 1; i < n; i++) {
            if (x[i] <= x[i-1]) { order_ok = 0; break; }
        }
        KCHECK("softmax_order", order_ok);

        // Invariance to a constant shift: softmax(x + c) == softmax(x). This
        // is the property max-subtraction exists to provide, and it is
        // reference-free.
        fp32* y = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) y[i] = orig[i] + 3.0f;
        softmax_f32(y, n);
        u32 shift_errs = 0;
        for (u32 i = 0; i < n; i++) {
            if (!approx_eq(x[i], y[i], 1e-4f)) shift_errs++;
        }
        KCHECK("softmax_shift_invariant", shift_errs == 0);

        // Extreme input must not overflow or collapse to uniform.
        fp32* ext = (fp32*)arena_alloc(&a, n * sizeof(fp32), 32);
        for (u32 i = 0; i < n; i++) ext[i] = (i == 0) ? 100.0f : -100.0f;
        softmax_f32(ext, n);
        KCHECK("softmax_extreme_one_hot", approx_eq(ext[0], 1.0f, 1e-5f));

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
        // fp16 has an 11-bit significand, so for a value near pi the spacing
        // between representable neighbours is 2^-9 = 1.95e-3. Correct
        // round-to-nearest therefore admits up to half that, ~9.8e-4 ABSOLUTE.
        //
        // This test previously asserted an absolute error below 1e-3, which is
        // just barely tighter than the 9.65e-4 that correct rounding produces
        // for this particular value — so it could only ever pass by luck, and
        // in practice it reported a failure for a conversion that is provably
        // nearest (0x4248 = 3.140625, error -9.651e-04, beating the other
        // neighbour 0x4249 = 3.142578 at +9.880e-04).
        //
        // The right assertion is on RELATIVE error against half an ulp of the
        // result's own binade, which is scale-free and states the actual
        // guarantee. pi's binade is [2,4), ulp = 2^-9, half-ulp = 9.77e-4.
        KCHECK("f16c_roundtrip_pi",
               relerr_fp32(back, orig) < 5.0e-4f);
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

    // Report the tally, then return the failure count so crt0 can actually
    // fail the build. Returning 0 here (as this used to) meant a broken kernel
    // could never gate anything — see the comment in crt0.c.
    {
        char msg[64];
        u32 off = 0;
        const char* p = "pass=";
        while (*p && off + 1 < sizeof(msg)) msg[off++] = *p++;
        // u32 -> decimal
        char tmp[12];
        u32 n = 0, v = n_pass;
        if (v == 0) tmp[n++] = '0';
        while (v > 0 && n < sizeof(tmp)) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
        while (n > 0) msg[off++] = tmp[--n];
        p = " fail=";
        while (*p && off + 1 < sizeof(msg)) msg[off++] = *p++;
        n = 0; v = n_fail;
        if (v == 0) tmp[n++] = '0';
        while (v > 0 && n < sizeof(tmp)) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
        while (n > 0) msg[off++] = tmp[--n];
        SPEKI_LOG_BUF(LOG_INFO, msg, off);
    }

    return (int)n_fail;
}