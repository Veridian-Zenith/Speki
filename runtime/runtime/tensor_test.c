// SPDX-License-Identifier: OSL-3.0
// tensor_test.c — exercises the tensor.h API.
//
// This file is linked into speki when the build is run with `just test`.
// It allocates a few tensors, fills them with known values, and prints
// shapes, strides, and per-element samples to stderr.
//
// We do NOT link any external test framework — we run the tests in-process
// and log results via SPEKI_LOG. If a tensor alloc fails we log CRIT and
// return 1 from main (which becomes the exit code of speki).

#include "../crt/raw_syscalls.h"
#include "../crt/raw_log.h"
#include "../crt/raw_alloc.h"
#include "types.h"
#include "tensor.h"

// Test helper: log "PASS <name>" or "FAIL <name> <reason>".
static u32 n_pass = 0;
static u32 n_fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { SPEKI_LOG(LOG_INFO, "PASS " name); n_pass++; } \
    else      { SPEKI_LOG(LOG_CRIT, "FAIL " name); n_fail++; } \
} while (0)

// Format a usize as decimal ASCII into buf (no libc).
__attribute__((no_stack_protector))
static usize fmt_dec(char* buf, usize cap, usize v) {
    char tmp[24];
    usize j = 0;
    if (v == 0) { tmp[j++] = '0'; }
    else {
        while (v && j < sizeof(tmp)) {
            tmp[j++] = (char)('0' + (v % 10));
            v /= 10;
        }
    }
    usize k = 0;
    while (j > 0 && k + 1 < cap) buf[k++] = tmp[--j];
    buf[k] = 0;
    return k;
}

__attribute__((no_stack_protector))
static void log_shape(const char* label, const Shape* s) {
    char buf[128];
    usize off = 0;
    while (*label && off + 1 < sizeof(buf)) buf[off++] = *label++;
    if (off + 1 < sizeof(buf)) buf[off++] = '=';
    for (u32 i = 0; i < s->n; i++) {
        if (off + 1 < sizeof(buf)) buf[off++] = (i == 0) ? '[' : 'x';
        off += fmt_dec(buf + off, sizeof(buf) - off, s->d[i]);
    }
    if (off + 1 < sizeof(buf)) buf[off++] = ']';
    SPEKI_LOG_BUF(LOG_INFO, buf, off);
}

// tensor_test_main: the actual test entry point.
//
// Returns 0 if all checks pass, 1 if any fail.
__attribute__((noinline, used, no_stack_protector))
int tensor_test_main(void) {
    SPEKI_LOG(LOG_INFO, "=== tensor_test ===");
    n_pass = 0;
    n_fail = 0;

    // ── Test 1: 1D FP32 tensor ─────────────────────────────────────────────
    {
        Arena a = arena_create(1 << 20); // 1 MiB
        Shape s = { .n = 1, .d = { 1024 } };
        Tensor t = tensor_alloc(&a, s, DT_F32);
        CHECK("1d_alloc", t.data != NULL);
        CHECK("1d_dtype", t.dtype == DT_F32);
        CHECK("1d_rank", t.rank == 1);
        CHECK("1d_byte_cap", t.byte_cap == 1024 * 4);
        CHECK("1d_stride_b", t.stride_b >= (1024 * 4));  // padded

        // Fill with a known pattern.
        fp32* p = (fp32*)t.data;
        for (u32 i = 0; i < 1024; i++) p[i] = (fp32)i * 0.5f;

        // Spot-check a few elements.
        CHECK("1d_value_0", p[0] == 0.0f);
        CHECK("1d_value_512", p[512] == 256.0f);
        CHECK("1d_value_1023", p[1023] == 511.5f);

        char shape_log[64];
        shape_log[0] = 0;
        log_shape("1d", &s);
        (void)shape_log;

        arena_destroy(&a);
    }

    // ── Test 2: 2D FP32 tensor (transformer-sized) ─────────────────────────
    {
        Arena a = arena_create(2 << 20);
        // 12 layers, hidden_dim=768, head_dim=64 (typical small LLM)
        Shape s = { .n = 2, .d = { 12, 64 * 12 } };  // 12x768
        Tensor t = tensor_alloc(&a, s, DT_F32);
        CHECK("2d_alloc", t.data != NULL);
        CHECK("2d_byte_cap_non_zero", t.byte_cap > 0);
        CHECK("2d_byte_cap_exact", t.byte_cap == 12 * 768 * 4);
        // Row stride must be ≥ row_bytes and 32-byte aligned.
        usize row_bytes = 64 * 12 * 4;  // 3072
        CHECK("2d_stride_at_least_row", t.stride_b >= row_bytes);
        CHECK("2d_stride_aligned_32", (t.stride_b % 32) == 0);

        // Walk a few rows.
        fp32* p = (fp32*)t.data;
        for (u32 row = 0; row < 12; row++) {
            fp32* r = (fp32*)((u8*)p + (usize)row * t.stride_b);
            for (u32 col = 0; col < 64 * 12; col++) {
                r[col] = (fp32)(row * 1000 + col);
            }
        }

        // Verify cross-row access (row stride honored).
        u8* base = (u8*)p;
        fp32* r0 = (fp32*)base;
        fp32* r5p = (fp32*)(base + 5 * t.stride_b);
        CHECK("2d_row_0_first", r0[0] == 0.0f);
        CHECK("2d_row_5_first", r5p[0] == 5000.0f);
        // r5p[i] = 5000 + i; for i = 64*12-1 = 767, value = 5767.
        CHECK("2d_row_5_last",  r5p[64 * 12 - 1] == 5767.0f);

        arena_destroy(&a);
    }

    // ── Test 3: FP16 weight tensor (transformer matmul weight) ──────────────
    {
        // 768x768 FP16 = 1.18 MiB; use a 2 MiB arena to fit comfortably.
        Arena a = arena_create(2 << 20);
        Shape s = { .n = 2, .d = { 768, 768 } };   // square matmul weight
        Tensor t = tensor_alloc(&a, s, DT_F16);
        CHECK("f16_alloc", t.data != NULL);
        CHECK("f16_dtype",  t.dtype == DT_F16);
        CHECK("f16_byte_cap", t.byte_cap == 768 * 768 * 2);  // no padding for power-of-2 row

        // FP16 storage: each element is 2 bytes. We can't easily verify
        // numeric correctness without F16C, but we can confirm the bytes
        // are present.
        u8* p = (u8*)t.data;
        u64 sum = 0;
        for (usize i = 0; i < t.byte_cap; i++) sum += p[i];
        CHECK("f16_zero_sum", sum == 0);  // arena_alloc zero-fills

        arena_destroy(&a);
    }

    // ── Test 4: 3D tensor (sequence, batch, hidden) ─────────────────────────
    {
        // seq=128, batch=4, hidden=768. byte_cap = 128 * (768*4 padded to
        // 32-align = 3072) * 4 = 1.5 MiB. Use a 2 MiB arena.
        Arena a = arena_create(2 << 20);
        Shape s = { .n = 3, .d = { 128, 4, 768 } };  // seq=128, batch=4, hidden=768
        Tensor t = tensor_alloc(&a, s, DT_F32);
        CHECK("3d_alloc", t.data != NULL);
        CHECK("3d_rank",  t.rank == 3);
        // The byte_cap computation: outer-most * padded_stride * inner_dims
        //   = 128 * (768*4=3072 padded to 32-align = 3072) * 4
        //   = 128 * 3072 * 4 = 1,572,864
        CHECK("3d_byte_cap", t.byte_cap == 1572864);

        arena_destroy(&a);
    }

    // ── Test 5: shape_byte_size helper ──────────────────────────────────────
    {
        Shape s1 = { .n = 2, .d = { 100, 200 } };
        CHECK("byte_size_f32", shape_byte_size(&s1, DT_F32) == 100 * 200 * 4);
        CHECK("byte_size_f16", shape_byte_size(&s1, DT_F16) == 100 * 200 * 2);
        CHECK("byte_size_i8",  shape_byte_size(&s1, DT_I8)  == 100 * 200 * 1);
    }

    // Summary
    {
        char buf[64];
        usize off = 0;
        const char* pre = "pass=";
        for (usize i = 0; pre[i] && off + 1 < sizeof(buf); i++) buf[off++] = pre[i];
        off += fmt_dec(buf + off, sizeof(buf) - off, n_pass);
        const char* mid = " fail=";
        for (usize i = 0; mid[i] && off + 1 < sizeof(buf); i++) buf[off++] = mid[i];
        off += fmt_dec(buf + off, sizeof(buf) - off, n_fail);
        SPEKI_LOG_BUF(LOG_INFO, buf, off);
    }

    return (n_fail == 0) ? 0 : 1;
}