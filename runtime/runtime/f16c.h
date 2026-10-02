// SPDX-License-Identifier: OSL-3.0
// f16c.h — FP16 ↔ FP32 conversion via F16C intrinsics.
//
// Alder Lake exposes F16C (a.k.a. CVT16) which does half↔single conversion
// at one conversion per cycle per lane. We use it to promote FP16 weights
// to FP32 for our matmul accumulator and to demote FP32 activations to
// FP16 for storage.
//
// <immintrin.h> is provided by clang's resource-dir and is header-only —
// it doesn't pull in any libc. We include it directly.

#ifndef SPEKI_F16C_H
#define SPEKI_F16C_H

#include "../crt/raw_syscalls.h"
#include "types.h"

#include <immintrin.h>

// ─── Scalar conversions (FP16 <-> FP32) ────────────────────────────────────

// f16_to_f32_bits(h) — bit-cast FP16 to FP32 by promoting through F16C.
//
// We use the scalar intrinsic _cvtsh_ss which returns the FP32 value
// directly. The FP16 is held in the low 16 bits of a __fp16/int.
static inline fp32 f16_to_f32(fp16 h) {
    return (fp32)_cvtsh_ss((__fp16)h);
}

// f32_to_f16(f) — demote FP32 to FP16 via F16C.
//
// _cvtss_sh takes an FP32 and returns a __fp16 (which on x86 is just a
// 16-bit half). We pack it into our u16 storage.
static inline fp16 f32_to_f16(fp32 f) {
    return (fp16)(u16)_cvtss_sh(f, 0);
}

// ─── SIMD conversions (8x or 16x in one instruction) ──────────────────────

// f16x8_to_f32x8: convert 8 packed FP16 values to 8 FP32 values.
//
// `src` is a pointer to 8 contiguous FP16 (16 bytes total).
// `dst` is a pointer to 8 contiguous FP32 (32 bytes total).
static inline void f16x8_to_f32x8(const fp16* src, fp32* dst) {
    __m128i h = _mm_loadu_si128((const __m128i*)src);
    __m256  f = _mm256_cvtph_ps(h);
    _mm256_storeu_ps(dst, f);
}

// f32x8_to_f16x8: convert 8 packed FP32 values to 8 FP16 values (packed).
//
// `src` is 8 contiguous FP32 (32 bytes).
// `dst` is 8 contiguous FP16 (16 bytes).
static inline void f32x8_to_f16x8(const fp32* src, fp16* dst) {
    __m256  f = _mm256_loadu_ps(src);
    __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)dst, h);
}

// f16x16_to_f32x16: convert 16 packed FP16 (32 bytes) to 16 FP32 (64 bytes).
//
// Two f16x8 conversions in sequence.
static inline void f16x16_to_f32x16(const fp16* src, fp32* dst) {
    f16x8_to_f32x8(src,      dst);
    f16x8_to_f32x8(src + 8,  dst + 8);
}

// f32x16_to_f16x16: convert 16 packed FP32 (64 bytes) to 16 FP16 (32 bytes).
static inline void f32x16_to_f16x16(const fp32* src, fp16* dst) {
    f32x8_to_f16x8(src,      dst);
    f32x8_to_f16x8(src + 8,  dst + 8);
}

#endif // SPEKI_F16C_H